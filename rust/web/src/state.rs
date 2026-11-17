//! Shared state: the live matching engine, the training pipeline (Python children + Rust steps) and
//! the status every live region renders.

use crate::pipeline::{can_step, Phase};
use anyhow::{bail, ensure, Context, Result};
use palmprint::card::ModelCard;
use palmprint::ops::{self, DbPaths};
use palmprint::server::{Engine, Outcome};
use serde::Serialize;
use serde_json::Value;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::io::AsyncBufReadExt;
use tokio::sync::watch;

pub const MODEL_FILE: &str = "palmprint_encoder_512.onnx";

#[derive(Clone, Debug)]
pub struct Config {
    /// repository root (holds training/), relative to where the app runs
    pub repo: PathBuf,
    /// working folder for trained models, the trained database and intermediate files
    pub work: PathBuf,
    pub python: String,
    pub threads: usize,
    /// default image folders offered in the form (comma-separated in PP_DATA)
    pub data_dirs: Vec<String>,
    /// model and database served before anything is trained
    pub base_model: PathBuf,
    pub base_db: DbPaths,
    pub roi_dir: Option<PathBuf>,
}

impl Config {
    pub fn from_env() -> Self {
        let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
        let repo = PathBuf::from(env("PP_REPO", ".."));
        let base = repo.join("output/embeddings");
        Self {
            work: PathBuf::from(env("PP_WORK", "palmprint_work")),
            python: env("PP_PYTHON", "python3"),
            threads: env("PP_THREADS", "4").parse().unwrap_or(4),
            data_dirs: env("PP_DATA", "").split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
            base_model: PathBuf::from(env("PP_MODEL", "models/palmprint_encoder_512.onnx")),
            base_db: DbPaths::in_dir(&base),
            roi_dir: std::env::var("PP_ROI_DIR").ok().map(PathBuf::from),
            repo,
        }
    }
    pub fn models(&self) -> PathBuf { self.work.join("models") }
    pub fn model(&self) -> PathBuf { self.models().join(MODEL_FILE) }
    pub fn card(&self) -> PathBuf { palmprint::card::card_path(&self.model()) }
    pub fn db(&self) -> DbPaths { DbPaths::in_dir(&self.work.join("db")) }
    pub fn rois(&self) -> PathBuf { self.work.join("data/rois.npz") }
    pub fn pairs(&self) -> PathBuf { self.work.join("data/cal_pairs.npy") }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Row {
    pub scorer: String,
    pub eer: Option<f64>,
    pub tar_at_01: Option<f64>,
    pub tar: Option<f64>,
    pub far: Option<f64>,
    pub mcc: Option<f64>,
    pub rank1: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Results {
    pub test_palms: usize,
    pub test_images: usize,
    pub rows: Vec<Row>,
    pub fusion_weight: Option<f64>,
    pub fusion_threshold: Option<f64>,
    pub verify_max_diff: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Live {
    pub model: String,
    pub images: usize,
    pub threshold: f32,
    pub scoring: String,
    pub trained: bool,
    pub error: Option<String>,
}

#[derive(Clone, Default, Serialize)]
pub struct LastMatch {
    pub file: String,
    pub outcome: Option<Outcome>,
    pub error: Option<String>,
}

#[derive(Clone, Default, Serialize)]
pub struct Status {
    pub phase: Phase,
    pub stage: String,
    pub progress: (usize, usize),
    pub epochs: usize,
    pub epoch: usize,
    pub losses: Vec<f64>,
    pub cal_eers: Vec<f64>,
    pub log: VecDeque<String>,
    pub error: Option<String>,
    pub results: Option<Results>,
    pub live: Live,
    pub last: Option<LastMatch>,
}

#[derive(Clone, Debug)]
pub struct TrainParams {
    pub dirs: Vec<PathBuf>,
    pub epochs: usize,
    pub loss: String,
    pub target_far: f64,
    pub pretrained: bool,
}

pub struct Inner {
    pub cfg: Config,
    status: Mutex<Status>,
    engine: Mutex<Option<Engine>>,
    status_tx: watch::Sender<u64>,
    job: Mutex<Option<tokio::task::AbortHandle>>,
    child_pid: Mutex<Option<u32>>,
    demo: Mutex<Vec<PathBuf>>,
}

#[derive(Clone)]
pub struct App(pub Arc<Inner>);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl App {
    pub fn new(cfg: Config) -> Self {
        let app = App(Arc::new(Inner {
            cfg,
            status: Mutex::new(Status::default()),
            engine: Mutex::new(None),
            status_tx: watch::channel(0).0,
            job: Mutex::new(None),
            child_pid: Mutex::new(None),
            demo: Mutex::new(vec![]),
        }));
        // serve a previously trained model if one is complete, else the committed one
        let cfg = &app.0.cfg;
        let trained = cfg.model().exists() && cfg.db().ridges.exists() && ModelCard::load_for(&cfg.model()).ok().flatten().is_some_and(|c| c.fusion.is_some());
        let r = if trained { app.load_engine(&cfg.model(), &cfg.db(), true) } else { app.load_engine(&cfg.base_model, &cfg.base_db, false) };
        if let Err(e) = r {
            app.update(|s| s.live.error = Some(format!("{e:#}")));
        }
        app
    }

    pub fn status(&self) -> Status {
        lock(&self.0.status).clone()
    }
    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.0.status_tx.subscribe()
    }
    fn update(&self, f: impl FnOnce(&mut Status)) {
        f(&mut lock(&self.0.status));
        self.0.status_tx.send_modify(|v| *v += 1);
    }
    fn log(&self, msg: impl Into<String>) {
        let msg = msg.into();
        self.update(|s| {
            s.log.push_back(msg);
            while s.log.len() > 60 {
                s.log.pop_front();
            }
        });
    }
    /// Phase changes go through the Lean-mirrored transition relation.
    fn set_phase(&self, to: Phase, stage: impl Into<String>) -> Result<()> {
        let stage = stage.into();
        let mut st = lock(&self.0.status);
        ensure!(can_step(st.phase, to), "cannot go from {:?} to {:?}", st.phase, to);
        st.phase = to;
        st.stage = stage;
        st.progress = (0, 0);
        drop(st);
        self.0.status_tx.send_modify(|v| *v += 1);
        Ok(())
    }

    fn load_engine(&self, model: &Path, db: &DbPaths, trained: bool) -> Result<()> {
        let e = ops::engine(model, db, None, 5, self.0.cfg.roi_dir.clone(), true, self.0.cfg.threads)?;
        let live = Live {
            model: model.display().to_string(),
            images: e.db.names.len(),
            threshold: e.fusion.as_ref().map(|f| f.threshold).unwrap_or(e.threshold),
            scoring: match &e.fusion {
                Some(f) => format!("CNN + palm ridges (w = {:.2})", f.weight),
                None if e.ridges.is_some() => "CNN (ridge distances shown, fusion not calibrated)".into(),
                None => "CNN only".into(),
            },
            trained,
            error: None,
        };
        *lock(&self.0.engine) = Some(e);
        self.update(|s| s.live = live);
        Ok(())
    }

    // ── inference ──────────────────────────────────────────────────────────────
    /// Match one uploaded image; the result is also kept for the results panel.
    pub fn match_image(&self, file: String, bytes: &[u8]) -> Result<Outcome> {
        let r = {
            let mut g = lock(&self.0.engine);
            let e = g.as_mut().ok_or_else(|| anyhow::anyhow!("no model loaded"))?;
            e.run(bytes)
        };
        match r {
            Ok(o) => {
                let o2 = o.clone();
                self.update(|s| s.last = Some(LastMatch { file, outcome: Some(o2), error: None }));
                Ok(o)
            }
            Err(e) => {
                let m = format!("{e:#}");
                self.update(|s| s.last = Some(LastMatch { file, outcome: None, error: Some(m.clone()) }));
                bail!(m)
            }
        }
    }

    // ── demo data ──────────────────────────────────────────────────────────────
    /// Synthetic hands (training/synth.py) for trying the pipeline without a dataset.
    pub async fn make_demo_data(&self) -> Result<String> {
        ensure!(!lock(&self.0.status).phase.running(), "wait for the current run to finish");
        let out = self.0.cfg.work.join("synth").canonicalize().unwrap_or_else(|_| std::env::current_dir().unwrap_or_default().join(self.0.cfg.work.join("synth")));
        let st = tokio::process::Command::new(&self.0.cfg.python)
            .arg(self.0.cfg.repo.join("training/synth.py"))
            .arg(&out)
            .args(["--palms", "60"])
            .status()
            .await
            .context("running training/synth.py")?;
        ensure!(st.success(), "synth.py failed ({st})");
        *lock(&self.0.demo) = vec![out.join("session1"), out.join("session2")];
        let dirs = format!("{},{}", out.join("session1").display(), out.join("session2").display());
        self.log(format!("demo data: 60 synthetic palms × 2 sessions × 10 images in {}", out.display()));
        Ok(dirs)
    }

    /// Folders of the last demo data (empty if none was made in this run of the app).
    pub fn demo_dirs(&self) -> Vec<PathBuf> {
        lock(&self.0.demo).clone()
    }

    // ── training pipeline ──────────────────────────────────────────────────────
    pub fn start_training(&self, p: TrainParams) -> Result<String> {
        ensure!(!p.dirs.is_empty(), "give at least one image folder");
        for d in &p.dirs {
            ensure!(d.is_dir(), "{} is not a folder", d.display());
        }
        ensure!((1..=500).contains(&p.epochs), "epochs must be 1–500");
        ensure!(p.loss == "supcon" || p.loss == "simclr", "loss must be supcon or simclr");
        ensure!(p.target_far > 0.0 && p.target_far < 0.5, "target FAR must be between 0 and 0.5");
        self.set_phase(Phase::Preparing, "cutting palm regions with the shared preprocessing")?;
        self.update(|s| {
            let log = std::mem::take(&mut s.log);
            let (live, last) = (s.live.clone(), s.last.take());
            *s = Status { phase: Phase::Preparing, stage: "cutting palm regions".into(), epochs: p.epochs, log, live, last, ..Default::default() };
        });
        let app = self.clone();
        let msg = format!("training {} epochs ({}, SAM) on {} folder(s)", p.epochs, p.loss, p.dirs.len());
        let h = tokio::spawn(async move {
            if let Err(e) = app.run(p).await {
                app.log(format!("error: {e:#}"));
                let _ = app.set_phase(Phase::Failed, "failed");
                app.update(|s| s.error = Some(format!("{e:#}")));
            }
            *lock(&app.0.job) = None;
        });
        *lock(&self.0.job) = Some(h.abort_handle());
        Ok(msg)
    }

    pub fn cancel(&self) -> bool {
        let Some(h) = lock(&self.0.job).take() else { return false };
        h.abort();
        if let Some(pid) = lock(&self.0.child_pid).take() {
            let _ = std::process::Command::new("kill").arg(pid.to_string()).status();
        }
        let _ = self.set_phase(Phase::Failed, "cancelled");
        self.update(|s| s.error = Some("cancelled".into()));
        true
    }

    /// Run a Python script from the repository, handing each JSON progress line to `on`.
    async fn python(&self, script: &str, args: &[String], mut on: impl FnMut(&App, &Value)) -> Result<()> {
        let cfg = &self.0.cfg;
        let mut child = tokio::process::Command::new(&cfg.python)
            .arg("-u")
            .arg(cfg.repo.join(script))
            .args(args)
            .env("OMP_NUM_THREADS", cfg.threads.to_string())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("starting {} {script}", cfg.python))?;
        *lock(&self.0.child_pid) = child.id();
        let mut out = tokio::io::BufReader::new(child.stdout.take().unwrap()).lines();
        let err = child.stderr.take().unwrap();
        let tail = tokio::spawn(async move {
            let mut l = tokio::io::BufReader::new(err).lines();
            let mut t = VecDeque::new();
            while let Ok(Some(x)) = l.next_line().await {
                if x.contains("Warning") || x.trim().is_empty() {
                    continue;
                }
                t.push_back(x);
                if t.len() > 15 {
                    t.pop_front();
                }
            }
            t.into_iter().collect::<Vec<_>>().join("\n")
        });
        while let Some(line) = out.next_line().await? {
            match serde_json::from_str::<Value>(&line) {
                Ok(ev) => on(self, &ev),
                Err(_) if !line.trim().is_empty() => self.log(line),
                Err(_) => {}
            }
        }
        let st = child.wait().await?;
        *lock(&self.0.child_pid) = None;
        if !st.success() {
            bail!("{script} exited ({st}):\n{}", tail.await.unwrap_or_default());
        }
        Ok(())
    }

    async fn blocking<T: Send + 'static>(&self, f: impl FnOnce(App) -> Result<T> + Send + 'static) -> Result<T> {
        let app = self.clone();
        tokio::task::spawn_blocking(move || f(app)).await?
    }

    async fn run(&self, p: TrainParams) -> Result<()> {
        let cfg = self.0.cfg.clone();
        std::fs::create_dir_all(cfg.models())?;
        std::fs::create_dir_all(cfg.work.join("data"))?;
        let abs = |x: &Path| std::path::absolute(x).unwrap_or_else(|_| x.to_path_buf()).display().to_string();

        // 1 · prepare (Python, same preprocessing as Rust)
        let mut args: Vec<String> = p.dirs.iter().map(|d| abs(d)).collect();
        args.extend(["-o".into(), abs(&cfg.rois())]);
        self.python("training/prepare.py", &args, |a, ev| match ev["event"].as_str() {
            Some("prepare") => a.update(|s| s.progress = (ev["done"].as_u64().unwrap_or(0) as usize, 0)),
            Some("prepared") => a.log(format!("prepared {} images of {} palms ({} failed)", ev["images"], ev["palms"], ev["failed"])),
            _ => {}
        })
        .await?;

        // 2 · train (Python, SAM)
        self.set_phase(Phase::Training, format!("{} epochs, {} loss, SAM", p.epochs, p.loss))?;
        let mut args = vec![abs(&cfg.rois()), "--out".into(), abs(&cfg.models()), "--epochs".into(), p.epochs.to_string(), "--loss".into(), p.loss.clone(), "--target-far".into(), p.target_far.to_string()];
        if !p.pretrained {
            args.push("--no-pretrained".into());
        }
        self.python("training/train.py", &args, |a, ev| match ev["event"].as_str() {
            Some("data") => a.log(format!("{} images · palms train/cal/test {}/{}/{} · {}", ev["images"], ev["train_palms"], ev["cal_palms"], ev["test_palms"], ev["device"].as_str().unwrap_or("?"))),
            Some("warning") => a.log(ev["msg"].as_str().unwrap_or("").to_string()),
            Some("epoch") => {
                a.update(|s| {
                    s.epoch = ev["epoch"].as_u64().unwrap_or(0) as usize;
                    s.progress = (s.epoch, s.epochs);
                    if let Some(l) = ev["loss"].as_f64() { s.losses.push(l) }
                    if let Some(e) = ev["cal_eer"].as_f64() { s.cal_eers.push(e) }
                });
                a.log(format!("epoch {}/{}  loss {}  calibration EER {}", ev["epoch"], ev["of"], ev["loss"], ev["cal_eer"]));
            }
            Some("calibrated") => a.log(format!("Python, test palms: threshold {:.4} → TAR {:.3} FAR {:.5} MCC {} · EER {:.4}", ev["threshold"].as_f64().unwrap_or(0.0), ev["test_tar"].as_f64().unwrap_or(0.0), ev["test_far"].as_f64().unwrap_or(0.0), ev["test_mcc"], ev["test_eer"].as_f64().unwrap_or(0.0))),
            Some("exported") => a.log(format!("exported ONNX (max |Δ| vs PyTorch {})", ev["max_abs_vs_pytorch"])),
            _ => {}
        })
        .await?;

        // 3 · verify (Rust loads exactly what Python exported)
        self.set_phase(Phase::Verifying, "SHA-256 and embedding parity")?;
        let v = self.blocking(|a| ops::verify_model(&a.0.cfg.model(), a.0.cfg.threads, 1e-4)).await?;
        ensure!(v["ok"] == true, "model verification failed: {v}");
        let max_diff = v["max_abs_diff"].as_f64();
        self.log(format!("verified: SHA-256 matches the card, max |Δ| {}", v["max_abs_diff"]));

        // 4 · index (Rust: embeddings + ridge codes)
        self.set_phase(Phase::Indexing, "embedding every image and coding its palm lines")?;
        let dirs = p.dirs.clone();
        let (n, failed) = self
            .blocking(move |a| {
                let a2 = a.clone();
                ops::build_db(&a.0.cfg.model(), &dirs, &a.0.cfg.db(), None, a.0.cfg.threads, move |k, total| {
                    if k % 20 == 0 || k == total {
                        a2.update(|s| s.progress = (k, total));
                    }
                })
            })
            .await?;
        self.log(format!("database: {n} images ({failed} failed preprocessing)"));

        // 5 · fusion (Rust pair scores → Python chooses weight + threshold)
        self.set_phase(Phase::Fusing, "scoring calibration pairs, then choosing the weight")?;
        let (imgs, pairs) = self
            .blocking(|a| {
                let card = ModelCard::load_for(&a.0.cfg.model())?;
                ops::pair_scores(&a.0.cfg.db(), card.as_ref(), "cal", 1, &a.0.cfg.pairs(), a.0.cfg.threads)
            })
            .await?;
        self.log(format!("calibration pairs: {pairs} from {imgs} images"));
        self.python("training/calibrate_fusion.py", &[abs(&cfg.pairs()), abs(&cfg.card())], |a, ev| {
            if ev["event"] == "fusion" {
                a.log(format!("fusion: w = {} · threshold {:.4} · calibration EER {:.4} (CNN alone {:.4}, ridges alone {:.4})", ev["weight"], ev["threshold"].as_f64().unwrap_or(0.0), ev["cal_eer"].as_f64().unwrap_or(0.0), ev["cal_eer_cnn"].as_f64().unwrap_or(0.0), ev["cal_eer_ridges"].as_f64().unwrap_or(0.0)));
            }
        })
        .await?;

        // 6 · evaluate on unseen palms (Rust)
        self.set_phase(Phase::Evaluating, "CNN, ridges and fused on the test palms")?;
        let rep = self
            .blocking(|a| {
                let card = ModelCard::load_for(&a.0.cfg.model())?.ok_or_else(|| anyhow::anyhow!("model card missing"))?;
                ops::evaluate(&a.0.cfg.db(), Some(&card), "test", card.identity_block, card.threshold as f64, a.0.cfg.threads)
            })
            .await?;
        let results = summarize(&rep, max_diff);

        // Lean `ready_only_after_eval`: swap the live engine only on Evaluating → Ready
        let (model, db) = (cfg.model(), cfg.db());
        self.blocking(move |a| a.load_engine(&model, &db, true)).await?;
        self.set_phase(Phase::Ready, "")?;
        let line = results.rows.iter().map(|r| format!("{} MCC {}", r.scorer, r.mcc.map_or("n/a".into(), |m| format!("{m:.3}")))).collect::<Vec<_>>().join(" · ");
        self.update(|s| s.results = Some(results));
        self.log(format!("model live · test palms: {line}"));
        Ok(())
    }
}

fn summarize(rep: &Value, verify_max_diff: Option<f64>) -> Results {
    let row = |name: &str, v: &Value| Row {
        scorer: name.into(),
        eer: v["eer"].as_f64(),
        tar_at_01: v["tar_at_far"].as_array().and_then(|a| a.iter().find(|x| x["far"].as_f64() == Some(1e-3))).and_then(|x| x["tar"].as_f64()),
        tar: v["at_threshold"]["tar"].as_f64(),
        far: v["at_threshold"]["far"].as_f64(),
        mcc: v["at_threshold"]["mcc"].as_f64(),
        rank1: v["rank1_cross_session"].as_f64(),
    };
    let mut rows = vec![row("CNN", &rep["cnn_exact"]), row("palm ridges", &rep["ridge"])];
    if rep["fused"].is_object() {
        rows.push(row("CNN + ridges", &rep["fused"]));
    }
    Results {
        test_palms: rep["report"]["identities"].as_u64().unwrap_or(0) as usize,
        test_images: rep["report"]["images"].as_u64().unwrap_or(0) as usize,
        rows,
        fusion_weight: rep["fused"]["fusion"]["weight"].as_f64(),
        fusion_threshold: rep["fused"]["fusion"]["threshold"].as_f64(),
        verify_max_diff,
    }
}
