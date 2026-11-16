use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use palmprint::card::{card_path, sha256_file, ModelCard};
use palmprint::encoder::Encoder;
use palmprint::matcher::Database;
use palmprint::preprocess::{load, preprocess, TARGET};
use palmprint::{eval, npy, server};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "palmprint", version, about = "Rust inference for the palmprint recognition prototype")]
struct Cli {
    #[arg(long, global = true, default_value = "models/palmprint_encoder_512.onnx")]
    model: PathBuf,
    #[arg(long, global = true, default_value = "../output/embeddings/all_embeddings.npy")]
    embeddings: PathBuf,
    #[arg(long, global = true, default_value = "../output/embeddings/image_names.npy")]
    names: PathBuf,
    #[arg(long, global = true, default_value_t = 4)]
    threads: usize,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Preprocess + embed one image; prints the 512-d embedding as JSON
    Embed { image: PathBuf },
    /// Match one image against the database (top-k, threshold — same rule as the Streamlit app)
    Match {
        image: PathBuf,
        /// default: the model card's calibrated threshold, else 0.80 (as app.py)
        #[arg(long)]
        threshold: Option<f32>,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
    /// Rebuild the database from image folders (replaces batch_preprocess.py + extract_features.py).
    /// Names are `<folder>_<file stem>`, as in the prototype.
    BuildDb {
        dirs: Vec<PathBuf>,
        /// also save each preprocessed ROI as <name>.npy here (for UI thumbnails)
        #[arg(long)]
        roi_dir: Option<PathBuf>,
    },
    /// EER, TAR@FAR, MCC at a threshold, rank-1 (identity from names: (number − 1) / block)
    Eval {
        /// default: the model card's identity block, else 10
        #[arg(long)]
        block: Option<usize>,
        /// default: the model card's calibrated threshold, else 0.80
        #[arg(long)]
        threshold: Option<f64>,
        /// restrict to the card's palms: test | cal | train | all (default: test when a card exists)
        #[arg(long)]
        split: Option<String>,
    },
    /// Check the ONNX file against the Python trainer's export: SHA-256 in the card, and embeddings of
    /// the parity inputs vs onnxruntime-Python (<model>.parity_input.npy / .parity_embedding.npy)
    VerifyModel {
        #[arg(long, default_value_t = 1e-4)]
        tol: f32,
    },
    /// Web UI + OpenAI-compatible API
    Serve {
        #[arg(long, default_value = "127.0.0.1:8501")]
        addr: String,
        /// default: the model card's calibrated threshold, else 0.80
        #[arg(long)]
        threshold: Option<f32>,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        /// folder of preprocessed ROIs (<name>.npy) for match thumbnails, e.g. ../dataset/preprocessed_images
        #[arg(long)]
        roi_dir: Option<PathBuf>,
    },
}

fn images_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(), Some("tif" | "tiff" | "png" | "jpg" | "jpeg")))
        .collect();
    v.sort();
    Ok(v)
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let card = ModelCard::load_for(&cli.model)?;
    let thr = |t: Option<f32>| t.or(card.as_ref().map(|c| c.threshold)).unwrap_or(0.8);
    match &cli.cmd {
        Cmd::Embed { image } => {
            let st = preprocess(&load(image)?).context("preprocessing failed (no palm region)")?;
            let e = Encoder::new(&cli.model, cli.threads)?.embed(&st.normalized, TARGET)?;
            println!("{}", serde_json::json!({"preprocess": st.trace, "embedding": e}));
        }
        Cmd::Match { image, threshold, top_k } => {
            let st = preprocess(&load(image)?).context("preprocessing failed (no palm region)")?;
            let e = Encoder::new(&cli.model, cli.threads)?.embed(&st.normalized, TARGET)?;
            let db = Database::load(&cli.embeddings, &cli.names)?;
            println!("{}", serde_json::to_string_pretty(&db.best_matches(&e, thr(*threshold), *top_k))?);
        }
        Cmd::BuildDb { dirs, roi_dir } => {
            let mut enc = Encoder::new(&cli.model, cli.threads)?;
            if let Some(d) = roi_dir {
                std::fs::create_dir_all(d)?;
            }
            let (mut names, mut data, mut failed) = (vec![], vec![], 0usize);
            for dir in dirs {
                let folder = dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                for p in images_in(dir)? {
                    let name = format!("{folder}_{}", p.file_stem().unwrap().to_string_lossy());
                    match load(&p).ok().and_then(|i| preprocess(&i)) {
                        Some(st) => {
                            if let Some(d) = roi_dir {
                                npy::write_f32(&d.join(format!("{name}.npy")), TARGET, TARGET, &st.normalized)?;
                            }
                            data.extend(enc.embed(&st.normalized, TARGET)?);
                            names.push(name);
                        }
                        None => {
                            failed += 1;
                            eprintln!("preprocessing failed: {}", p.display());
                        }
                    }
                    if names.len() % 500 == 0 && !names.is_empty() {
                        eprintln!("[build-db] {} embedded", names.len());
                    }
                }
            }
            npy::write_f32(&cli.embeddings, names.len(), palmprint::encoder::EMBED_DIM, &data)?;
            npy::write_strings(&cli.names, &names)?;
            eprintln!("[build-db] {} embedded, {failed} failed → {} / {}", names.len(), cli.embeddings.display(), cli.names.display());
        }
        Cmd::Eval { block, threshold, split } => {
            let block = block.or(card.as_ref().map(|c| c.identity_block)).unwrap_or(10);
            let threshold = threshold.unwrap_or(thr(None) as f64);
            let db = Database::load(&cli.embeddings, &cli.names)?;
            let split = split.clone().unwrap_or_else(|| if card.is_some() { "test".into() } else { "all".into() });
            let db = match (split.as_str(), &card) {
                ("all", _) => db,
                (s, Some(c)) => {
                    let ids: std::collections::HashSet<usize> = c.ids(s).ok_or_else(|| anyhow::anyhow!("--split must be test, cal, train or all"))?.iter().copied().collect();
                    db.subset(|_, n| n.rsplit_once('_').and_then(|(_, k)| k.parse::<usize>().ok()).is_some_and(|k| k >= 1 && ids.contains(&((k - 1) / block))))
                }
                (_, None) => anyhow::bail!("--split needs a model card ({})", card_path(&cli.model).display()),
            };
            anyhow::ensure!(db.names.len() >= 2, "fewer than 2 images in split {split}");
            let r = eval::evaluate(&db, block, threshold)?;
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({"split": split, "report": r}))?);
        }
        Cmd::VerifyModel { tol } => {
            let c = card.as_ref().ok_or_else(|| anyhow::anyhow!("no model card at {}", card_path(&cli.model).display()))?;
            let sha = sha256_file(&cli.model)?;
            let (n, len, x) = npy::read_f32(&cli.model.with_extension("parity_input.npy"))?;
            let (m, dim, want) = npy::read_f32(&cli.model.with_extension("parity_embedding.npy"))?;
            anyhow::ensure!(n == m && len == TARGET * TARGET && dim == palmprint::encoder::EMBED_DIM, "parity files have unexpected shapes");
            let mut enc = Encoder::new(&cli.model, cli.threads)?;
            let mut worst = 0f32;
            for i in 0..n {
                let got = enc.embed(&x[i * len..(i + 1) * len], TARGET)?;
                worst = got.iter().zip(&want[i * dim..(i + 1) * dim]).map(|(a, b)| (a - b).abs()).fold(worst, f32::max);
            }
            let ok = sha == c.onnx_sha256 && worst <= *tol;
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                "sha256_matches_card": sha == c.onnx_sha256, "parity_inputs": n, "max_abs_diff": worst, "tolerance": tol,
                "threshold": c.threshold, "target_far": c.target_far, "test_metrics": c.metrics.get("test"), "ok": ok,
            }))?);
            anyhow::ensure!(ok, "model verification failed");
        }
        Cmd::Serve { addr, threshold, top_k, roi_dir } => {
            let engine = server::Engine {
                encoder: Encoder::new(&cli.model, cli.threads)?,
                db: Database::load(&cli.embeddings, &cli.names)?,
                roi_dir: roi_dir.clone(),
                threshold: thr(*threshold),
                top_k: *top_k,
            };
            eprintln!("[serve] {} images in database · http://{addr}/ · API /v1/chat/completions", engine.db.names.len());
            let l = tokio::net::TcpListener::bind(addr.as_str()).await?;
            axum::serve(l, server::router(engine)).await?;
        }
    }
    Ok(())
}
