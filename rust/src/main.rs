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
    /// palm-ridge (CompCode) codes, row-aligned with the embeddings; written by build-db
    #[arg(long, global = true, default_value = "../output/embeddings/ridge_codes.npy")]
    ridges: PathBuf,
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
    /// CNN cosine and ridge distance for every pair of a split's images → <out>.npy rows
    /// [cosine, ridge_distance, genuine]; training/calibrate_fusion.py picks the fusion from these
    PairScores {
        #[arg(long, default_value = "cal")]
        split: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1)]
        ridge_shift: isize,
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
            let mut eng = engine(&cli, card.as_ref(), thr(*threshold), *top_k, None, false)?;
            let mut o = eng.run(&std::fs::read(image)?)?;
            o.roi_png.clear();
            println!("{}", serde_json::to_string_pretty(&o)?);
        }
        Cmd::BuildDb { dirs, roi_dir } => {
            let mut enc = Encoder::new(&cli.model, cli.threads)?;
            if let Some(d) = roi_dir {
                std::fs::create_dir_all(d)?;
            }
            let (mut names, mut data, mut failed) = (vec![], vec![], 0usize);
            let rc = palmprint::ridges::RidgeCoder::new(1);
            let mut codes = vec![];
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
                            codes.push(rc.code(&st.resized));
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
            palmprint::ridges::RidgeDb::save(&cli.ridges, &codes)?;
            eprintln!("[build-db] {} embedded, {failed} failed → {} / {} (+ ridge codes {})", names.len(), cli.embeddings.display(), cli.names.display(), cli.ridges.display());
        }
        Cmd::Eval { block, threshold, split } => {
            let block = block.or(card.as_ref().map(|c| c.identity_block)).unwrap_or(10);
            let threshold = threshold.unwrap_or(thr(None) as f64);
            let full = Database::load(&cli.embeddings, &cli.names)?;
            let split = split.clone().unwrap_or_else(|| if card.is_some() { "test".into() } else { "all".into() });
            let rows = split_rows(&full, card.as_ref(), &split, block, &cli.model)?;
            let db = full.subset(|i, _| rows.binary_search(&i).is_ok());
            anyhow::ensure!(db.names.len() >= 2, "fewer than 2 images in split {split}");
            let r = eval::evaluate(&db, block, threshold)?;
            let mut out = serde_json::json!({"split": split, "report": r});
            if cli.ridges.exists() {
                // CNN alone, ridges alone and fused, exact, on the same pairs
                let ridges = palmprint::ridges::RidgeDb::load(&cli.ridges)?;
                anyhow::ensure!(ridges.codes.len() == full.names.len(), "ridge codes ({}) and embeddings ({}) differ in count; rerun build-db", ridges.codes.len(), full.names.len());
                let fusion = card.as_ref().and_then(|c| c.fusion.clone());
                let rc = palmprint::ridges::RidgeCoder::new(fusion.as_ref().map(|f| f.ridge_shift).unwrap_or(1));
                let p = palmprint::fusion::pairs(&full, &ridges, &rc, &rows, block, cli.threads);
                let names: Vec<String> = rows.iter().map(|&r| full.names[r].clone()).collect();
                let ridge_sim: Vec<f32> = p.ridge.iter().map(|d| 1.0 - d).collect();
                out["cnn_exact"] = palmprint::fusion::exact(&p.cos, &p, &names, block, Some(threshold as f32));
                out["ridge"] = palmprint::fusion::exact(&ridge_sim, &p, &names, block, None);
                out["fused"] = match &fusion {
                    Some(f) => {
                        let mut v = palmprint::fusion::exact(&palmprint::fusion::fused_scores(&p, f.weight), &p, &names, block, Some(f.threshold));
                        v["fusion"] = serde_json::json!(f);
                        v
                    }
                    None => serde_json::json!("no fusion in the model card: run training/calibrate_fusion.py"),
                };
            }
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        Cmd::PairScores { split, out, ridge_shift } => {
            let block = card.as_ref().map(|c| c.identity_block).unwrap_or(10);
            let full = Database::load(&cli.embeddings, &cli.names)?;
            let rows = split_rows(&full, card.as_ref(), split, block, &cli.model)?;
            let ridges = palmprint::ridges::RidgeDb::load(&cli.ridges)?;
            let p = palmprint::fusion::pairs(&full, &ridges, &palmprint::ridges::RidgeCoder::new(*ridge_shift), &rows, block, cli.threads);
            let data: Vec<f32> = (0..p.cos.len()).flat_map(|k| [p.cos[k], p.ridge[k], p.genuine[k] as u8 as f32]).collect();
            npy::write_f32(out, p.cos.len(), 3, &data)?;
            eprintln!("[pair-scores] {} images, {} pairs ({} genuine), ridge shift {ridge_shift} → {}", rows.len(), p.cos.len(), p.genuine.iter().filter(|&&g| g).count(), out.display());
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
            let engine = engine(&cli, card.as_ref(), thr(*threshold), *top_k, roi_dir.clone(), true)?;
            eprintln!("[serve] {} images in database · http://{addr}/ · API /v1/chat/completions", engine.db.names.len());
            let l = tokio::net::TcpListener::bind(addr.as_str()).await?;
            axum::serve(l, server::router(engine)).await?;
        }
    }
    Ok(())
}

/// Database rows of a split (sorted), from the model card's palm ids.
fn split_rows(db: &Database, card: Option<&ModelCard>, split: &str, block: usize, model: &Path) -> Result<Vec<usize>> {
    let rows: Vec<usize> = match (split, card) {
        ("all", _) => (0..db.names.len()).collect(),
        (s, Some(c)) => {
            let ids: std::collections::HashSet<usize> = c.ids(s).ok_or_else(|| anyhow::anyhow!("--split must be test, cal, train or all"))?.iter().copied().collect();
            (0..db.names.len()).filter(|&i| palmprint::fusion::identity(&db.names[i], block).is_some_and(|(_, k)| ids.contains(&k))).collect()
        }
        (_, None) => anyhow::bail!("--split needs a model card ({})", card_path(model).display()),
    };
    anyhow::ensure!(rows.len() >= 2, "fewer than 2 images in split {split}");
    Ok(rows)
}

/// Inference engine: CNN, plus palm ridges when the codes exist, plus faces when asked.
fn engine(cli: &Cli, card: Option<&ModelCard>, threshold: f32, top_k: usize, roi_dir: Option<PathBuf>, faces: bool) -> Result<server::Engine> {
    let db = Database::load(&cli.embeddings, &cli.names)?;
    let fusion = card.and_then(|c| c.fusion.clone());
    let ridges = if cli.ridges.exists() {
        let r = palmprint::ridges::RidgeDb::load(&cli.ridges)?;
        anyhow::ensure!(r.codes.len() == db.names.len(), "ridge codes and embeddings differ in count; rerun build-db");
        Some((r, palmprint::ridges::RidgeCoder::new(fusion.as_ref().map(|f| f.ridge_shift).unwrap_or(1))))
    } else {
        None
    };
    Ok(server::Engine {
        encoder: Encoder::new(&cli.model, cli.threads)?,
        index: db.names.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect(),
        db,
        roi_dir,
        threshold,
        top_k,
        ridges,
        fusion,
        faces: if faces { Some(palmprint::faces::Faces::new(cli.threads)?) } else { None },
    })
}
