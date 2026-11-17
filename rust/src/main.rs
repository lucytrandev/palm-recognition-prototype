use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use palmprint::card::ModelCard;
use palmprint::encoder::Encoder;
use palmprint::preprocess::{load, preprocess, TARGET};
use palmprint::{ops, server};
use std::path::PathBuf;

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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let card = ModelCard::load_for(&cli.model)?;
    let thr = |t: Option<f32>| t.or(card.as_ref().map(|c| c.threshold)).unwrap_or(0.8);
    let db = ops::DbPaths { embeddings: cli.embeddings.clone(), names: cli.names.clone(), ridges: cli.ridges.clone() };
    match &cli.cmd {
        Cmd::Embed { image } => {
            let st = preprocess(&load(image)?).context("preprocessing failed (no palm region)")?;
            let e = Encoder::new(&cli.model, cli.threads)?.embed(&st.normalized, TARGET)?;
            println!("{}", serde_json::json!({"preprocess": st.trace, "embedding": e}));
        }
        Cmd::Match { image, threshold, top_k } => {
            let mut eng = ops::engine(&cli.model, &db, *threshold, *top_k, None, false, cli.threads)?;
            let mut o = eng.run(&std::fs::read(image)?)?;
            o.roi_png.clear();
            println!("{}", serde_json::to_string_pretty(&o)?);
        }
        Cmd::BuildDb { dirs, roi_dir } => {
            let (n, failed) = ops::build_db(&cli.model, dirs, &db, roi_dir.as_deref(), cli.threads, |k, total| {
                if k % 500 == 0 || k == total { eprintln!("[build-db] {k} / {total}"); }
            })?;
            eprintln!("[build-db] {n} embedded, {failed} failed → {} / {} (+ ridge codes {})", db.embeddings.display(), db.names.display(), db.ridges.display());
        }
        Cmd::Eval { block, threshold, split } => {
            let block = block.or(card.as_ref().map(|c| c.identity_block)).unwrap_or(10);
            let threshold = threshold.unwrap_or(thr(None) as f64);
            let split = split.clone().unwrap_or_else(|| if card.is_some() { "test".into() } else { "all".into() });
            println!("{}", serde_json::to_string_pretty(&ops::evaluate(&db, card.as_ref(), &split, block, threshold, cli.threads)?)?);
        }
        Cmd::PairScores { split, out, ridge_shift } => {
            let (n, pairs) = ops::pair_scores(&db, card.as_ref(), split, *ridge_shift, out, cli.threads)?;
            eprintln!("[pair-scores] {n} images, {pairs} pairs, ridge shift {ridge_shift} → {}", out.display());
        }
        Cmd::VerifyModel { tol } => {
            let v = ops::verify_model(&cli.model, cli.threads, *tol)?;
            println!("{}", serde_json::to_string_pretty(&v)?);
            anyhow::ensure!(v["ok"] == true, "model verification failed");
        }
        Cmd::Serve { addr, threshold, top_k, roi_dir } => {
            let engine = ops::engine(&cli.model, &db, *threshold, *top_k, roi_dir.clone(), true, cli.threads)?;
            eprintln!("[serve] {} images in database · http://{addr}/ · API /v1/chat/completions", engine.db.names.len());
            let l = tokio::net::TcpListener::bind(addr.as_str()).await?;
            axum::serve(l, server::router(engine)).await?;
        }
    }
    Ok(())
}
