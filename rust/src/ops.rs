//! The CLI's operations as library calls, shared by `palmprint` and the `palmprint-web` app:
//! build the database, verify a trained model, split rows, evaluate (CNN / ridges / fused), pair scores.

use crate::card::{card_path, sha256_file, ModelCard};
use crate::encoder::{Encoder, EMBED_DIM};
use crate::matcher::Database;
use crate::preprocess::{load, preprocess, TARGET};
use crate::ridges::{RidgeCoder, RidgeDb};
use crate::{eval, fusion, npy, server};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Where a database lives: embeddings, names and ridge codes, row-aligned.
#[derive(Clone, Debug)]
pub struct DbPaths {
    pub embeddings: PathBuf,
    pub names: PathBuf,
    pub ridges: PathBuf,
}

impl DbPaths {
    pub fn in_dir(dir: &Path) -> Self {
        Self { embeddings: dir.join("all_embeddings.npy"), names: dir.join("image_names.npy"), ridges: dir.join("ridge_codes.npy") }
    }
}

pub fn images_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(), Some("tif" | "tiff" | "png" | "jpg" | "jpeg")))
        .collect();
    v.sort();
    Ok(v)
}

/// (images embedded, images that failed preprocessing)
pub fn build_db(model: &Path, dirs: &[PathBuf], db: &DbPaths, roi_dir: Option<&Path>, threads: usize, mut progress: impl FnMut(usize, usize)) -> Result<(usize, usize)> {
    let mut enc = Encoder::new(model, threads)?;
    if let Some(d) = roi_dir {
        std::fs::create_dir_all(d)?;
    }
    let files: Vec<(String, PathBuf)> = dirs
        .iter()
        .map(|dir| {
            let folder = dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            Ok(images_in(dir)?.into_iter().map(move |p| (format!("{folder}_{}", p.file_stem().unwrap().to_string_lossy()), p)))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    let (mut names, mut data, mut codes, mut failed) = (vec![], vec![], vec![], 0usize);
    let rc = RidgeCoder::new(1);
    for (k, (name, p)) in files.iter().enumerate() {
        match load(p).ok().and_then(|i| preprocess(&i)) {
            Some(st) => {
                if let Some(d) = roi_dir {
                    npy::write_f32(&d.join(format!("{name}.npy")), TARGET, TARGET, &st.normalized)?;
                }
                data.extend(enc.embed(&st.normalized, TARGET)?);
                codes.push(rc.code(&st.resized));
                names.push(name.clone());
            }
            None => failed += 1,
        }
        progress(k + 1, files.len());
    }
    ensure!(!names.is_empty(), "no image could be preprocessed");
    for p in [&db.embeddings, &db.names, &db.ridges] {
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
    }
    npy::write_f32(&db.embeddings, names.len(), EMBED_DIM, &data)?;
    npy::write_strings(&db.names, &names)?;
    RidgeDb::save(&db.ridges, &codes)?;
    Ok((names.len(), failed))
}

/// SHA-256 against the card and embeddings of the parity inputs vs the Python export.
pub fn verify_model(model: &Path, threads: usize, tol: f32) -> Result<Value> {
    let c = ModelCard::load_for(model)?.ok_or_else(|| anyhow::anyhow!("no model card at {}", card_path(model).display()))?;
    let sha = sha256_file(model)?;
    let (n, len, x) = npy::read_f32(&model.with_extension("parity_input.npy"))?;
    let (m, dim, want) = npy::read_f32(&model.with_extension("parity_embedding.npy"))?;
    ensure!(n == m && len == TARGET * TARGET && dim == EMBED_DIM, "parity files have unexpected shapes");
    let mut enc = Encoder::new(model, threads)?;
    let mut worst = 0f32;
    for i in 0..n {
        let got = enc.embed(&x[i * len..(i + 1) * len], TARGET)?;
        worst = got.iter().zip(&want[i * dim..(i + 1) * dim]).map(|(a, b)| (a - b).abs()).fold(worst, f32::max);
    }
    let ok = sha == c.onnx_sha256 && worst <= tol;
    Ok(json!({
        "sha256_matches_card": sha == c.onnx_sha256, "parity_inputs": n, "max_abs_diff": worst, "tolerance": tol,
        "threshold": c.threshold, "target_far": c.target_far, "test_metrics": c.metrics.get("test"), "ok": ok,
    }))
}

/// Database rows of a split (sorted), from the model card's palm ids.
pub fn split_rows(db: &Database, card: Option<&ModelCard>, split: &str, block: usize) -> Result<Vec<usize>> {
    let rows: Vec<usize> = match (split, card) {
        ("all", _) => (0..db.names.len()).collect(),
        (s, Some(c)) => {
            let ids: std::collections::HashSet<usize> = c.ids(s).ok_or_else(|| anyhow::anyhow!("split must be test, cal, train or all"))?.iter().copied().collect();
            (0..db.names.len()).filter(|&i| fusion::identity(&db.names[i], block).is_some_and(|(_, k)| ids.contains(&k))).collect()
        }
        (_, None) => bail!("a split other than `all` needs a model card"),
    };
    ensure!(rows.len() >= 2, "fewer than 2 images in split {split}");
    Ok(rows)
}

/// The `eval` report: binned CNN report, plus exact CNN / ridge / fused when ridge codes exist.
pub fn evaluate(db_paths: &DbPaths, card: Option<&ModelCard>, split: &str, block: usize, threshold: f64, threads: usize) -> Result<Value> {
    let full = Database::load(&db_paths.embeddings, &db_paths.names)?;
    let rows = split_rows(&full, card, split, block)?;
    let db = full.subset(|i, _| rows.binary_search(&i).is_ok());
    let r = eval::evaluate(&db, block, threshold)?;
    let mut out = json!({"split": split, "report": r});
    if db_paths.ridges.exists() {
        let ridges = RidgeDb::load(&db_paths.ridges)?;
        ensure!(ridges.codes.len() == full.names.len(), "ridge codes ({}) and embeddings ({}) differ in count; rerun build-db", ridges.codes.len(), full.names.len());
        let fz = card.and_then(|c| c.fusion.clone());
        let rc = RidgeCoder::new(fz.as_ref().map(|f| f.ridge_shift).unwrap_or(1));
        let p = fusion::pairs(&full, &ridges, &rc, &rows, block, threads);
        let names: Vec<String> = rows.iter().map(|&r| full.names[r].clone()).collect();
        let ridge_sim: Vec<f32> = p.ridge.iter().map(|d| 1.0 - d).collect();
        out["cnn_exact"] = fusion::exact(&p.cos, &p, &names, block, Some(threshold as f32));
        out["ridge"] = fusion::exact(&ridge_sim, &p, &names, block, None);
        out["fused"] = match &fz {
            Some(f) => {
                let mut v = fusion::exact(&fusion::fused_scores(&p, f.weight), &p, &names, block, Some(f.threshold));
                v["fusion"] = json!(f);
                v
            }
            None => json!("no fusion in the model card: run training/calibrate_fusion.py"),
        };
    }
    Ok(out)
}

/// Rows [cosine, ridge_distance, genuine] for every pair of a split → npy. Returns (images, pairs).
pub fn pair_scores(db_paths: &DbPaths, card: Option<&ModelCard>, split: &str, ridge_shift: isize, out: &Path, threads: usize) -> Result<(usize, usize)> {
    let block = card.map(|c| c.identity_block).unwrap_or(10);
    let full = Database::load(&db_paths.embeddings, &db_paths.names)?;
    let rows = split_rows(&full, card, split, block)?;
    let ridges = RidgeDb::load(&db_paths.ridges)?;
    let p = fusion::pairs(&full, &ridges, &RidgeCoder::new(ridge_shift), &rows, block, threads);
    let data: Vec<f32> = (0..p.cos.len()).flat_map(|k| [p.cos[k], p.ridge[k], p.genuine[k] as u8 as f32]).collect();
    npy::write_f32(out, p.cos.len(), 3, &data)?;
    Ok((rows.len(), p.cos.len()))
}

/// Inference engine: CNN, plus palm ridges when the codes exist, plus faces when asked. The
/// threshold defaults to the card's, else 0.80.
pub fn engine(model: &Path, db_paths: &DbPaths, threshold: Option<f32>, top_k: usize, roi_dir: Option<PathBuf>, faces: bool, threads: usize) -> Result<server::Engine> {
    let card = ModelCard::load_for(model)?;
    let db = Database::load(&db_paths.embeddings, &db_paths.names)?;
    let fusion = card.as_ref().and_then(|c| c.fusion.clone());
    let ridges = if db_paths.ridges.exists() {
        let r = RidgeDb::load(&db_paths.ridges)?;
        ensure!(r.codes.len() == db.names.len(), "ridge codes and embeddings differ in count; rerun build-db");
        Some((r, RidgeCoder::new(fusion.as_ref().map(|f| f.ridge_shift).unwrap_or(1))))
    } else {
        None
    };
    Ok(server::Engine {
        encoder: Encoder::new(model, threads)?,
        index: db.names.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect(),
        db,
        roi_dir,
        threshold: threshold.or(card.as_ref().map(|c| c.threshold)).unwrap_or(0.8),
        top_k,
        ridges,
        fusion,
        faces: if faces { Some(crate::faces::Faces::new(threads)?) } else { None },
    })
}
