//! Model card written by the Python trainer (`training/train.py`) next to the ONNX file:
//! `<model>.card.json`. Inference takes its threshold, identity block and test split from here.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct Split {
    pub train: Vec<usize>,
    pub cal: Vec<usize>,
    pub test: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelCard {
    pub model: String,
    pub onnx_sha256: String,
    pub threshold: f32,
    pub target_far: f64,
    pub identity_block: usize,
    pub split: Split,
    #[serde(default)]
    pub metrics: serde_json::Value,
}

/// `models/x.onnx` → `models/x.card.json`
pub fn card_path(model: &Path) -> PathBuf {
    model.with_extension("card.json")
}

impl ModelCard {
    pub fn load_for(model: &Path) -> Result<Option<Self>> {
        let p = card_path(model);
        if !p.exists() {
            return Ok(None);
        }
        let s = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        Ok(Some(serde_json::from_str(&s).with_context(|| format!("parsing {}", p.display()))?))
    }

    pub fn ids(&self, split: &str) -> Option<&[usize]> {
        match split {
            "train" => Some(&self.split.train),
            "cal" => Some(&self.split.cal),
            "test" => Some(&self.split.test),
            _ => None,
        }
    }
}

/// Lower-case hex SHA-256 of a file.
pub fn sha256_file(p: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(Sha256::digest(&b).iter().map(|x| format!("{x:02x}")).collect())
}
