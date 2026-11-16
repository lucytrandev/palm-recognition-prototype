//! Cosine matching against the embedding database — `client/cosine_similarity.py`.

use crate::lean_mirror::match_found;
use crate::npy;
use anyhow::{bail, Result};
use serde::Serialize;
use std::path::Path;

pub struct Database {
    pub names: Vec<String>,
    pub dim: usize,
    /// L2-normalised rows (cosine = dot product)
    pub unit: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Match {
    pub name: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatchResult {
    pub matches: Vec<Match>,
    pub match_found: bool,
    pub threshold: f32,
}

pub fn normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt() as f32;
    if n == 0.0 { v.to_vec() } else { v.iter().map(|x| x / n).collect() }
}

impl Database {
    pub fn load(embeddings: &Path, names: &Path) -> Result<Self> {
        let (rows, dim, data) = npy::read_f32(embeddings)?;
        let names = npy::read_strings(names)?;
        if names.len() != rows {
            bail!("{} embeddings but {} names", rows, names.len());
        }
        let unit = data.chunks_exact(dim).flat_map(normalize).collect();
        Ok(Self { names, dim, unit })
    }

    /// Keep only rows whose index passes `keep` (e.g. the model card's test palms).
    pub fn subset(&self, keep: impl Fn(usize, &str) -> bool) -> Self {
        let mut out = Self { names: vec![], dim: self.dim, unit: vec![] };
        for (i, n) in self.names.iter().enumerate() {
            if keep(i, n) {
                out.names.push(n.clone());
                out.unit.extend_from_slice(&self.unit[i * self.dim..(i + 1) * self.dim]);
            }
        }
        out
    }

    pub fn similarities(&self, query: &[f32]) -> Vec<f32> {
        let q = normalize(query);
        self.unit.chunks_exact(self.dim).map(|r| r.iter().zip(&q).map(|(a, b)| a * b).sum()).collect()
    }

    /// `find_best_matches`: take the top-k by score; keep those ≥ threshold; if none are, return all top-k.
    /// `match_found` ⇔ some returned score ≥ threshold (Lean `matchFound`).
    pub fn best_matches(&self, query: &[f32], threshold: f32, top_k: usize) -> MatchResult {
        let s = self.similarities(query);
        let mut idx: Vec<usize> = (0..s.len()).collect();
        idx.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap().then(b.cmp(&a))); // argsort()[::-1] tie order
        idx.truncate(top_k);
        let above: Vec<usize> = idx.iter().copied().filter(|&i| s[i] >= threshold).collect();
        let chosen = if above.is_empty() { idx } else { above };
        let matches: Vec<Match> = chosen.iter().map(|&i| Match { name: self.names[i].clone(), score: s[i] }).collect();
        let found = match_found(&matches.iter().map(|m| m.score).collect::<Vec<_>>(), threshold);
        MatchResult { matches, match_found: found, threshold }
    }
}
