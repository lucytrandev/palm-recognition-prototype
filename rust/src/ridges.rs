//! Palm-ridge (principal line / crease) matching next to the CNN embedding: Competitive Code
//! (Kong & Zhang 2004: per cell, the orientation of the strongest dark-line Gabor response) on the
//! prototype's 138×138 ROI, compared by shift-tolerant angular distance. Implementation from the
//! `palmid` crate (dhaneshkk/test1_palm), which is checked bit-exact against its reference.
//!
//! Fused score = w · cosine + (1 − w) · (1 − ridge distance). w and the fused threshold are chosen in
//! Python (`training/calibrate_fusion.py`) on the calibration palms and stored in the model card.

use crate::imgproc::GrayImage;
use anyhow::{ensure, Result};
use palmid::palmprint::{code_distance, Code, PalmCoder, PalmParams};
use std::path::Path;

pub const ROI: usize = 128;

pub struct RidgeCoder {
    coder: PalmCoder,
    pub shift: isize,
}

impl RidgeCoder {
    pub fn new(shift: isize) -> Self {
        Self { coder: PalmCoder::new(PalmParams { roi: ROI, ..PalmParams::default() }), shift }
    }

    /// Code of the prototype's resized ROI (already CLAHE-enhanced by the preprocessing).
    pub fn code(&self, resized: &GrayImage) -> Code {
        let g = palmid::imgproc::GrayImage { w: resized.w, h: resized.h, data: resized.data.clone() };
        self.coder.code(&palmid::imgproc::resize_area(&g, ROI, ROI))
    }

    pub fn distance(&self, a: &Code, b: &Code) -> f64 {
        code_distance(a, b, self.shift)
    }
}

/// Ridge codes for every database image, row-aligned with the embeddings.
pub struct RidgeDb {
    pub h: usize,
    pub w: usize,
    pub codes: Vec<Code>,
}

impl RidgeDb {
    pub fn load(path: &Path) -> Result<Self> {
        let (n, cols, v) = crate::npy::read_f32(path)?;
        let side = (cols as f64).sqrt() as usize;
        ensure!(side * side == cols, "ridge codes are not square grids ({cols} cells)");
        let codes = (0..n).map(|i| Code { h: side, w: side, cells: v[i * cols..(i + 1) * cols].iter().map(|&x| x as u8).collect() }).collect();
        Ok(Self { h: side, w: side, codes })
    }

    pub fn save(path: &Path, codes: &[Code]) -> Result<()> {
        let cols = codes.first().map(|c| c.cells.len()).unwrap_or(0);
        let data: Vec<f32> = codes.iter().flat_map(|c| c.cells.iter().map(|&x| x as f32)).collect();
        crate::npy::write_f32(path, codes.len(), cols, &data)
    }
}

pub fn fuse(cos: f32, ridge_dist: f64, w: f32) -> f32 {
    w * cos + (1.0 - w) * (1.0 - ridge_dist as f32)
}
