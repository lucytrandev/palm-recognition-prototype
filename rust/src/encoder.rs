//! ResNet-18 backbone from `output/model/palmprint_encoder.pth`, exported to ONNX by
//! `tools/export_onnx.py`. Output: the 512-d global-average-pooled features — what `app.py` and
//! `client/extract_features.py` actually use (they load only `PalmprintEncoder().encoder`; the
//! 256-d projection head is never used for matching).

use anyhow::{anyhow, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

pub const EMBED_DIM: usize = 512;

pub fn init_ort() -> Result<()> {
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        let c = ["/usr/lib/libonnxruntime.so", "/usr/local/lib/libonnxruntime.so", "/opt/homebrew/lib/libonnxruntime.dylib"];
        match c.iter().find(|p| Path::new(p).exists()) {
            Some(p) => std::env::set_var("ORT_DYLIB_PATH", p),
            None => return Err(anyhow!("set ORT_DYLIB_PATH to libonnxruntime (pip install onnxruntime → .../onnxruntime/capi/libonnxruntime.so.*)")),
        }
    }
    Ok(())
}

pub struct Encoder {
    session: Session,
}

impl Encoder {
    pub fn new(onnx: &Path, threads: usize) -> Result<Self> {
        init_ort()?;
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(onnx)
            .with_context(|| format!("loading {}", onnx.display()))?;
        Ok(Self { session })
    }

    /// `roi`: side×side values in [0,1], replicated to 3 channels with no mean/std (as the prototype does).
    pub fn embed(&mut self, roi: &[f32], side: usize) -> Result<Vec<f32>> {
        let mut x = Vec::with_capacity(3 * roi.len());
        for _ in 0..3 {
            x.extend_from_slice(roi);
        }
        let out = self.session.run(ort::inputs!["input" => Tensor::from_array(([1usize, 3, side, side], x.into_boxed_slice()))?])?;
        let (_, data) = out["embedding"].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }
}
