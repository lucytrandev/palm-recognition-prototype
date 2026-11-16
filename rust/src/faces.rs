//! Faces in an input image: YuNet detection, MediaPipe's 468-point face mesh and its Delaunay
//! triangulation (all from the `palmid` crate). Display only — faces are never matched.

use crate::imgproc::RgbImage;
use anyhow::Result;
use palmid::facemesh::{delaunay, draw_mesh, FaceMesher};
use palmid::teachers::FaceDetector;
use serde::Serialize;

/// opencv_zoo `face_detection_yunet_2023mar.onnx` (MIT), see models/NOTICE-yunet.md
pub const YUNET_ONNX: &[u8] = include_bytes!("../models/face_detection_yunet_2023mar.onnx");

#[derive(Serialize, Clone, Debug)]
pub struct FaceOut {
    pub bbox: [f32; 4],
    pub presence: f32,
    pub landmarks: usize,
    pub triangles: usize,
}

pub struct Faces {
    det: FaceDetector,
    mesh: FaceMesher,
}

impl Faces {
    pub fn new(threads: usize) -> Result<Self> {
        Ok(Self { det: FaceDetector::from_bytes(YUNET_ONNX, threads)?, mesh: FaceMesher::embedded(threads)? })
    }

    /// Detect faces, mesh and triangulate each; returns the faces and the gray image with meshes drawn.
    pub fn run(&mut self, img: &RgbImage) -> Result<(Vec<FaceOut>, Option<palmid::imgproc::GrayImage>)> {
        let rgb = palmid::imgproc::RgbImage { w: img.w, h: img.h, data: img.data.clone() };
        let boxes = self.det.faces(&rgb)?;
        if boxes.is_empty() {
            return Ok((vec![], None));
        }
        let gray = rgb.to_gray();
        let mut vis = gray.clone();
        let mut out = vec![];
        for b in boxes {
            vis.draw_rect(b, 255, 2);
            if let Some(m) = self.mesh.mesh(&gray, b)? {
                let tri = delaunay(&m.points.iter().map(|p| (p[0] as f64, p[1] as f64)).collect::<Vec<_>>());
                draw_mesh(&mut vis, &m, 150, 255);
                out.push(FaceOut { bbox: b, presence: m.presence, landmarks: m.points.len(), triangles: tri.len() });
            } else {
                out.push(FaceOut { bbox: b, presence: 0.0, landmarks: 0, triangles: 0 });
            }
        }
        Ok((out, Some(vis)))
    }
}
