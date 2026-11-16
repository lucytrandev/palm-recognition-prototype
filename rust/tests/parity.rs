//! Parity with the Python prototype (fixtures from tools/make_fixtures.py, which calls the repo's own
//! preprocess/preprocessor.py). Embedding test needs ORT_DYLIB_PATH; skipped otherwise.

use palmprint::contours::{area, moments, outer_contours};
use palmprint::imgproc::GrayImage;
use palmprint::preprocess::{load, preprocess, TARGET};
use std::path::{Path, PathBuf};

fn fx(n: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(n)
}
fn png(n: &str) -> GrayImage {
    let i = image::open(fx(n)).unwrap().to_luma8();
    GrayImage { w: i.width() as usize, h: i.height() as usize, data: i.into_raw() }
}
fn index() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fx("index.json")).unwrap()).unwrap()
}
fn ndiff(a: &GrayImage, b: &GrayImage) -> usize {
    assert_eq!((a.w, a.h), (b.w, b.h), "size");
    a.data.iter().zip(&b.data).filter(|(x, y)| x != y).count()
}
const CASES: [&str; 3] = ["original", "synthetic", "edge"];

#[test]
fn every_stage_matches_python() {
    let idx = index();
    for c in CASES {
        let st = preprocess(&load(&fx(&format!("{c}_input.png"))).unwrap()).expect("preprocess");
        let r = &idx[c];
        let d = [
            ("gray", ndiff(&st.gray, &png(&format!("{c}_gray.png")))),
            ("enhanced", ndiff(&st.enhanced, &png(&format!("{c}_enhanced.png")))),
            ("binary", ndiff(&st.binary, &png(&format!("{c}_binary.png")))),
            ("roi", ndiff(&st.roi, &png(&format!("{c}_roi.png")))),
            ("resized", ndiff(&st.resized, &png(&format!("{c}_resized.png")))),
        ];
        eprintln!("{c}: {:?} | contours rust {} py {} | area rust {} py {} | centroid rust ({}, {}) py ({}, {})",
            d, st.trace.n_contours, r["n_contours"], st.trace.largest_area, r["largest_area"], st.trace.cx, st.trace.cy, r["cx"], r["cy"]);
        for (k, n) in d {
            assert_eq!(n, 0, "{c}: {k} differs in {n} pixels");
        }
        assert_eq!(st.trace.n_contours as u64, r["n_contours"].as_u64().unwrap(), "{c}: number of external contours");
        assert_eq!(st.trace.largest_area, r["largest_area"].as_f64().unwrap(), "{c}: largest contour area");
        assert_eq!((st.trace.cx, st.trace.cy), (r["cx"].as_i64().unwrap(), r["cy"].as_i64().unwrap()), "{c}: centroid");
    }
}

#[test]
fn contour_moments_match_opencv() {
    let idx = index();
    for c in CASES {
        let cs = outer_contours(&png(&format!("{c}_binary.png")));
        let best = cs.iter().max_by(|a, b| area(a).partial_cmp(&area(b)).unwrap()).unwrap();
        let m = moments(best);
        let r = &idx[c];
        for (k, v) in [("m00", m.m00), ("m10", m.m10), ("m01", m.m01)] {
            let p = r[k].as_f64().unwrap();
            assert!((v - p).abs() <= 1e-6 * p.abs().max(1.0), "{c}: {k} rust {v} py {p}");
        }
    }
}

#[test]
fn embedding_matches_python_onnx() {
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        eprintln!("skipped: set ORT_DYLIB_PATH");
        return;
    }
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/palmprint_encoder_512.onnx");
    let mut enc = palmprint::encoder::Encoder::new(&model, 2).unwrap();
    for c in CASES {
        let st = preprocess(&load(&fx(&format!("{c}_input.png"))).unwrap()).unwrap();
        let e = enc.embed(&st.normalized, TARGET).unwrap();
        let (_, _, py) = palmprint::npy::read_f32(&fx(&format!("{c}_embedding.npy"))).unwrap();
        let mx = e.iter().zip(&py).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let cos: f32 = palmprint::matcher::normalize(&e).iter().zip(palmprint::matcher::normalize(&py)).map(|(a, b)| a * b).sum();
        eprintln!("{c}: embedding max|Δ| {mx:e}, cosine {cos:.7}");
        assert!(mx < 1e-3 && cos > 0.99999);
    }
}

#[test]
fn reads_the_committed_database() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../output/embeddings");
    let db = palmprint::matcher::Database::load(&base.join("all_embeddings.npy"), &base.join("image_names.npy")).unwrap();
    assert_eq!((db.names.len(), db.dim), (11996, 512));
    assert!(db.names[0].starts_with("session"));
}
