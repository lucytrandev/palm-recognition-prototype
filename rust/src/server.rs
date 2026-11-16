//! Web UI (replaces the Streamlit app) and OpenAI-compatible API.
//! * `GET /` — upload page: shows the uploaded image, the preprocessed ROI, top matches + scores, verdict
//! * `POST /api/match` — multipart `file` → JSON
//! * `POST /v1/chat/completions` — user `image_url` parts (data: URIs) → JSON results as message content
//! * `GET /v1/models`

use crate::encoder::Encoder;
use crate::imgproc::GrayImage;
use crate::matcher::{Database, MatchResult};
use crate::preprocess::{load_bytes, preprocess, Trace, TARGET};
use anyhow::{anyhow, Result};
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::response::{Html, IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use base64::Engine as _;
use serde::Serialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub const MODEL_ID: &str = "palmprint-resnet18-simclr";

pub struct Engine {
    pub encoder: Encoder,
    pub db: Database,
    /// name → database row
    pub index: std::collections::HashMap<String, usize>,
    /// palm-ridge codes for the database and the coder (when `ridge_codes.npy` exists)
    pub ridges: Option<(crate::ridges::RidgeDb, crate::ridges::RidgeCoder)>,
    /// fusion weight / threshold from the model card (Python-calibrated)
    pub fusion: Option<crate::card::Fusion>,
    /// YuNet + 468-point face mesh with Delaunay triangulation (display only)
    pub faces: Option<crate::faces::Faces>,
    /// Optional folder of preprocessed ROIs (`<name>.npy`, 138×138 float) for match thumbnails.
    pub roi_dir: Option<PathBuf>,
    pub threshold: f32,
    pub top_k: usize,
}

#[derive(Serialize)]
pub struct Thumb {
    pub name: String,
    /// the score used for ranking: fused when fusion is calibrated, else cosine
    pub score: f32,
    pub cosine: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ridge_distance: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumbnail: Option<String>,
}

#[derive(Serialize)]
pub struct Outcome {
    /// `None` when no palm region was found (the response then carries only faces, if any)
    pub preprocess: Option<Trace>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub roi_png: String,
    pub match_found: bool,
    pub threshold: f32,
    pub matches: Vec<Thumb>,
    /// "cnn" or "cnn+ridges (w=…)"
    pub scoring: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub faces: Vec<crate::faces::FaceOut>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub face_png: String,
    pub latency_ms: f64,
}

fn png_data_uri(g: &GrayImage) -> String {
    let mut buf = Vec::new();
    image::codecs::png::PngEncoder::new(&mut buf)
        .write_image_ext(&g.data, g.w as u32, g.h as u32)
        .ok();
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(buf))
}

trait PngExt {
    fn write_image_ext(self, d: &[u8], w: u32, h: u32) -> image::ImageResult<()>;
}
impl<W: std::io::Write> PngExt for image::codecs::png::PngEncoder<W> {
    fn write_image_ext(self, d: &[u8], w: u32, h: u32) -> image::ImageResult<()> {
        use image::ImageEncoder;
        self.write_image(d, w, h, image::ExtendedColorType::L8)
    }
}

impl Engine {
    pub fn run(&mut self, bytes: &[u8]) -> Result<Outcome> {
        let t = std::time::Instant::now();
        let img = load_bytes(bytes)?;
        let (faces, face_png) = match self.faces.as_mut() {
            Some(f) => {
                let (out, vis) = f.run(&img)?;
                let png = vis.map(|v| {
                    let s = (480.0 / v.w.max(v.h) as f32).min(1.0);
                    let v = if s < 1.0 { palmid::imgproc::resize_area(&v, (v.w as f32 * s) as usize, (v.h as f32 * s) as usize) } else { v };
                    png_data_uri(&GrayImage { w: v.w, h: v.h, data: v.data })
                }).unwrap_or_default();
                (out, png)
            }
            None => (vec![], String::new()),
        };
        let Some(st) = preprocess(&img) else {
            if faces.is_empty() {
                return Err(anyhow!("could not preprocess the image (no palm region found above the threshold)"));
            }
            return Ok(Outcome { preprocess: None, note: Some("no palm region found; faces only".into()), roi_png: String::new(), match_found: false,
                threshold: self.fusion.as_ref().map(|f| f.threshold).unwrap_or(self.threshold), matches: vec![], scoring: "none".into(), faces, face_png,
                latency_ms: t.elapsed().as_secs_f64() * 1e3 });
        };
        let emb = self.encoder.embed(&st.normalized, TARGET)?;
        // (name, ranking score, cosine, ridge distance)
        let (ranked, match_found, threshold, scoring): (Vec<(String, f32, f32, Option<f32>)>, bool, f32, String) =
            match (&self.ridges, &self.fusion) {
                (Some((rdb, rc)), Some(f)) => {
                    // re-rank the 50 best cosine candidates by the fused score
                    let MatchResult { matches, .. } = self.db.best_matches(&emb, f32::INFINITY, 50);
                    let q = rc.code(&st.resized);
                    let mut v: Vec<_> = matches.into_iter().map(|m| {
                        let d = rc.distance(&q, &rdb.codes[self.index[&m.name]]) as f32;
                        (m.name, crate::ridges::fuse(m.score, d as f64, f.weight), m.score, Some(d))
                    }).collect();
                    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    v.truncate(self.top_k);
                    let found = v.first().is_some_and(|m| m.1 >= f.threshold);
                    (v, found, f.threshold, format!("cnn+ridges (w={:.2})", f.weight))
                }
                (ridges, _) => {
                    let MatchResult { matches, match_found, threshold } = self.db.best_matches(&emb, self.threshold, self.top_k);
                    let q = ridges.as_ref().map(|(_, rc)| rc.code(&st.resized));
                    let v = matches.into_iter().map(|m| {
                        let d = ridges.as_ref().zip(q.as_ref()).map(|((rdb, rc), q)| rc.distance(q, &rdb.codes[self.index[&m.name]]) as f32);
                        (m.name, m.score, m.score, d)
                    }).collect();
                    (v, match_found, threshold, "cnn".into())
                }
            };
        let matches = ranked
            .into_iter()
            .map(|(name, score, cosine, ridge_distance)| {
                let m = crate::matcher::Match { name, score };
                let thumbnail = self.roi_dir.as_ref().and_then(|d| {
                    let (_, _, v) = crate::npy::read_f32(&d.join(format!("{}.npy", m.name))).ok()?;
                    let side = (v.len() as f64).sqrt() as usize;
                    Some(png_data_uri(&GrayImage { w: side, h: side, data: v.iter().map(|x| (x * 255.0).round().clamp(0.0, 255.0) as u8).collect() }))
                });
                Thumb { name: m.name, score: m.score, cosine, ridge_distance, thumbnail }
            })
            .collect();
        Ok(Outcome { preprocess: Some(st.trace), note: None, roi_png: png_data_uri(&st.resized), match_found, threshold, matches, scoring, faces, face_png, latency_ms: t.elapsed().as_secs_f64() * 1e3 })
    }
}

type Shared = Arc<Mutex<Engine>>;

async fn index() -> Html<&'static str> {
    Html(include_str!("ui.html"))
}

async fn api_match(State(e): State<Shared>, mut mp: Multipart) -> Response {
    let mut bytes = None;
    while let Ok(Some(field)) = mp.next_field().await {
        if field.name() == Some("file") {
            bytes = field.bytes().await.ok();
        }
    }
    let Some(b) = bytes else { return (axum::http::StatusCode::BAD_REQUEST, Json(json!({"error": "multipart field 'file' required"}))).into_response() };
    let r = tokio::task::spawn_blocking(move || e.lock().unwrap_or_else(|p| p.into_inner()).run(&b)).await;
    match r {
        Ok(Ok(o)) => Json(o).into_response(),
        Ok(Err(err)) => (axum::http::StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": format!("{err:#}")}))).into_response(),
        Err(err) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": err.to_string()}))).into_response(),
    }
}

async fn models() -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}))
}

async fn chat(State(e): State<Shared>, Json(body): Json<Value>) -> Json<Value> {
    let mut images = vec![];
    for m in body["messages"].as_array().into_iter().flatten().filter(|m| m["role"] == "user") {
        for p in m["content"].as_array().into_iter().flatten().filter(|p| p["type"] == "image_url") {
            images.push(p["image_url"]["url"].as_str().unwrap_or("").to_string());
        }
    }
    let content = tokio::task::spawn_blocking(move || {
        let mut eng = e.lock().unwrap_or_else(|p| p.into_inner());
        let mut results = vec![];
        for url in images {
            let r = url
                .split_once(',')
                .filter(|(h, _)| h.starts_with("data:"))
                .and_then(|(_, b)| base64::engine::general_purpose::STANDARD.decode(b).ok())
                .ok_or_else(|| anyhow!("image_url must be a data: URI"))
                .and_then(|b| eng.run(&b));
            results.push(match r {
                Ok(mut o) => {
                    o.roi_png.clear();
                    o.face_png.clear();
                    o.matches.iter_mut().for_each(|m| m.thumbnail = None);
                    json!(o)
                }
                Err(err) => json!({"error": format!("{err:#}")}),
            });
        }
        if results.is_empty() {
            json!({"database_size": eng.db.names.len(), "threshold": eng.threshold, "top_k": eng.top_k})
        } else {
            json!({"results": results})
        }
    })
    .await
    .unwrap_or_else(|e| json!({"error": e.to_string()}));
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    Json(json!({
        "id": format!("chatcmpl-{now}"), "object": "chat.completion", "created": now, "model": MODEL_ID,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content.to_string()}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }))
}

pub fn router(engine: Engine) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/match", post(api_match))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(Arc::new(Mutex::new(engine)))
}
