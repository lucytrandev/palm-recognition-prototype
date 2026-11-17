//! JSON endpoints: the upload used by the page, and the OpenAI-compatible API on the same live engine.

use crate::state::App;
use base64::Engine as _;
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};
use topcoat::{
    Result,
    context::{Cx, context},
    router::{content::Json, route},
};

const MODEL_ID: &str = "palmprint-resnet18-ridges";

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn decode(uri: &str) -> Option<Vec<u8>> {
    let (head, b64) = uri.split_once(',')?;
    head.starts_with("data:").then_some(())?;
    base64::engine::general_purpose::STANDARD.decode(b64).ok()
}

/// Page upload: {"file": name, "image": data URI} → match outcome (also shown in the results panel).
#[route(POST "/api/match")]
async fn upload(cx: &Cx, Json(body): Json<Value>) -> Result<Json<Value>> {
    let app = context::<App>(cx).clone();
    let file = body["file"].as_str().unwrap_or("upload").to_string();
    let Some(bytes) = decode(body["image"].as_str().unwrap_or("")) else {
        return Ok(Json(json!({"error": "image must be a data: URI"})));
    };
    let out = tokio::task::spawn_blocking(move || app.match_image(file, &bytes)).await?;
    Ok(Json(match out {
        Ok(mut o) => {
            o.roi_png.clear();
            o.face_png.clear();
            json!(o)
        }
        Err(e) => json!({"error": format!("{e:#}")}),
    }))
}

#[route(GET "/v1/models")]
async fn models(cx: &Cx) -> Result<Json<Value>> {
    Ok(Json(json!({"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "palmprint"}]})))
}

/// `image_url` parts (data: URIs) → match results as the message content; no image → live status.
#[route(POST "/v1/chat/completions")]
async fn chat(cx: &Cx, Json(body): Json<Value>) -> Result<Json<Value>> {
    let app = context::<App>(cx).clone();
    let mut images = vec![];
    for m in body["messages"].as_array().into_iter().flatten().filter(|m| m["role"] == "user") {
        for p in m["content"].as_array().into_iter().flatten().filter(|p| p["type"] == "image_url") {
            images.push(p["image_url"]["url"].as_str().unwrap_or("").to_string());
        }
    }
    let content = tokio::task::spawn_blocking(move || {
        if images.is_empty() {
            let s = app.status();
            return json!({"phase": s.phase, "live": s.live, "results": s.results});
        }
        let results: Vec<Value> = images
            .iter()
            .enumerate()
            .map(|(i, u)| match decode(u).ok_or_else(|| anyhow::anyhow!("image_url must be a data: URI")).and_then(|b| app.match_image(format!("api image {}", i + 1), &b)) {
                Ok(mut o) => {
                    o.roi_png.clear();
                    o.face_png.clear();
                    json!(o)
                }
                Err(e) => json!({"error": format!("{e:#}")}),
            })
            .collect();
        json!({"results": results})
    })
    .await?;
    Ok(Json(json!({
        "id": format!("chatcmpl-{}", now()), "object": "chat.completion", "created": now(), "model": MODEL_ID,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content.to_string()}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    })))
}
