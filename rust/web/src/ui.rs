//! Dashboard: inference (upload → matches, palm region, faces) on the left; training controls, live
//! pipeline status (phase, progress, loss and calibration-EER charts, log, test-palm results) and the
//! live model on the right. Live regions re-render over the runtime WebSocket when state changes.

use crate::state::{App, Status, TrainParams};
use std::path::PathBuf;
use std::time::Duration;
use topcoat::{
    Result,
    context::{Cx, context},
    router::{Body, response::{IntoResponse, Response}, route},
    runtime::{Event, connected, procedure, shard, signal},
    view::{View, emit, live, view},
};

fn report(r: anyhow::Result<String>) -> Result<String> {
    Ok(match r {
        Ok(m) => m,
        Err(e) => format!("⚠ {e:#}"),
    })
}

// ── actions (browser → server) ────────────────────────────────────────────────
#[procedure]
async fn train(cx: &Cx, dirs: String, epochs: String, loss: String, far: String, pretrained: bool) -> Result<String> {
    let app = context::<App>(cx);
    let p = (|| -> anyhow::Result<TrainParams> {
        Ok(TrainParams {
            dirs: {
                let d: Vec<PathBuf> = dirs.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).map(PathBuf::from).collect();
                if d.is_empty() { app.demo_dirs() } else { d }
            },
            epochs: epochs.trim().parse().map_err(|_| anyhow::anyhow!("epochs must be a number"))?,
            loss: loss.trim().to_lowercase(),
            target_far: far.trim().parse().map_err(|_| anyhow::anyhow!("target FAR must be a number, e.g. 0.001"))?,
            pretrained,
        })
    })();
    report(p.and_then(|p| app.start_training(p)))
}

#[procedure]
async fn cancel(cx: &Cx) -> Result<String> {
    Ok(if context::<App>(cx).cancel() { "cancelled".into() } else { "nothing is running".into() })
}

#[procedure]
async fn demo(cx: &Cx) -> Result<String> {
    let app = context::<App>(cx).clone();
    report(app.make_demo_data().await.map(|d| format!("demo data ready (synthetic hands, for trying the pipeline only): {d} — leave folders empty to train on it")))
}

// ── static files (routes, so CSS and JS need no escaping) ─────────────────────
pub struct Static(&'static str, &'static str);
impl IntoResponse for Static {
    fn into_response(self, _cx: &Cx) -> Result<Response> {
        Ok(Response::builder().header("Content-Type", self.0).body(Body::from(self.1))?)
    }
}

#[route(GET "/app.css")]
async fn stylesheet(cx: &Cx) -> Result<Static> {
    Ok(Static("text/css; charset=utf-8", include_str!("app.css")))
}

#[route(GET "/upload.js")]
async fn upload_js(cx: &Cx) -> Result<Static> {
    Ok(Static("text/javascript; charset=utf-8", include_str!("upload.js")))
}

// ── page ──────────────────────────────────────────────────────────────────────
pub async fn dashboard(cx: &Cx) -> Result<impl View> {
    Ok(view! {
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>"Palmprint recognition"</title>
                <link rel="stylesheet" href="/app.css">
                <link rel="icon" href="data:,">
                topcoat::runtime::script()
                <script src="/upload.js" defer="defer"></script>
            </head>
            <body>
                <header>
                    <h1>"Palmprint recognition"</h1>
                    <span class="sub">"training in Python (SupCon + SAM) · inference in Rust (ResNet-18 + palm-ridge fusion) · face mesh"</span>
                </header>
                <main>
                    <section class="side">
                        <div class="card">
                            <h2>"Match a palm image"</h2>
                            <p class="hint">"JPG, PNG or TIFF of a palm on a dark background, like the dataset's. The image is matched against every palm in the live database; faces in the image get a 468-point mesh (display only)."</p>
                            <input type="file" id="palm-file" accept=".jpg,.jpeg,.png,.tif,.tiff,image/*">
                            <span id="upload-note" class="hint"></span>
                        </div>
                        results()
                    </section>
                    <section class="side">
                        controls()
                        status()
                    </section>
                </main>
                <footer>"API: POST /v1/chat/completions (OpenAI-compatible, image_url parts; no image → status) · GET /v1/models · POST /api/match"</footer>
            </body>
        </html>
    })
}

#[topcoat::view::component]
async fn controls(cx: &Cx) -> Result<impl View> {
    let default_dirs = context::<App>(cx).0.cfg.data_dirs.join(",");
    let dirs = signal(cx, move || default_dirs.clone());
    let epochs = signal(cx, || "40".to_string());
    let loss = signal(cx, || "supcon".to_string());
    let far = signal(cx, || "0.001".to_string());
    let pretrained = signal(cx, || true);
    let msg = signal(cx, String::new);
    Ok(view! {
        <div class="card controls">
            <h2>"Train"</h2>
            <p class="hint">"Image folders, comma-separated, one per capture session (e.g. …/session1,…/session2; empty = the demo data, if made), 10 consecutive images per palm. Python cuts the palm regions, trains with SAM on 70 % of the palms, calibrates on 15 % and tests on the other 15 %; Rust then verifies the export, builds the database, calibrates the CNN + palm-ridge fusion with Python and evaluates. The live model switches only after evaluation."</p>
            <label class="wide">"folders " <input class="wide" :value=$(dirs.get()) @input=$(|e: Event| dirs.set(e.target.value))></label>
            <br>
            <label>"epochs " <input size="4" :value=$(epochs.get()) @input=$(|e: Event| epochs.set(e.target.value))></label>
            <label>"loss " <input size="7" :value=$(loss.get()) @input=$(|e: Event| loss.set(e.target.value))></label>
            <label>"target FAR " <input size="6" :value=$(far.get()) @input=$(|e: Event| far.set(e.target.value))></label>
            <label class="check"><input type="checkbox" :checked=$(pretrained.get()) @change=$(|e: Event| pretrained.set(e.target.checked))>" ImageNet start"</label>
            <br>
            <button @click=$(async |_e| { let m = train(dirs.get(), epochs.get(), loss.get(), far.get(), pretrained.get()).await; msg.set(m); })>"Start training"</button>
            <button class="ghost" @click=$(async |_e| { let m = cancel().await; msg.set(m); })>"Cancel"</button>
            <button class="ghost" @click=$(async |_e| { let m = demo().await; msg.set(m); })>"Make demo data"</button>
            <p class="hint">"loss: supcon (uses palm identities) or simclr (the original self-supervised loss). ImageNet start needs internet for the weights."</p>
            <p class="msg">$(msg.get())</p>
        </div>
    })
}

// ── live regions ──────────────────────────────────────────────────────────────
fn points(v: &[f64]) -> String {
    if v.len() < 2 {
        return String::new();
    }
    let (lo, hi) = v.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    let span = (hi - lo).max(1e-9);
    v.iter().enumerate().map(|(i, &x)| format!("{:.1},{:.1}", i as f64 / (v.len() - 1) as f64 * 300.0, 80.0 - (x - lo) / span * 76.0 - 2.0)).collect::<Vec<_>>().join(" ")
}

fn pct(v: Option<f64>) -> String {
    v.map_or("—".into(), |x| format!("{:.1} %", 100.0 * x))
}

fn num(v: Option<f64>) -> String {
    v.map_or("n/a".into(), |x| format!("{x:.3}"))
}

macro_rules! status_shard {
    ($name:ident, $panel:ident) => {
        #[shard]
        async fn $name(cx: &Cx) -> Result<impl View> {
            Ok(live! {
                let app = context::<App>(cx).clone();
                let mut changed = app.subscribe();
                loop {
                    let s: Status = app.status();
                    let token = emit! { $panel(s: s) }?;
                    if !connected(cx) {
                        break Ok(token);
                    }
                    changed.changed().await.ok();
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
            })
        }
    };
}

status_shard!(status, status_panels);
status_shard!(results, results_panel);

#[topcoat::view::component]
async fn status_panels(cx: &Cx, s: Status) -> Result<impl View> {
    let (done, total) = s.progress;
    let bar = if total > 0 { done * 100 / total } else { 0 };
    let loss_pts = points(&s.losses);
    let eer_pts = points(&s.cal_eers);
    Ok(view! {
        <div class="card">
            <h2>"Pipeline " <span class=(s.phase.css())>(s.phase.label())</span></h2>
            if !s.stage.is_empty() { <p class="hint">(s.stage.clone())</p> }
            if total > 0 || done > 0 {
                <p>(if total > 0 { format!("{done} / {total}") } else { format!("{done} done") })</p>
                if total > 0 { <div class="bar"><div style=(format!("width:{bar}%"))></div></div> }
            }
            if s.losses.len() >= 2 {
                <p class="hint">"training loss per epoch"</p>
                <svg class="chart" viewBox="0 0 300 80" preserveAspectRatio="none" role="img" aria-label="training loss per epoch">
                    <polyline points=(loss_pts) fill="none" stroke="currentColor" stroke-width="1.5"></polyline>
                </svg>
                <p class="hint">"equal error rate on the calibration palms (lower is better; the best epoch is kept) — last " (pct(s.cal_eers.last().copied()))</p>
                <svg class="chart ok" viewBox="0 0 300 80" preserveAspectRatio="none" role="img" aria-label="calibration EER per epoch">
                    <polyline points=(eer_pts) fill="none" stroke="currentColor" stroke-width="1.5"></polyline>
                </svg>
            }
            if let Some(e) = &s.error { <p class="err">(e.clone())</p> }
            if let Some(r) = &s.results {
                <h2>"Unseen test palms"</h2>
                <p class="hint">(r.test_palms) " palms · " (r.test_images) " images"
                    if let Some(w) = r.fusion_weight { " · fusion weight " (format!("{w:.2}")) " on the CNN" }
                    if let Some(d) = r.verify_max_diff { " · Rust vs Python max |Δ| " (format!("{d:.1e}")) }</p>
                <table>
                    <tr><th>"scorer"</th><th>"EER"</th><th>"TAR @ 0.1 % FAR"</th><th>"TAR / FAR at threshold"</th><th>"MCC"</th><th>"rank-1"</th></tr>
                    for row in &r.rows {
                        <tr><td>(row.scorer.clone())</td><td>(pct(row.eer))</td><td>(pct(row.tar_at_01))</td>
                            <td>(if row.tar.is_some() { format!("{} / {}", pct(row.tar), pct(row.far)) } else { "—".into() })</td>
                            <td><b>(num(row.mcc))</b></td><td>(pct(row.rank1))</td></tr>
                    }
                </table>
            }
            if !s.log.is_empty() {
                <pre class="log">(s.log.iter().cloned().collect::<Vec<_>>().join("\n"))</pre>
            }
        </div>
        <div class="card">
            <h2>"Live model"</h2>
            <div class="kv">
                <span>"model"</span><b>(s.live.model.clone())</b>
                <span>"source"</span><b>(if s.live.trained { "trained here" } else { "committed prototype model" })</b>
                <span>"database"</span><b>(s.live.images) " images"</b>
                <span>"scoring"</span><b>(s.live.scoring.clone())</b>
                <span>"threshold"</span><b>(format!("{:.4}", s.live.threshold))</b>
            </div>
            if let Some(e) = &s.live.error { <p class="err">(e.clone())</p> }
        </div>
    })
}

#[topcoat::view::component]
async fn results_panel(cx: &Cx, s: Status) -> Result<impl View> {
    Ok(view! {
        <div class="card">
            <h2>"Result"</h2>
            match &s.last {
                None => { <p class="hint">"no image matched yet"</p> }
                Some(l) => {
                    <p><b>(l.file.clone())</b></p>
                    if let Some(e) = &l.error { <p class="err">(e.clone())</p> }
                    if let Some(o) = &l.outcome {
                        <p>
                            if o.preprocess.is_none() { <span class="hint">(o.note.clone().unwrap_or_default())</span> }
                            else if o.match_found { <span class="phase p5">"match found"</span> }
                            else { <span class="phase p6">"no match"</span> }
                            " " <span class="hint">"threshold " (format!("{:.4}", o.threshold)) " · " (o.scoring.clone()) " · " (format!("{:.0} ms", o.latency_ms))</span>
                        </p>
                        <div class="imgs">
                            if !o.roi_png.is_empty() { <figure><img src=(o.roi_png.clone()) width="138" height="138" alt="palm region"><figcaption>"palm region the model saw"</figcaption></figure> }
                            if !o.face_png.is_empty() { <figure><img class="faceimg" src=(o.face_png.clone()) alt="face mesh"><figcaption>(o.faces.len()) " face(s), " (o.faces.iter().map(|f| f.triangles.to_string()).collect::<Vec<_>>().join(", ")) " triangles"</figcaption></figure> }
                        </div>
                        if !o.matches.is_empty() {
                            <table>
                                <tr><th>"#"</th><th>"image"</th><th>"score"</th><th>"CNN cosine"</th><th>"ridge distance"</th></tr>
                                for (i, m) in o.matches.iter().enumerate() {
                                    <tr>
                                        <td>(i + 1)</td>
                                        <td>
                                            if let Some(t) = &m.thumbnail { <img class="thumb" src=(t.clone()) alt=""> " " }
                                            (m.name.clone())
                                        </td>
                                        <td><b>(format!("{:.2} %", 100.0 * m.score))</b></td>
                                        <td>(format!("{:.2} %", 100.0 * m.cosine))</td>
                                        <td>(m.ridge_distance.map_or("—".into(), |d| format!("{d:.3}")))</td>
                                    </tr>
                                }
                            </table>
                        }
                    }
                }
            }
        </div>
    })
}
