//! palmprint-web — Topcoat web app for palmprint recognition: training runs the Python pipeline
//! (prepare → SupCon + SAM → calibration) and the Rust steps (verify → index → fusion → evaluate);
//! inference matches uploaded images live. Status streams to the browser over Topcoat's WebSocket.

mod api;
mod pipeline;
mod state;
mod ui;

use state::{App, Config};
use topcoat::{
    asset::{AssetBundle, RouterBuilderAssetExt},
    router::{RouterBuilderDiscoverExt, module_router},
    runtime::RuntimeLayer,
};

#[tokio::main]
async fn main() {
    let cfg = Config::from_env();
    eprintln!("[palmprint-web] repo {} · work {} · python {}", cfg.repo.display(), cfg.work.display(), cfg.python);
    let app = tokio::task::spawn_blocking(move || App::new(cfg)).await.expect("app state");
    let s = app.status();
    eprintln!("[palmprint-web] serving {} ({} images) · {}", s.live.model, s.live.images, s.live.scoring);
    let assets = AssetBundle::load().expect("asset bundle missing: run `topcoat asset bundle -p palmprint-web`");
    topcoat::start(module_router!().assets(assets).with(app).discover().layer(RuntimeLayer::new()).build())
        .await
        .unwrap();
}

#[topcoat::router::page]
async fn page(cx: &topcoat::context::Cx) -> topcoat::Result<impl topcoat::view::View> {
    ui::dashboard(cx).await
}
