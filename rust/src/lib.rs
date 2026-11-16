//! Rust inference for dhaneshkk/palm-recognition-prototype: the same preprocessing (bit-exact with
//! OpenCV), the trained ResNet-18 encoder via ONNX, cosine matching against the committed embedding
//! database, a database builder, evaluation, a web UI and an OpenAI-compatible API.

pub mod card;
pub mod contours;
pub mod encoder;
pub mod faces;
pub mod fusion;
pub mod eval;
pub mod imgproc;
pub mod lean_mirror;
pub mod matcher;
pub mod npy;
pub mod preprocess;
pub mod ridges;
pub mod server;
