//! Pairwise CNN + ridge scores for a set of database rows, and exact (unbinned) verification metrics.
//! `pair-scores` hands the calibration pairs to Python, which picks the fusion weight and threshold;
//! `eval` reports CNN, ridge and fused metrics on the test palms with the same definitions as
//! training/metrics.py.

use crate::eval::mcc;
use crate::lean_mirror::far_threshold;
use crate::matcher::Database;
use crate::ridges::{fuse, RidgeCoder, RidgeDb};
use serde_json::{json, Value};

pub struct Pairs {
    pub i: Vec<u32>,
    pub j: Vec<u32>,
    pub cos: Vec<f32>,
    pub ridge: Vec<f32>,
    pub genuine: Vec<bool>,
}

/// (session, palm id) from `<session>_<number>`
pub fn identity(name: &str, block: usize) -> Option<(String, usize)> {
    let (s, k) = name.rsplit_once('_')?;
    let k: usize = k.parse().ok()?;
    (k >= 1).then(|| (s.to_string(), (k - 1) / block))
}

/// All pairs i<j among `rows` (indices into `db` / `ridges`).
pub fn pairs(db: &Database, ridges: &RidgeDb, coder: &RidgeCoder, rows: &[usize], block: usize, threads: usize) -> Pairs {
    let pid: Vec<Option<usize>> = rows.iter().map(|&r| identity(&db.names[r], block).map(|x| x.1)).collect();
    let n = rows.len();
    let d = db.dim;
    let threads = threads.max(1);
    let chunks: Vec<Vec<usize>> = (0..threads).map(|t| (t..n).step_by(threads).collect()).collect();
    let parts: Vec<Pairs> = std::thread::scope(|s| {
        let hs: Vec<_> = chunks.iter().map(|rows_a| {
            let pid = &pid;
            s.spawn(move || {
                let mut p = Pairs { i: vec![], j: vec![], cos: vec![], ridge: vec![], genuine: vec![] };
                for &a in rows_a {
                    let ea = &db.unit[rows[a] * d..(rows[a] + 1) * d];
                    for b in a + 1..n {
                        let eb = &db.unit[rows[b] * d..(rows[b] + 1) * d];
                        p.i.push(a as u32);
                        p.j.push(b as u32);
                        p.cos.push(ea.iter().zip(eb).map(|(x, y)| x * y).sum());
                        p.ridge.push(coder.distance(&ridges.codes[rows[a]], &ridges.codes[rows[b]]) as f32);
                        p.genuine.push(pid[a].is_some() && pid[a] == pid[b]);
                    }
                }
                p
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out = Pairs { i: vec![], j: vec![], cos: vec![], ridge: vec![], genuine: vec![] };
    for p in parts {
        out.i.extend(p.i);
        out.j.extend(p.j);
        out.cos.extend(p.cos);
        out.ridge.extend(p.ridge);
        out.genuine.extend(p.genuine);
    }
    out
}

/// EER, TAR @ FAR {1, 0.1, 0.01 %} (thresholds by the calibrated rule), accept/reject MCC at
/// `threshold`, cross-session rank-1 — exact, on the given scores (higher = more similar).
pub fn exact(scores: &[f32], p: &Pairs, names: &[String], block: usize, threshold: Option<f32>) -> Value {
    let (mut gen, mut imp) = (vec![], vec![]);
    for (s, &g) in scores.iter().zip(&p.genuine) {
        if g { gen.push(*s) } else { imp.push(*s) }
    }
    let mut gs = gen.clone();
    gs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut is = imp.clone();
    is.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut thr: Vec<f32> = scores.to_vec();
    thr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    thr.dedup();
    let (mut best, mut eer, mut eer_t) = (f64::MAX, 1.0, 0.0f32);
    for &t in &thr {
        let far = 1.0 - is.partition_point(|&v| v < t) as f64 / is.len().max(1) as f64;
        let frr = gs.partition_point(|&v| v < t) as f64 / gs.len().max(1) as f64;
        if (far - frr).abs() < best {
            best = (far - frr).abs();
            eer = (far + frr) / 2.0;
            eer_t = t;
        }
    }
    let tar = |t: f32| gen.iter().filter(|&&v| v >= t).count() as f64 / gen.len().max(1) as f64;
    let tar_at: Vec<Value> = [1e-2, 1e-3, 1e-4].iter().map(|&a| { let t = far_threshold(&imp, a); json!({"far": a, "tar": tar(t), "threshold": t}) }).collect();
    // rank-1: probes from the second session vs gallery from the first
    let rows: Vec<(String, usize)> = (0..names.len()).map(|r| identity(&names[r], block).unwrap_or_default()).collect();
    let mut sessions: Vec<&String> = rows.iter().map(|r| &r.0).collect();
    sessions.sort();
    sessions.dedup();
    let rank1 = if sessions.len() >= 2 {
        let (g, pr) = (sessions[0].clone(), sessions[1].clone());
        let mut best: std::collections::HashMap<usize, (f32, usize)> = Default::default();
        for (k, &s) in scores.iter().enumerate() {
            let (a, b) = (p.i[k] as usize, p.j[k] as usize);
            for (probe, gal) in [(a, b), (b, a)] {
                if rows[probe].0 == pr && rows[gal].0 == g {
                    let e = best.entry(probe).or_insert((f32::MIN, usize::MAX));
                    if s > e.0 { *e = (s, gal) }
                }
            }
        }
        let ok = best.iter().filter(|(pr, (_, g))| rows[**pr].1 == rows[*g].1).count();
        Some(ok as f64 / best.len().max(1) as f64)
    } else { None };
    let mut out = json!({"genuine_pairs": gen.len(), "impostor_pairs": imp.len(), "eer": eer, "eer_threshold": eer_t, "tar_at_far": tar_at, "rank1_cross_session": rank1});
    if let Some(t) = threshold {
        let tp = gen.iter().filter(|&&v| v >= t).count() as u64;
        let fp = imp.iter().filter(|&&v| v >= t).count() as u64;
        let (fn_, tn) = (gen.len() as u64 - tp, imp.len() as u64 - fp);
        out["at_threshold"] = json!({"threshold": t, "tar": tp as f64 / gen.len().max(1) as f64, "far": fp as f64 / imp.len().max(1) as f64, "mcc": mcc(tp, fp, fn_, tn), "tp": tp, "fp": fp, "fn": fn_, "tn": tn});
    }
    out
}

pub fn fused_scores(p: &Pairs, w: f32) -> Vec<f32> {
    p.cos.iter().zip(&p.ridge).map(|(&c, &r)| fuse(c, r as f64, w)).collect()
}
