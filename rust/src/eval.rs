//! Verification and identification metrics on an embedding database whose names encode identity.
//! Default layout (`session{1,2}_{NNNNN}`, 10 consecutive images per palm per session) matches the
//! dataset the prototype used; identity = (NNNNN − 1) / block.

use crate::matcher::Database;
use anyhow::{anyhow, Result};
use serde::Serialize;

#[derive(Serialize, Debug)]
pub struct Report {
    pub images: usize,
    pub identities: usize,
    pub genuine_pairs: u64,
    pub impostor_pairs: u64,
    pub eer: f64,
    pub eer_threshold: f64,
    pub tar_at_far: Vec<(f64, f64, f64)>, // (FAR target, TAR, threshold)
    pub at_threshold: ThresholdStats,
    pub rank1_cross_session: Option<f64>,
    pub rank1_any_other: f64,
}

#[derive(Serialize, Debug)]
pub struct ThresholdStats {
    pub threshold: f64,
    pub tar: f64,
    pub far: f64,
    /// MCC of the accept/reject decision over all pairs; `None` if undefined.
    pub mcc: Option<f64>,
    pub tp: u64,
    pub fp: u64,
    pub fn_: u64,
    pub tn: u64,
}

const BINS: usize = 20_000; // cosine in [-1, 1] → 1e-4 resolution

fn bin(s: f32) -> usize {
    (((s as f64 + 1.0) / 2.0 * BINS as f64) as usize).min(BINS - 1)
}

pub fn mcc(tp: u64, fp: u64, fn_: u64, tn: u64) -> Option<f64> {
    let (tp, fp, fn_, tn) = (tp as f64, fp as f64, fn_ as f64, tn as f64);
    let d = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
    (d > 0.0).then(|| (tp * tn - fp * fn_) / d)
}

/// `at_threshold` is exact (score ≥ threshold as f32); EER and TAR@FAR use 1e-4 score bins.
pub fn evaluate(db: &Database, block: usize, threshold: f64) -> Result<Report> {
    let n = db.names.len();
    let parse = |s: &str| -> Result<(String, usize)> {
        let (sess, num) = s.rsplit_once('_').ok_or_else(|| anyhow!("name {s} is not <session>_<number>"))?;
        Ok((sess.to_string(), num.parse::<usize>()?))
    };
    let parsed: Vec<(String, usize)> = db.names.iter().map(|s| parse(s)).collect::<Result<_>>()?;
    let pid: Vec<usize> = parsed.iter().map(|(_, k)| (k - 1) / block).collect();
    let d = db.dim;
    let thr32 = threshold as f32;
    let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4);
    // all pairs i<j: genuine/impostor histograms + nearest other neighbour per row
    let rows: Vec<usize> = (0..n).collect();
    let chunks: Vec<Vec<usize>> = (0..threads).map(|t| rows.iter().copied().filter(|r| r % threads == t).collect()).collect();
    let results: Vec<(Vec<u64>, Vec<u64>, Vec<(usize, usize, f32)>, [u64; 2])> = std::thread::scope(|sc| {
        let hs: Vec<_> = chunks
            .iter()
            .map(|rows| {
                let (unit, pid) = (&db.unit, &pid);
                sc.spawn(move || {
                    let (mut g, mut im) = (vec![0u64; BINS], vec![0u64; BINS]);
                    let mut nn = vec![];
                    let mut acc = [0u64; 2]; // exact accepts (score ≥ threshold): genuine, impostor
                    for &i in rows {
                        let a = &unit[i * d..(i + 1) * d];
                        let (mut best, mut bj) = (f32::MIN, usize::MAX);
                        for j in 0..n {
                            if j == i {
                                continue;
                            }
                            let s: f32 = a.iter().zip(&unit[j * d..(j + 1) * d]).map(|(x, y)| x * y).sum();
                            if s > best {
                                best = s;
                                bj = j;
                            }
                            if j > i {
                                let gen = pid[i] == pid[j];
                                if gen { g[bin(s)] += 1 } else { im[bin(s)] += 1 }
                                if s >= thr32 {
                                    acc[!gen as usize] += 1;
                                }
                            }
                        }
                        nn.push((i, bj, best));
                    }
                    (g, im, nn, acc)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (mut g, mut im) = (vec![0u64; BINS], vec![0u64; BINS]);
    let mut nn = vec![(0usize, 0f32); n];
    let mut exact = [0u64; 2];
    for (gg, ii, nns, acc) in results {
        exact[0] += acc[0];
        exact[1] += acc[1];
        for k in 0..BINS {
            g[k] += gg[k];
            im[k] += ii[k];
        }
        for (i, j, s) in nns {
            nn[i] = (j, s);
        }
    }
    let (gt, it): (u64, u64) = (g.iter().sum(), im.iter().sum());
    // accept ⇔ score ≥ threshold bin; suffix sums give accepts at each bin edge
    let mut g_acc = vec![0u64; BINS + 1];
    let mut i_acc = vec![0u64; BINS + 1];
    for k in (0..BINS).rev() {
        g_acc[k] = g_acc[k + 1] + g[k];
        i_acc[k] = i_acc[k + 1] + im[k];
    }
    let thr_of = |k: usize| k as f64 / BINS as f64 * 2.0 - 1.0;
    let (mut eer, mut eer_t, mut best_gap) = (1.0, 0.0, f64::MAX);
    for k in 0..=BINS {
        let far = i_acc[k] as f64 / it.max(1) as f64;
        let frr = 1.0 - g_acc[k] as f64 / gt.max(1) as f64;
        if (far - frr).abs() < best_gap {
            best_gap = (far - frr).abs();
            eer = (far + frr) / 2.0;
            eer_t = thr_of(k);
        }
    }
    let tar_at_far = [1e-2, 1e-3, 1e-4]
        .iter()
        .map(|&target| {
            let k = (0..=BINS).find(|&k| (i_acc[k] as f64 / it.max(1) as f64) <= target).unwrap_or(BINS);
            (target, g_acc[k] as f64 / gt.max(1) as f64, thr_of(k))
        })
        .collect();
    let (tp, fp) = (exact[0], exact[1]); // exact (not binned) at the operating threshold
    let (fn_, tn) = (gt - tp, it - fp);
    let at_threshold = ThresholdStats { threshold, tar: tp as f64 / gt.max(1) as f64, far: fp as f64 / it.max(1) as f64, mcc: mcc(tp, fp, fn_, tn), tp, fp, fn_, tn };
    let rank1_any_other = (0..n).filter(|&i| pid[nn[i].0] == pid[i]).count() as f64 / n as f64;
    // cross-session: probes from the second session name, gallery from the first
    let mut sessions: Vec<&String> = parsed.iter().map(|(s, _)| s).collect();
    sessions.sort();
    sessions.dedup();
    let rank1_cross_session = (sessions.len() >= 2).then(|| {
        let (gs, ps) = (sessions[0].clone(), sessions[1].clone());
        let gal: Vec<usize> = (0..n).filter(|&i| parsed[i].0 == gs).collect();
        let probes: Vec<usize> = (0..n).filter(|&i| parsed[i].0 == ps).collect();
        let hits = probes
            .iter()
            .filter(|&&p| {
                let a = &db.unit[p * d..(p + 1) * d];
                let best = gal.iter().max_by(|&&x, &&y| {
                    let sx: f32 = a.iter().zip(&db.unit[x * d..(x + 1) * d]).map(|(u, v)| u * v).sum();
                    let sy: f32 = a.iter().zip(&db.unit[y * d..(y + 1) * d]).map(|(u, v)| u * v).sum();
                    sx.partial_cmp(&sy).unwrap()
                });
                best.is_some_and(|&b| pid[b] == pid[p])
            })
            .count();
        hits as f64 / probes.len().max(1) as f64
    });
    let mut ids = pid.clone();
    ids.sort();
    ids.dedup();
    Ok(Report { images: n, identities: ids.len(), genuine_pairs: gt, impostor_pairs: it, eer, eer_threshold: eer_t, tar_at_far, at_threshold, rank1_cross_session, rank1_any_other })
}
