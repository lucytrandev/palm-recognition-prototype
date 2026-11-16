//! Mirrors of `lean/Match.lean`. The matcher calls these, and the tests check the theorem statements.

/// Lean `matchFound`: some returned score reaches the threshold.
pub fn match_found(scores: &[f32], threshold: f32) -> bool {
    scores.iter().any(|&s| s >= threshold)
}

/// Lean `chosen`: scores at/above threshold, or all of them if none are.
pub fn chosen(scores: &[f32], threshold: f32) -> Vec<f32> {
    let above: Vec<f32> = scores.iter().copied().filter(|&s| s >= threshold).collect();
    if above.is_empty() { scores.to_vec() } else { above }
}

/// Mirror of `training/metrics.py::far_threshold` (Lean `far_threshold_bound`, `far_le_alpha`):
/// the smallest f32 above the ⌊α·N⌋-th largest impostor score (0-based). Calibration happens in
/// Python; this exists so the theorem statement can be checked on the same arithmetic.
pub fn far_threshold(impostor: &[f32], alpha: f64) -> f32 {
    let mut s = impostor.to_vec();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let k = (alpha * s.len() as f64).floor() as usize;
    if k >= s.len() { -1.0 } else { s[k].next_up() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn theorems_hold_on_grid() {
        let grid: Vec<f32> = (0..=20).map(|i| i as f32 / 20.0).collect();
        for a in &grid {
            for b in &grid {
                for c in &grid {
                    let l = [*a, *b, *c];
                    for t in &grid {
                        let ch = chosen(&l, *t);
                        assert!(!ch.is_empty()); // chosen_nonempty
                        assert_eq!(match_found(&ch, *t), match_found(&l, *t)); // chosen_found_iff
                        if match_found(&l, *t) {
                            assert!(ch.iter().all(|s| s >= t)); // found_all_above
                        }
                        for t2 in &grid {
                            if t <= t2 && match_found(&l, *t2) {
                                assert!(match_found(&l, *t)); // found_antitone
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn far_threshold_matches_python() {
        // training/metrics.py: far_threshold([.5,.9,.1,.9,.3], .2) = 0.9000000357627869, (…, .4) = 0.5000000596046448
        let s = [0.5f32, 0.9, 0.1, 0.9, 0.3];
        assert_eq!(far_threshold(&s, 0.2) as f64, 0.9000000357627869);
        assert_eq!(far_threshold(&s, 0.4) as f64, 0.5000000596046448);
    }

    #[test]
    fn far_threshold_bound_holds() {
        let mut x = 0x9E3779B97F4A7C15u64;
        for n in [1usize, 2, 7, 100, 1000, 5000] {
            for &alpha in &[0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5] {
                // scores with many ties (coarse grid) and continuous ones
                for coarse in [true, false] {
                    let s: Vec<f32> = (0..n).map(|_| {
                        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                        let u = (x >> 11) as f32 / (1u64 << 53) as f32;
                        if coarse { (u * 20.0).floor() / 20.0 } else { u * 2.0 - 1.0 }
                    }).collect();
                    let t = far_threshold(&s, alpha);
                    let accepted = s.iter().filter(|&&v| v >= t).count();
                    assert!(accepted <= (alpha * n as f64).floor() as usize, "n={n} α={alpha} accepted={accepted}");
                }
            }
        }
    }
}
