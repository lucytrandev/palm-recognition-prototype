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
}
