//! Training-pipeline phases and the allowed transitions (lean/Pipeline.lean). The live matching
//! engine is replaced only on Evaluating → Ready, so a model that hasn't been verified, indexed,
//! fusion-calibrated and evaluated never serves.

use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub enum Phase {
    #[default]
    Idle,
    Preparing,
    Training,
    Verifying,
    Indexing,
    Fusing,
    Evaluating,
    Ready,
    Failed,
}

#[cfg_attr(not(test), allow(dead_code))]
pub const ALL: [Phase; 9] = [Phase::Idle, Phase::Preparing, Phase::Training, Phase::Verifying, Phase::Indexing, Phase::Fusing, Phase::Evaluating, Phase::Ready, Phase::Failed];

impl Phase {
    pub fn running(self) -> bool {
        !matches!(self, Phase::Idle | Phase::Ready | Phase::Failed)
    }
    pub fn label(self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::Preparing => "1/6 preparing ROIs (Python)",
            Phase::Training => "2/6 training with SAM (Python)",
            Phase::Verifying => "3/6 verifying the export (Rust)",
            Phase::Indexing => "4/6 building the database (Rust)",
            Phase::Fusing => "5/6 calibrating CNN + ridge fusion (Rust → Python)",
            Phase::Evaluating => "6/6 evaluating on unseen palms (Rust)",
            Phase::Ready => "trained model live",
            Phase::Failed => "failed",
        }
    }
    pub fn css(self) -> &'static str {
        match self {
            Phase::Idle => "phase",
            Phase::Ready => "phase p5",
            Phase::Failed => "phase p6",
            _ => "phase p3",
        }
    }
}

/// Mirror of `canStep` in lean/Pipeline.lean.
pub fn can_step(from: Phase, to: Phase) -> bool {
    use Phase::*;
    matches!(
        (from, to),
        (Idle | Ready | Failed, Preparing)
            | (Preparing, Training)
            | (Training, Verifying)
            | (Verifying, Indexing)
            | (Indexing, Fusing)
            | (Fusing, Evaluating)
            | (Evaluating, Ready)
    ) || (from.running() && to == Failed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Lean theorems, checked exhaustively on the mirror.
    #[test]
    fn lean_statements_hold() {
        for a in ALL {
            for b in ALL {
                if can_step(a, b) && b == Phase::Ready {
                    assert_eq!(a, Phase::Evaluating, "ready_only_after_eval");
                }
                if can_step(a, b) && b == Phase::Preparing {
                    assert!(!a.running(), "start_only_when_stopped");
                }
                if can_step(a, b) && b.running() && a.running() {
                    assert_eq!(ALL.iter().position(|&x| x == b).unwrap(), ALL.iter().position(|&x| x == a).unwrap() + 1, "no_skip");
                }
                assert!(!(can_step(a, b) && a == b), "no_self_loops");
            }
        }
    }
}
