/-
  Training pipeline of the palmprint web app (rust/web/src/pipeline.rs `can_step`).
  Lean 4.31, kernel only. The live engine is swapped on Evaluating → Ready only.
-/

inductive Phase where
  | idle | preparing | training | verifying | indexing | fusing | evaluating | ready | failed
  deriving DecidableEq, Repr

open Phase

def running : Phase → Bool
  | idle | ready | failed => false
  | _ => true

def canStep : Phase → Phase → Bool
  | idle, preparing | ready, preparing | failed, preparing => true
  | preparing, training | training, verifying | verifying, indexing => true
  | indexing, fusing | fusing, evaluating | evaluating, ready => true
  | p, failed => running p
  | _, _ => false

/-- A trained model goes live only after evaluation. -/
theorem ready_only_after_eval (p : Phase) (h : canStep p ready = true) : p = evaluating := by
  cases p <;> simp_all [canStep]

/-- A new run can start only when nothing is running. -/
theorem start_only_when_stopped (p : Phase) (h : canStep p preparing = true) : running p = false := by
  cases p <;> simp_all [canStep, running]

/-- Evaluation is reached only from fusion, fusion only from indexing, indexing only from
    verification: no stage can be skipped. -/
theorem no_skip_eval (p : Phase) (h : canStep p evaluating = true) : p = fusing := by
  cases p <;> simp_all [canStep]

theorem no_skip_fusing (p : Phase) (h : canStep p fusing = true) : p = indexing := by
  cases p <;> simp_all [canStep]

theorem no_skip_indexing (p : Phase) (h : canStep p indexing = true) : p = verifying := by
  cases p <;> simp_all [canStep]

theorem no_skip_verifying (p : Phase) (h : canStep p verifying = true) : p = training := by
  cases p <;> simp_all [canStep]

theorem no_self_loops (p : Phase) : canStep p p = false := by
  cases p <;> rfl
