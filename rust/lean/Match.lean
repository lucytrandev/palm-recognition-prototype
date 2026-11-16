/-
Palmprint matcher (`find_best_matches`) — formal properties. Lean 4.31, kernel only (no Mathlib).
Scores are fixed-point naturals (cosine × 10⁴ shifted to ≥ 0); the threshold uses the same scale.
-/

/-- "Match found": some returned score reaches the threshold. -/
def matchFound (l : List Nat) (t : Nat) : Bool := l.any (fun s => decide (t ≤ s))

/-- The returned list: the top-k scores at or above the threshold, or all top-k if none are. -/
def chosen (l : List Nat) (t : Nat) : List Nat :=
  if (l.filter (fun s => decide (t ≤ s))).isEmpty then l else l.filter (fun s => decide (t ≤ s))

/-- The UI always has something to show when the database is non-empty. -/
theorem chosen_nonempty (l : List Nat) (t : Nat) (h : l ≠ []) : chosen l t ≠ [] := by
  unfold chosen
  split
  · exact h
  · rename_i hne; intro hnil; simp_all

/-- The fallback never manufactures a match: the verdict on the returned list equals the verdict
    on the top-k itself. -/
theorem chosen_found_iff (l : List Nat) (t : Nat) : matchFound (chosen l t) t = matchFound l t := by
  unfold chosen matchFound
  split
  · rfl
  · rename_i hne
    simp only [List.isEmpty_iff] at hne
    have : (List.filter (fun s => decide (t ≤ s)) l).any (fun s => decide (t ≤ s)) = true := by
      obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil _ hne
      exact List.any_eq_true.mpr ⟨x, hx, (List.mem_filter.mp hx).2⟩
    rw [this]
    obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil _ hne
    exact (List.any_eq_true.mpr ⟨x, (List.mem_filter.mp hx).1, (List.mem_filter.mp hx).2⟩).symm

/-- Raising the threshold never turns "no match" into "match". -/
theorem found_antitone (l : List Nat) (t t' : Nat) (ht : t ≤ t') (h : matchFound l t' = true) :
    matchFound l t = true := by
  unfold matchFound at *
  obtain ⟨x, hx, hxt⟩ := List.any_eq_true.mp h
  exact List.any_eq_true.mpr ⟨x, hx, by simp at hxt ⊢; omega⟩

/-- When a match is found, every returned score reaches the threshold (no below-threshold padding). -/
theorem found_all_above (l : List Nat) (t : Nat) (h : matchFound l t = true) :
    ∀ s ∈ chosen l t, t ≤ s := by
  unfold chosen
  split
  · rename_i hemp
    unfold matchFound at h
    obtain ⟨x, hx, hxt⟩ := List.any_eq_true.mp h
    simp only [List.isEmpty_iff, List.filter_eq_nil_iff] at hemp
    exact absurd hxt (by simpa using hemp x hx)
  · intro s hs; simpa using (List.mem_filter.mp hs).2
