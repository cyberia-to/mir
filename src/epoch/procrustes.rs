//! Procrustes alignment: align epoch positions to the canonical anchor frame.
//!
//! Minimises Σ_{p ∈ anchor} ‖Q·s·X(p) + t − X_anchor(p)‖² over Q∈O(3), s, t.
//! Uses acpu::matmul_f32 for the 3×3 SVD (anchor-1024 is small; full dense ok).
//!
//! R-1.0 §4.3. Step 2 — placeholder until anchor epoch-0 reference is stored.

use crate::epoch::eigensolver::SpectralCoords;

/// Apply Procrustes (Q, s, t) to all positions. In-place.
pub fn align(_coords: &mut SpectralCoords, _anchor_ref: &[[f32; 3]]) {
    // TODO step 2: SVD via acpu::matmul_f32 on anchor subset, apply to all.
}
