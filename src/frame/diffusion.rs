//! Focus luminosity animation: 1–2 PageRank diffusion steps per frame.

use crate::graph::Csr;

/// One in-place diffusion step: focus_new[i] = d_inv[i] * Σ_j A[i,j] * focus[j].
/// Renormalises to preserve Σ focus = 1.
pub fn diffusion_step(csr: &Csr, d_inv: &[f32], focus: &mut [f32]) {
    let n = csr.n;
    let mut next = vec![0.0f32; n];
    acpu::sparse::csr_matvec_set(&csr.row_ptr, &csr.col_idx, &csr.values, focus, &mut next);
    for i in 0..n {
        next[i] *= d_inv[i];
    }
    let sum: f32 = next.iter().sum();
    if sum > 1e-12 {
        let inv = 1.0 / sum;
        for f in next.iter_mut() { *f *= inv; }
    }
    focus.copy_from_slice(&next);
}
