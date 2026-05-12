//! Focus luminosity animation: 1–2 PageRank diffusion steps per frame.
//! y = D^{-1} A x — uses acpu::sparse::csr_matvec. Step 4 (alongside cull).

use crate::graph::Csr;

/// One in-place diffusion step: focus = D^{-1} A focus.
/// d_inv[i] = 1/D[i].
pub fn diffusion_step(csr: &Csr, d_inv: &[f32], focus: &mut [f32]) {
    let n = csr.n;
    let mut next = vec![0.0f32; n];
    // TODO: replace with acpu::sparse::csr_matvec once that kernel lands.
    for i in 0..n {
        let (cols, vals) = csr.row(i);
        let mut acc = 0.0f32;
        for (&c, &v) in cols.iter().zip(vals.iter()) {
            acc += v * focus[c as usize];
        }
        next[i] = acc * d_inv[i];
    }
    // Renormalize to preserve total focus = 1.
    let sum: f32 = next.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for f in next.iter_mut() { *f *= inv; }
    }
    focus.copy_from_slice(&next);
}
