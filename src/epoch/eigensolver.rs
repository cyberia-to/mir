//! LOBPCG eigensolver on the normalized Laplacian ℒ = I − D^{−½} A D^{−½}.
//!
//! Produces eigenvectors u₂, u₃, u₄ (skipping trivial u₁ = const) as the
//! 3D spectral coordinates of each particle (R-1.0 §3).
//!
//! Depends on acpu::sparse::csr_matvec for the sparse matrix-vector product.
//! Implementation: step 2 — in progress.

use crate::graph::Csr;

/// Spectral coordinates: n particles × 3 eigenvectors (row-major f32).
pub struct SpectralCoords {
    pub n: usize,
    pub coords: Vec<f32>, // length n * 3
}

impl SpectralCoords {
    pub fn position(&self, i: usize) -> [f32; 3] {
        let base = i * 3;
        [self.coords[base], self.coords[base + 1], self.coords[base + 2]]
    }
}

/// Compute degree vector D[i] = sum of row i weights.
pub fn degree_vec(csr: &Csr) -> Vec<f32> {
    (0..csr.n).map(|i| {
        let (_, vals) = csr.row(i);
        vals.iter().copied().sum()
    }).collect()
}

/// Normalized Laplacian matvec: y = ℒ x = x − D^{−½} A D^{−½} x.
///
/// d_inv_sqrt[i] = 1/√D[i] (precomputed, 0 for isolated nodes).
pub fn laplacian_matvec(
    csr: &Csr,
    d_inv_sqrt: &[f32],
    x: &[f32],
    y: &mut [f32],
) {
    let n = csr.n;
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(y.len(), n);

    // Compute z[i] = d_inv_sqrt[i] * x[i]
    let mut z = vec![0.0f32; n];
    for i in 0..n {
        z[i] = d_inv_sqrt[i] * x[i];
    }

    // Sparse matvec: w[i] = Σ_j A[i,j] * z[j]
    // TODO: replace inner loop with acpu::sparse::csr_matvec when available.
    let mut w = vec![0.0f32; n];
    for i in 0..n {
        let (cols, vals) = csr.row(i);
        let mut acc = 0.0f32;
        for (&c, &v) in cols.iter().zip(vals.iter()) {
            acc += v * z[c as usize];
        }
        w[i] = acc;
    }

    // y[i] = x[i] − d_inv_sqrt[i] * w[i]
    for i in 0..n {
        y[i] = x[i] - d_inv_sqrt[i] * w[i];
    }
}

/// Placeholder — full LOBPCG follows in step 2.
pub fn solve(_csr: &Csr, _k: usize) -> SpectralCoords {
    todo!("LOBPCG eigensolver — step 2")
}
