//! Lanczos eigensolver on the normalized Laplacian ℒ = I − D^{−½} A D^{−½}.
//!
//! Produces eigenvectors u₂, u₃, u₄ (skipping trivial u₁ = const) as the
//! 3D spectral coordinates of each particle (R-1.0 §3).
//!
//! Uses `acpu::sparse::csr_matvec_set` for the sparse matrix-vector product
//! inside `laplacian_matvec`, and implements a full Lanczos iteration with
//! QR eigendecomposition on the resulting tridiagonal matrix.

use crate::graph::Csr;

// ── Public types ──────────────────────────────────────────────────────────────

/// Spectral coordinates: n particles × 3 eigenvectors (row-major f32).
pub struct SpectralCoords {
    pub n: usize,
    pub coords: Vec<f32>, // n×3 row-major (u₂,u₃,u₄)
    pub extra: Vec<f32>,  // n×2 row-major (u₅,u₆) for hue; zeros if insufficient eigenvectors
}

impl SpectralCoords {
    pub fn position(&self, i: usize) -> [f32; 3] {
        let base = i * 3;
        [self.coords[base], self.coords[base + 1], self.coords[base + 2]]
    }
}

// ── Degree and Laplacian ──────────────────────────────────────────────────────

/// Compute degree vector D[i] = sum of row i weights.
pub fn degree_vec(csr: &Csr) -> Vec<f32> {
    (0..csr.n)
        .map(|i| {
            let (_, vals) = csr.row(i);
            vals.iter().copied().sum()
        })
        .collect()
}

/// Normalized Laplacian matvec: y = ℒ x = x − D^{−½} A D^{−½} x.
///
/// d_inv_sqrt[i] = 1/√D[i] (precomputed, 0 for isolated nodes).
/// Uses `acpu::sparse::csr_matvec_set` for the A·z step.
pub fn laplacian_matvec(csr: &Csr, d_inv_sqrt: &[f32], x: &[f32], y: &mut [f32]) {
    let n = csr.n;
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(y.len(), n);

    // z[i] = d_inv_sqrt[i] * x[i]
    let mut z = vec![0.0f32; n];
    for i in 0..n {
        z[i] = d_inv_sqrt[i] * x[i];
    }

    // w = A · z  (csr_matvec_set: y = A·x)
    acpu::sparse::csr_matvec_set(&csr.row_ptr, &csr.col_idx, &csr.values, &z, y);

    // y[i] = x[i] − d_inv_sqrt[i] * y[i]
    for i in 0..n {
        y[i] = x[i] - d_inv_sqrt[i] * y[i];
    }
}

// ── BLAS-like helpers ─────────────────────────────────────────────────────────

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn norm2(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    // y += alpha * x
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

fn scale(s: f32, v: &mut [f32]) {
    for vi in v.iter_mut() {
        *vi *= s;
    }
}

// ── Lanczos iteration ─────────────────────────────────────────────────────────

const KRYLOV: usize = 48; // number of Lanczos steps

/// Run Lanczos iteration on the normalized Laplacian.
///
/// Returns:
/// - `v_basis`: n × k matrix in row-major, v_basis[j * n .. (j+1)*n] = v_j
/// - `alpha`: k Lanczos diagonal coefficients
/// - `beta`: k Lanczos off-diagonal coefficients (beta[0] unused)
fn lanczos(
    csr: &Csr,
    d_inv_sqrt: &[f32],
    n: usize,
    k: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Seed: random unit vector with seed 42 (deterministic LCG).
    let mut v_basis = vec![0.0f32; k * n]; // v_basis[j*n..(j+1)*n] = v_j
    let mut alpha = vec![0.0f32; k];
    let mut beta = vec![0.0f32; k]; // beta[0] = 0 (not used)

    // Generate starting vector using a simple LCG (seed=42).
    let v0 = &mut v_basis[0..n];
    let mut state: u64 = 42;
    for vi in v0.iter_mut() {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        *vi = ((state >> 33) as f32) / (u32::MAX as f32) - 0.5;
    }

    // Project out the constant vector u₁ = 1/√n.
    // v₀ -= <v₀, u₁> u₁
    let sum: f32 = v0.iter().sum();
    let proj = sum / n as f32;
    for vi in v0.iter_mut() {
        *vi -= proj;
    }

    // Normalize v₀.
    let nrm = norm2(v0);
    if nrm < 1e-12 {
        // Degenerate start: use standard basis e_0 projected.
        for vi in v0.iter_mut() {
            *vi = 0.0;
        }
        v0[0] = 1.0;
        let proj = 1.0 / n as f32;
        for vi in v0.iter_mut() {
            *vi -= proj;
        }
        let nrm = norm2(v0);
        scale(1.0 / nrm, v0);
    } else {
        scale(1.0 / nrm, v0);
    }

    let mut w = vec![0.0f32; n];
    let mut v_prev = vec![0.0f32; n]; // v_{j-1}

    for j in 0..k {
        // w = ℒ v_j
        let vj = v_basis[j * n..(j + 1) * n].to_vec();
        laplacian_matvec(csr, d_inv_sqrt, &vj, &mut w);

        // α_j = v_j^T w
        alpha[j] = dot(&vj, &w);

        // w = w − α_j v_j − β_{j-1} v_{j-1}
        axpy(-alpha[j], &vj, &mut w);
        if j > 0 {
            axpy(-beta[j], &v_prev, &mut w);
        }

        // Full re-orthogonalization: w -= Σ_{i=0..j} <w, v_i> v_i
        for i in 0..=j {
            let vi = &v_basis[i * n..(i + 1) * n];
            let c = dot(&w, vi);
            axpy(-c, vi, &mut w);
        }
        // Second pass for numerical stability.
        for i in 0..=j {
            let vi = &v_basis[i * n..(i + 1) * n];
            let c = dot(&w, vi);
            axpy(-c, vi, &mut w);
        }

        // β_j = ‖w‖
        let nrm = norm2(&w);

        if j + 1 < k {
            beta[j + 1] = nrm;
            // v_{j+1} = w / β_j
            // Split v_basis at (j+1)*n so we can borrow the previous vectors
            // immutably while writing to vj1 = v_basis[(j+1)*n..(j+2)*n].
            let (prev_vecs, rest) = v_basis.split_at_mut((j + 1) * n);
            let vj1 = &mut rest[..n];
            if nrm < 1e-12 {
                // Invariant subspace found — fill with a new random direction.
                let mut st: u64 = 42u64.wrapping_mul(j as u64 + 1).wrapping_add(999);
                for vi in vj1.iter_mut() {
                    st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    *vi = ((st >> 33) as f32) / (u32::MAX as f32) - 0.5;
                }
                // Re-orthogonalize new random vector against all previous basis
                // vectors (prev_vecs contains v_0 .. v_j, each of length n).
                for i in 0..=j {
                    let vi_ref = &prev_vecs[i * n..(i + 1) * n];
                    let c = dot(vj1, vi_ref);
                    axpy(-c, vi_ref, vj1);
                }
                let new_nrm = norm2(vj1);
                if new_nrm > 1e-12 {
                    scale(1.0 / new_nrm, vj1);
                } else {
                    vj1.fill(0.0);
                    vj1[0] = 1.0;
                }
                beta[j + 1] = 0.0;
            } else {
                for (vj1_i, wi) in vj1.iter_mut().zip(w.iter()) {
                    *vj1_i = wi / nrm;
                }
            }
        }

        v_prev.copy_from_slice(&vj);
    }

    (v_basis, alpha, beta)
}

// ── QR algorithm on symmetric tridiagonal ────────────────────────────────────

/// Eigendecompose k×k symmetric tridiagonal matrix T.
///
/// T[i,i]   = alpha[i]
/// T[i,i+1] = T[i+1,i] = beta[i+1]   (beta[0] unused, beta[1..k-1] = off-diag)
///
/// Uses f64 and the dstev/tqli QR algorithm (Numerical Recipes §11.3, 0-indexed).
///
/// Returns (eigenvalues sorted ascending, eigenvector matrix Y k×k row-major).
/// Y[row * k + col] = coordinate 'row' of the col-th eigenvector.
fn tridiag_eig(alpha: &[f32], beta: &[f32], k: usize) -> (Vec<f32>, Vec<f32>) {
    if k == 0 {
        return (vec![], vec![]);
    }
    if k == 1 {
        return (vec![alpha[0]], vec![1.0f32]);
    }

    // Build the full k×k symmetric tridiagonal matrix in dense f64 form,
    // then apply Jacobi iteration to diagonalize it.
    //
    // For k ≤ 48 this is O(k³) per iteration but with tiny k the constant
    // factor is small and Jacobi is guaranteed correct.

    // A[i*k+j] = T[i,j]; symmetric tridiagonal.
    let mut a = vec![0.0f64; k * k];
    for i in 0..k {
        a[i * k + i] = alpha[i] as f64;
    }
    for i in 0..k - 1 {
        let b = beta[i + 1] as f64;
        a[i * k + (i + 1)] = b;
        a[(i + 1) * k + i] = b;
    }

    // Eigenvector accumulation matrix, identity.
    let mut v = vec![0.0f64; k * k];
    for i in 0..k {
        v[i * k + i] = 1.0;
    }

    // Jacobi iteration: zero off-diagonal elements one by one.
    // For small k, use cyclic-by-row Jacobi.
    for _sweep in 0..200 {
        let mut max_off = 0.0f64;
        for p in 0..k {
            for q in p + 1..k {
                max_off = max_off.max(a[p * k + q].abs());
            }
        }
        if max_off < 1e-13 {
            break;
        }

        for p in 0..k {
            for q in p + 1..k {
                let apq = a[p * k + q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                // Jacobi rotation to zero a[p,q]:
                let tau = (a[q * k + q] - a[p * k + p]) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update matrix a = G^T · a · G.
                // Update diagonal.
                let app = a[p * k + p];
                let aqq = a[q * k + q];
                a[p * k + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                a[q * k + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                a[p * k + q] = 0.0;
                a[q * k + p] = 0.0;

                // Update off-diagonal entries in rows/cols p and q.
                for r in 0..k {
                    if r != p && r != q {
                        let arp = a[r * k + p];
                        let arq = a[r * k + q];
                        a[r * k + p] = c * arp - s * arq;
                        a[p * k + r] = a[r * k + p];
                        a[r * k + q] = s * arp + c * arq;
                        a[q * k + r] = a[r * k + q];
                    }
                }

                // Accumulate rotation into eigenvector matrix v.
                for r in 0..k {
                    let vrp = v[r * k + p];
                    let vrq = v[r * k + q];
                    v[r * k + p] = c * vrp - s * vrq;
                    v[r * k + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    // Extract eigenvalues from diagonal of a, convert to f32.
    let mut d_f32: Vec<f32> = (0..k).map(|i| a[i * k + i] as f32).collect();
    let mut z_f32: Vec<f32> = v.iter().map(|&x| x as f32).collect();

    // Insertion sort: sort eigenvalues ascending, permute eigenvectors.
    for i in 1..k {
        let di = d_f32[i];
        let zi: Vec<f32> = z_f32[i * k..(i + 1) * k].to_vec();
        let mut j = i as isize - 1;
        while j >= 0 && d_f32[j as usize] > di {
            d_f32[(j + 1) as usize] = d_f32[j as usize];
            z_f32.copy_within(j as usize * k..(j as usize + 1) * k, (j as usize + 1) * k);
            j -= 1;
        }
        d_f32[(j + 1) as usize] = di;
        z_f32[(j + 1) as usize * k..(j + 2) as usize * k].copy_from_slice(&zi);
    }

    (d_f32, z_f32)
}

// ── Main solver ───────────────────────────────────────────────────────────────

/// Compute spectral coordinates via Lanczos + QR eigendecomposition.
///
/// Produces 3D coordinates (eigenvectors u₂, u₃, u₄) scaled to R_scene=1000.
pub fn solve(csr: &Csr) -> SpectralCoords {
    let n = csr.n;

    if n < 4 {
        // Too small for spectral layout; return zeros.
        return SpectralCoords { n, coords: vec![0.0f32; n * 3], extra: vec![0.0f32; n * 2] };
    }

    // Precompute D^{-½}.
    let deg = degree_vec(csr);
    let d_inv_sqrt: Vec<f32> = deg
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    let k = KRYLOV.min(n - 1);

    // Run Lanczos.
    let (v_basis, alpha, beta) = lanczos(csr, &d_inv_sqrt, n, k);

    // Eigendecompose k×k tridiagonal.
    let (evals, evecs) = tridiag_eig(&alpha, &beta, k);

    // Find 5 smallest non-trivial eigenvectors (skip λ ≈ 0).
    // The first eigenvalue of the Laplacian is always 0 (constant vector).
    let mut selected: Vec<usize> = Vec::new();
    for j in 0..k {
        if evals[j].abs() < 1e-6 && selected.is_empty() {
            continue; // skip the trivial eigenvalue
        }
        if evals[j] >= -1e-6 {
            // Accept non-trivial eigenvalues ≥ 0
            selected.push(j);
        }
        if selected.len() == 5 {
            break;
        }
    }

    // Pad with the next available if fewer than 5 found.
    for j in 0..k {
        if selected.len() >= 5 {
            break;
        }
        if !selected.contains(&j) {
            selected.push(j);
        }
    }

    // Compute Ritz vectors X[:,i] = V * Y[:,i] for selected eigenvectors.
    // v_basis is k*n (v_basis[j*n .. (j+1)*n] = v_j).
    // evecs is k*k row-major (evecs[row*k + col] = Y[row, col]).
    // X[:,i][p] = Σ_{j=0}^{k-1} v_j[p] * Y[j, selected[i]]
    let compute_ritz = |col: usize| -> Vec<f32> {
        let mut vec = vec![0.0f32; n];
        for p in 0..n {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += v_basis[j * n + p] * evecs[j * k + col];
            }
            vec[p] = acc;
        }
        vec
    };

    let mut coords = vec![0.0f32; n * 3];
    for (dim, &col) in selected[..3.min(selected.len())].iter().enumerate() {
        let ritz = compute_ritz(col);
        for p in 0..n {
            coords[p * 3 + dim] = ritz[p];
        }
    }

    // Scale to R_scene = 1000.
    let max_norm = (0..n)
        .map(|p| {
            let x = coords[p * 3];
            let y = coords[p * 3 + 1];
            let z = coords[p * 3 + 2];
            (x * x + y * y + z * z).sqrt()
        })
        .fold(0.0f32, f32::max);

    if max_norm > 1e-8 {
        let scale = 1000.0 / max_norm;
        for v in coords.iter_mut() {
            *v *= scale;
        }
    }

    // Build extra (n×2) from selected[3] and selected[4] as Ritz vectors.
    let mut extra = vec![0.0f32; n * 2];
    if selected.len() >= 4 {
        let ritz3 = compute_ritz(selected[3]);
        for p in 0..n {
            extra[p * 2] = ritz3[p];
        }
    }
    if selected.len() >= 5 {
        let ritz4 = compute_ritz(selected[4]);
        for p in 0..n {
            extra[p * 2 + 1] = ritz4[p];
        }
    }

    SpectralCoords { n, coords, extra }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Csr, Cyberlink, ParticleIndex};

    fn hash(v: u8) -> [u8; 32] {
        let mut h = [0u8; 32];
        h[0] = v;
        h
    }

    fn link(from: u8, to: u8) -> Cyberlink {
        Cyberlink { neuron: [0u8; 32], from: hash(from), to: hash(to), token: 0, amount: 1, valence: 1, block: 1 }
    }

    /// Build CSR for a 6-node path: 0-1-2-3-4-5.
    fn path6_csr() -> Csr {
        let links: Vec<Cyberlink> = (0u8..5).map(|i| link(i, i + 1)).collect();
        let vocab = ParticleIndex::build(links.iter().copied());
        Csr::build(links.into_iter(), &vocab)
    }

    #[test]
    fn laplacian_matvec_constant_vector() {
        // The kernel of the normalized Laplacian ℒ = I − D^{-½} A D^{-½} is
        // spanned by D^{½} · 1 (the degree-weighted constant).
        // Verify: ℒ applied to D^{½} · 1  (normalized) yields ~0.
        let csr = path6_csr();
        let n = csr.n;
        let deg = degree_vec(&csr);
        let d_inv_sqrt: Vec<f32> = deg.iter().map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }).collect();
        let d_sqrt: Vec<f32> = deg.iter().map(|&d| d.sqrt()).collect();

        // Kernel vector: D^{½} · 1, then normalize.
        let mut x = d_sqrt.clone();
        let nrm: f32 = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        for vi in x.iter_mut() { *vi /= nrm; }

        let mut y = vec![0.0f32; n];
        laplacian_matvec(&csr, &d_inv_sqrt, &x, &mut y);

        for (i, &yi) in y.iter().enumerate() {
            assert!(yi.abs() < 1e-4, "ℒ(D^½·1)[{i}] = {yi} should be ~0");
        }
    }

    #[test]
    fn lanczos_small_path_graph() {
        // 6-node path graph: known eigenvalues of normalized Laplacian are
        // λ_k = 1 − cos(kπ/n) for k = 0..n-1 (approximately, for the path graph).
        // Verify that:
        //   1. solve() returns n=6 coords
        //   2. the 3 chosen eigenvectors are mutually orthonormal
        //   3. eigenvalues are all ≥ 0

        let csr = path6_csr();
        let n = csr.n;
        let deg = degree_vec(&csr);
        let d_inv_sqrt: Vec<f32> = deg.iter().map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }).collect();

        let k = KRYLOV.min(n - 1);
        let (v_basis, alpha, beta) = lanczos(&csr, &d_inv_sqrt, n, k);
        let (evals, evecs) = tridiag_eig(&alpha, &beta, k);

        // Eigenvalues should all be ≥ -ε (Laplacian is positive semi-definite).
        for (j, &lam) in evals.iter().enumerate() {
            assert!(lam >= -1e-4, "eigenvalue[{j}] = {lam} should be ≥ 0");
        }

        // Compute 3 Ritz vectors.
        let mut selected = Vec::new();
        for j in 0..k {
            if evals[j].abs() < 1e-4 && selected.is_empty() { continue; }
            if evals[j] >= -1e-4 {
                selected.push(j);
            }
            if selected.len() == 3 { break; }
        }
        // Pad.
        for j in 0..k {
            if selected.len() >= 3 { break; }
            if !selected.contains(&j) { selected.push(j); }
        }

        // Build Ritz vectors.
        let mut ritz = vec![vec![0.0f32; n]; 3];
        for (dim, &col) in selected.iter().enumerate() {
            for p in 0..n {
                let mut acc = 0.0f32;
                for j in 0..k {
                    acc += v_basis[j * n + p] * evecs[j * k + col];
                }
                ritz[dim][p] = acc;
            }
        }

        // Check orthonormality: <u_i, u_j> ≈ δ_{ij}.
        for i in 0..3 {
            let norm = dot(&ritz[i], &ritz[i]).sqrt();
            assert!((norm - 1.0).abs() < 0.01, "‖ritz[{i}]‖ = {norm}");
            for j in (i + 1)..3 {
                let ip = dot(&ritz[i], &ritz[j]);
                assert!(ip.abs() < 0.01, "<ritz[{i}], ritz[{j}]> = {ip}");
            }
        }

        // Solve should return n=6 coords with finite values.
        let coords = solve(&csr);
        assert_eq!(coords.n, n);
        assert_eq!(coords.coords.len(), n * 3);
        for (i, &v) in coords.coords.iter().enumerate() {
            assert!(v.is_finite(), "coords[{i}] = {v}");
        }
    }
}
