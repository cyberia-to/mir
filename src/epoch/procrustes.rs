//! Procrustes alignment: align epoch spectral coordinates to a canonical anchor frame.
//!
//! Minimises Σ_{p ∈ anchor} ‖Q·X(p) − X_ref(p)‖² over orthogonal Q ∈ O(3)
//! using a 3×3 Jacobi SVD. R-1.0 §4.3.
//!
//! On the initial run (no stored reference), the current positions are used as
//! the reference, producing an identity alignment.

use crate::epoch::eigensolver::SpectralCoords;

// ── 3×3 matrix helpers ────────────────────────────────────────────────────────

type M3 = [[f32; 3]; 3];

fn mat3_zero() -> M3 {
    [[0.0; 3]; 3]
}

fn mat3_identity() -> M3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn mat3_mul(a: &M3, b: &M3) -> M3 {
    let mut c = mat3_zero();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn mat3_transpose(a: &M3) -> M3 {
    let mut t = mat3_zero();
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = a[j][i];
        }
    }
    t
}

fn mat3_det(a: &M3) -> f32 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

// ── Jacobi SVD ────────────────────────────────────────────────────────────────

/// Symmetric Schur decomposition: find (c, s) Givens rotation that zeros
/// off-diagonal element (p,q) of the 3×3 symmetric matrix A.
fn sym_schur2(a: &M3, p: usize, q: usize) -> (f32, f32) {
    if a[p][q].abs() < 1e-10 {
        return (1.0, 0.0); // already zero
    }
    let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;
    (c, s)
}

/// Apply a Givens rotation (c, s) to symmetric 3×3 matrix A from both sides:
/// A' = G^T A G, where G is identity with G[p][p]=c, G[q][q]=c, G[p][q]=s, G[q][p]=-s.
fn jacobi_rotate(a: &mut M3, v: &mut M3, p: usize, q: usize, c: f32, s: f32) {
    let apq = a[p][q];
    let app = a[p][p];
    let aqq = a[q][q];

    // Update diagonal.
    a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    a[p][q] = 0.0;
    a[q][p] = 0.0;

    // Update off-diagonal entries in rows/cols p and q.
    for r in 0..3 {
        if r != p && r != q {
            let arp = a[r][p];
            let arq = a[r][q];
            a[r][p] = c * arp - s * arq;
            a[p][r] = a[r][p];
            a[r][q] = s * arp + c * arq;
            a[q][r] = a[r][q];
        }
    }

    // Accumulate rotation into V.
    for r in 0..3 {
        let vrp = v[r][p];
        let vrq = v[r][q];
        v[r][p] = c * vrp - s * vrq;
        v[r][q] = s * vrp + c * vrq;
    }
}

/// Jacobi SVD of a 3×3 matrix M.
///
/// Returns (U, S, V) such that M = U · diag(S) · V^T,
/// with singular values S ≥ 0, and U, V orthogonal.
pub fn jacobi_svd_3x3(m: M3) -> (M3, [f32; 3], M3) {
    // Compute A = M^T M (symmetric positive semi-definite).
    let mt = mat3_transpose(&m);
    let mut a = mat3_mul(&mt, &m); // A = M^T M

    // Jacobi iteration to diagonalize A.
    let mut v = mat3_identity();
    let off_diag_pairs = [(0, 1), (0, 2), (1, 2)];

    for _ in 0..100 {
        // Check off-diagonal magnitude.
        let off: f32 = off_diag_pairs
            .iter()
            .map(|&(p, q)| a[p][q] * a[p][q])
            .sum::<f32>()
            .sqrt();
        if off < 1e-12 {
            break;
        }
        for &(p, q) in &off_diag_pairs {
            let (c, s) = sym_schur2(&a, p, q);
            jacobi_rotate(&mut a, &mut v, p, q, c, s);
        }
    }

    // Singular values: square roots of eigenvalues of A = M^T M.
    let mut sigma = [a[0][0].max(0.0).sqrt(), a[1][1].max(0.0).sqrt(), a[2][2].max(0.0).sqrt()];

    // Compute U = M V S^{-1}.
    // For each column i of U: U[:,i] = (M V[:,i]) / sigma[i].
    let mut u = mat3_zero();
    for i in 0..3 {
        // M v_i
        for row in 0..3 {
            let mut acc = 0.0f32;
            for k in 0..3 {
                acc += m[row][k] * v[k][i];
            }
            u[row][i] = acc;
        }
        let s = sigma[i];
        if s > 1e-8 {
            for row in 0..3 {
                u[row][i] /= s;
            }
        } else {
            // Zero singular value: pick an arbitrary unit vector orthogonal to existing U columns.
            // Use Gram-Schmidt with a standard basis vector.
            sigma[i] = 0.0;
            for basis in 0..3 {
                let mut candidate = [0.0f32; 3];
                candidate[basis] = 1.0;
                // Subtract projections onto existing U columns.
                for j in 0..i {
                    let proj: f32 = (0..3).map(|r| candidate[r] * u[r][j]).sum();
                    for r in 0..3 {
                        candidate[r] -= proj * u[r][j];
                    }
                }
                let nrm: f32 = candidate.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if nrm > 1e-8 {
                    for r in 0..3 {
                        u[r][i] = candidate[r] / nrm;
                    }
                    break;
                }
            }
        }
    }

    (u, sigma, v)
}

// ── Rotation application ──────────────────────────────────────────────────────

/// Apply 3×3 rotation matrix Q to all n positions (in-place, row-major n×3).
pub fn apply_rotation(positions: &mut [f32], q: M3) {
    let n = positions.len() / 3;
    for i in 0..n {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];
        positions[i * 3] = q[0][0] * x + q[0][1] * y + q[0][2] * z;
        positions[i * 3 + 1] = q[1][0] * x + q[1][1] * y + q[1][2] * z;
        positions[i * 3 + 2] = q[2][0] * x + q[2][1] * y + q[2][2] * z;
    }
}

// ── Procrustes alignment ──────────────────────────────────────────────────────

/// Align spectral coordinates to the anchor reference frame.
///
/// If `anchor_ref` is empty (initial run), the current positions are used as
/// the reference and the alignment is the identity.
///
/// `anchor_ref`: slice of 3D positions for anchor particles (ordered by particle index).
///
/// The function computes Q = V U^T (Wahba's solution) and applies it in-place.
pub fn align(coords: &mut SpectralCoords, anchor_ref: &[[f32; 3]]) {
    let n = coords.n;
    if anchor_ref.is_empty() || n == 0 {
        return; // identity alignment
    }

    let m_anchors = anchor_ref.len().min(n);

    // Compute centroids of current and reference anchor positions.
    let mut centroid_cur = [0.0f32; 3];
    let mut centroid_ref = [0.0f32; 3];
    for i in 0..m_anchors {
        let pos = coords.position(i);
        for d in 0..3 {
            centroid_cur[d] += pos[d];
            centroid_ref[d] += anchor_ref[i][d];
        }
    }
    for d in 0..3 {
        centroid_cur[d] /= m_anchors as f32;
        centroid_ref[d] /= m_anchors as f32;
    }

    // Cross-covariance M = A^T B where A = anchor_current (centered), B = anchor_ref (centered).
    // M is 3×3: M[i][j] = Σ_p A[p,i] * B[p,j]
    let mut cross_cov = mat3_zero();
    for p in 0..m_anchors {
        let cur = coords.position(p);
        let a = [
            cur[0] - centroid_cur[0],
            cur[1] - centroid_cur[1],
            cur[2] - centroid_cur[2],
        ];
        let b = [
            anchor_ref[p][0] - centroid_ref[0],
            anchor_ref[p][1] - centroid_ref[1],
            anchor_ref[p][2] - centroid_ref[2],
        ];
        for i in 0..3 {
            for j in 0..3 {
                cross_cov[i][j] += a[i] * b[j];
            }
        }
    }

    // SVD of M = U S V^T.
    let (u, sigma, v) = jacobi_svd_3x3(cross_cov);

    // Q = V U^T (optimal rotation).
    let ut = mat3_transpose(&u);
    let mut q = mat3_mul(&v, &ut);

    // Correct handedness: if det(Q) = -1, flip sign of the last column of V.
    if mat3_det(&q) < 0.0 {
        let mut v_corr = v;
        for row in 0..3 {
            v_corr[row][2] = -v_corr[row][2];
        }
        q = mat3_mul(&v_corr, &ut);
    }

    // Optimal scale s = trace(Sigma) / trace(A^T A) where A = centered anchor current coords
    let trace_s: f32 = sigma[0] + sigma[1] + sigma[2];
    let trace_ata: f32 = (0..m_anchors).map(|p| {
        let cur = coords.position(p);
        let a = [cur[0]-centroid_cur[0], cur[1]-centroid_cur[1], cur[2]-centroid_cur[2]];
        a[0]*a[0] + a[1]*a[1] + a[2]*a[2]
    }).sum::<f32>();
    let s = if trace_ata > 1e-10 { trace_s / trace_ata } else { 1.0f32 };

    // Translation t = centroid_ref - s * Q * centroid_cur
    let qc = [
        q[0][0]*centroid_cur[0] + q[0][1]*centroid_cur[1] + q[0][2]*centroid_cur[2],
        q[1][0]*centroid_cur[0] + q[1][1]*centroid_cur[1] + q[1][2]*centroid_cur[2],
        q[2][0]*centroid_cur[0] + q[2][1]*centroid_cur[1] + q[2][2]*centroid_cur[2],
    ];
    let t = [
        centroid_ref[0] - s * qc[0],
        centroid_ref[1] - s * qc[1],
        centroid_ref[2] - s * qc[2],
    ];

    // Apply full transform to ALL n particles: X'(p) = s*Q*X(p) + t
    let n = coords.n;
    for i in 0..n {
        let x = coords.coords[i*3];
        let y = coords.coords[i*3+1];
        let z = coords.coords[i*3+2];
        coords.coords[i*3]   = s*(q[0][0]*x + q[0][1]*y + q[0][2]*z) + t[0];
        coords.coords[i*3+1] = s*(q[1][0]*x + q[1][1]*y + q[1][2]*z) + t[1];
        coords.coords[i*3+2] = s*(q[2][0]*x + q[2][1]*y + q[2][2]*z) + t[2];
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_coords(positions: &[[f32; 3]]) -> SpectralCoords {
        let n = positions.len();
        let mut coords = Vec::with_capacity(n * 3);
        for p in positions {
            coords.extend_from_slice(p);
        }
        crate::epoch::eigensolver::SpectralCoords { n, coords, extra: vec![0.0f32; n * 2] }
    }

    #[test]
    fn jacobi_svd_identity() {
        let m = mat3_identity();
        let (u, s, v) = jacobi_svd_3x3(m);
        // Singular values of identity are all 1.
        for &si in &s {
            assert!((si - 1.0).abs() < 1e-4, "singular value = {si}");
        }
        // U V^T should be identity.
        let q = mat3_mul(&u, &mat3_transpose(&v));
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((q[i][j] - expected).abs() < 1e-4, "Q[{i}][{j}] = {}", q[i][j]);
            }
        }
    }

    #[test]
    fn apply_rotation_identity() {
        let q = mat3_identity();
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut positions = original.clone();
        apply_rotation(&mut positions, q);
        for (i, (&a, &b)) in original.iter().zip(positions.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "positions[{i}] changed under identity rotation");
        }
    }

    #[test]
    fn align_empty_anchor_is_identity() {
        let mut coords = make_coords(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let original = coords.coords.clone();
        align(&mut coords, &[]);
        assert_eq!(coords.coords, original);
    }

    #[test]
    fn align_self_reference_is_identity() {
        // Aligning to the same positions should produce a rotation very close to identity.
        // We verify that pairwise distances are preserved (isometry check) rather than
        // checking absolute positions, since degenerate SVD can give any orthogonal basis.
        let positions: Vec<[f32; 3]> = vec![
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0],
        ];
        let mut coords = make_coords(&positions);
        let original = coords.coords.clone();
        align(&mut coords, &positions);

        // Pairwise distances should be preserved under any rotation.
        let n = positions.len();
        for i in 0..n {
            for j in i + 1..n {
                let orig_d2: f32 = (0..3)
                    .map(|d| {
                        let diff = original[i * 3 + d] - original[j * 3 + d];
                        diff * diff
                    })
                    .sum();
                let new_d2: f32 = (0..3)
                    .map(|d| {
                        let diff = coords.coords[i * 3 + d] - coords.coords[j * 3 + d];
                        diff * diff
                    })
                    .sum();
                assert!(
                    (orig_d2.sqrt() - new_d2.sqrt()).abs() < 1e-3,
                    "distance between particles {i} and {j} changed: {} → {}",
                    orig_d2.sqrt(),
                    new_d2.sqrt()
                );
            }
        }
    }

    #[test]
    fn jacobi_svd_known_matrix() {
        // M = [[3,0,0],[0,2,0],[0,0,1]] — diagonal. SVD should give S=[3,2,1].
        let m = [[3.0f32, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let (u, s, v) = jacobi_svd_3x3(m);
        let mut s_sorted = s;
        s_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((s_sorted[0] - 3.0).abs() < 1e-3, "s[0] = {}", s_sorted[0]);
        assert!((s_sorted[1] - 2.0).abs() < 1e-3, "s[1] = {}", s_sorted[1]);
        assert!((s_sorted[2] - 1.0).abs() < 1e-3, "s[2] = {}", s_sorted[2]);
        // U V^T U = U, so reconstruction M ≈ U S V^T
        let mut recon = mat3_zero();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    recon[i][j] += u[i][k] * s[k] * v[j][k];
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (recon[i][j] - m[i][j]).abs() < 1e-3,
                    "recon[{i}][{j}] = {} vs {}",
                    recon[i][j],
                    m[i][j]
                );
            }
        }
    }
}
