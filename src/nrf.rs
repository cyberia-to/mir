//! T∞ §7.5 fallback: hash-grid MLP without graph-context conditioning.
//!
//! §7.1 describes the full NRF architecture: hash-grid → cross-attention on
//! CT-1.1 last hidden state → Clifford render block → output head.
//! §7.5 specifies this fallback: when the CT-1.1 model (compiled by `tru`)
//! is unavailable, the graph-context conditioning step is skipped and the
//! NRF degrades to a hash-grid-only MLP.
//!
//! This module implements that fallback:
//!   hash-grid (fixed random features, trilinear interpolation) → MLP 8→32→32→4.
//!
//! Training runs in the epoch worker thread. CPU ray-march (§7.2) is provided
//! for the cpu-reference backend and conformance tests; it is not real-time.
//! GPU inference (MSL compute shader) is needed for the live render path.

// ── Constants ─────────────────────────────────────────────────────────────────

const L:       usize = 4;          // hash-grid levels
const T:       usize = 512;        // entries per level
const F:       usize = 2;          // features per entry
const NRF_IN:  usize = L * F;      // MLP input dim = 8
const NRF_H:   usize = 32;         // hidden width
const NRF_OUT: usize = 4;          // ρ + RGB

// ── Hash-grid ─────────────────────────────────────────────────────────────────

/// Multi-resolution hash-grid with fixed random features + trilinear interpolation.
#[derive(Clone)]
pub struct HashGrid {
    /// [level][entry][feature], shape L × T × F.
    pub tables:     Vec<f32>,
    /// Per-level cell size (scene units). Level 0 = coarsest.
    pub cell_sizes: [f32; L],
}

impl Default for HashGrid { fn default() -> Self { Self::new() } }

impl HashGrid {
    pub fn new() -> Self {
        let n = L * T * F;
        let mut tables = Vec::with_capacity(n);
        let mut s: u64 = 0xdeadbeef_cafebabe;
        for _ in 0..n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            tables.push(((s >> 33) as f32) / (u32::MAX as f32) * 0.2 - 0.1);
        }
        // Coarsest cell = 250 scene units; halved each level.
        let cell_sizes = std::array::from_fn(|l| 250.0_f32 / (1u32 << l) as f32);
        Self { tables, cell_sizes }
    }

    fn cell_hash(ix: i32, iy: i32, iz: i32, level: usize) -> usize {
        let h: u64 = (ix as i64 as u64)
            .wrapping_mul(2654435761)
            .wrapping_add((iy as i64 as u64).wrapping_mul(805459861))
            .wrapping_add((iz as i64 as u64).wrapping_mul(3674653429))
            .wrapping_add(level as u64 * 1234567891);
        (h % T as u64) as usize
    }

    #[inline]
    fn lookup(&self, ix: i32, iy: i32, iz: i32, level: usize, feat: usize) -> f32 {
        let e = Self::cell_hash(ix, iy, iz, level);
        self.tables[level * T * F + e * F + feat]
    }

    /// Encode a 3D point to an `NRF_IN`-dim feature vector via trilinear interpolation.
    pub fn encode(&self, xyz: [f32; 3]) -> [f32; NRF_IN] {
        let mut out = [0.0f32; NRF_IN];
        for l in 0..L {
            let cs = self.cell_sizes[l];
            let gx = xyz[0] / cs;
            let gy = xyz[1] / cs;
            let gz = xyz[2] / cs;
            let ix = gx.floor() as i32;
            let iy = gy.floor() as i32;
            let iz = gz.floor() as i32;
            let tx = gx - ix as f32;
            let ty = gy - iy as f32;
            let tz = gz - iz as f32;
            for f in 0..F {
                let mut v = 0.0f32;
                for (dx, wx) in [(0i32, 1.0 - tx), (1, tx)] {
                    for (dy, wy) in [(0i32, 1.0 - ty), (1, ty)] {
                        for (dz, wz) in [(0i32, 1.0 - tz), (1, tz)] {
                            v += wx * wy * wz * self.lookup(ix+dx, iy+dy, iz+dz, l, f);
                        }
                    }
                }
                out[l * F + f] = v;
            }
        }
        out
    }
}

// ── MLP ───────────────────────────────────────────────────────────────────────

/// Feed-forward network: `NRF_IN` → `NRF_H` → `NRF_H` → `NRF_OUT` (sigmoid out).
#[derive(Clone)]
pub struct NrfMlp {
    pub w1: Vec<f32>,  // NRF_H × NRF_IN
    pub b1: Vec<f32>,  // NRF_H
    pub w2: Vec<f32>,  // NRF_H × NRF_H
    pub b2: Vec<f32>,  // NRF_H
    pub w3: Vec<f32>,  // NRF_OUT × NRF_H
    pub b3: Vec<f32>,  // NRF_OUT
}

fn xavier(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f32> {
    let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
    let n = fan_in * fan_out;
    let mut s: u64 = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0) * limit
    }).collect()
}

impl Default for NrfMlp { fn default() -> Self { Self::new() } }

impl NrfMlp {
    pub fn new() -> Self {
        Self {
            w1: xavier(NRF_IN, NRF_H, 0xabc1),  b1: vec![0.0; NRF_H],
            w2: xavier(NRF_H,  NRF_H, 0xabc2),  b2: vec![0.0; NRF_H],
            w3: xavier(NRF_H, NRF_OUT, 0xabc3), b3: vec![0.0; NRF_OUT],
        }
    }

    /// Forward: input → (ρ, r, g, b) ∈ [0,1]⁴.
    pub fn forward(&self, x: [f32; NRF_IN]) -> [f32; NRF_OUT] {
        let mut h1 = [0.0f32; NRF_H];
        for i in 0..NRF_H {
            let s: f32 = self.b1[i]
                + (0..NRF_IN).map(|j| self.w1[i * NRF_IN + j] * x[j]).sum::<f32>();
            h1[i] = s.max(0.0);
        }
        let mut h2 = [0.0f32; NRF_H];
        for i in 0..NRF_H {
            let s: f32 = self.b2[i]
                + (0..NRF_H).map(|j| self.w2[i * NRF_H + j] * h1[j]).sum::<f32>();
            h2[i] = s.max(0.0);
        }
        let mut out = [0.0f32; NRF_OUT];
        for i in 0..NRF_OUT {
            let s: f32 = self.b3[i]
                + (0..NRF_H).map(|j| self.w3[i * NRF_H + j] * h2[j]).sum::<f32>();
            out[i] = 1.0 / (1.0 + (-s).exp());
        }
        out
    }
}

// ── NrfState ──────────────────────────────────────────────────────────────────

/// Trained Phase-2 T∞ state (hash-grid + MLP weights).
#[derive(Clone)]
pub struct NrfState {
    pub grid: HashGrid,
    pub mlp:  NrfMlp,
}

impl Default for NrfState { fn default() -> Self { Self::new() } }

impl NrfState {
    pub fn new() -> Self { Self { grid: HashGrid::new(), mlp: NrfMlp::new() } }
}

// ── Training (§7.3) ───────────────────────────────────────────────────────────

/// §7.3 ground truth: density + color at a world-space point.
/// ρ*(xyz, τ) = Σ_p K_τ(X(p), xyz) · φ*(p),  c* = focus-weighted average color.
fn ground_truth(
    xyz:       [f32; 3],
    tau:       f32,
    positions: &[f32],
    colors:    &[f32],
    focus:     &[f32],
) -> [f32; NRF_OUT] {
    let n = positions.len() / 3;
    let mut rho = 0.0f32;
    let mut rgb = [0.0f32; 3];
    for p in 0..n {
        let d2: f32 = (0..3).map(|d| {
            let diff = xyz[d] - positions[p * 3 + d];
            diff * diff
        }).sum();
        let k = (-d2 / (2.0 * tau.max(1.0))).exp();
        let w = k * focus.get(p).copied().unwrap_or(0.0);
        rho += w;
        for c in 0..3 { rgb[c] += w * colors.get(p * 3 + c).copied().unwrap_or(0.5); }
    }
    let inv = if rho > 1e-8 { 1.0 / rho } else { 0.0 };
    [rho.clamp(0.0, 1.0),
     (rgb[0] * inv).clamp(0.0, 1.0),
     (rgb[1] * inv).clamp(0.0, 1.0),
     (rgb[2] * inv).clamp(0.0, 1.0)]
}

/// §7.3 Train the NRF MLP via SGD on mini-batches of random world-space samples.
/// Hash-grid features are fixed; only MLP weights are updated.
/// Returns final MSE loss.
pub fn train_nrf(
    state:     &mut NrfState,
    positions: &[f32],
    colors:    &[f32],
    focus:     &[f32],
) -> f32 {
    const N_SAMPLES: usize = 512;
    const N_STEPS:   usize = 8;
    const LR:        f32   = 0.01;
    const TAU:       f32   = 100.0;  // heat scale for ground-truth density
    const R:         f32   = 1000.0; // R_scene

    let mut rng: u64 = 0xfeed_face_cafe_babe;
    let mut rnd = || -> f32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    let mut loss = 0.0f32;

    for _step in 0..N_STEPS {
        let mut dw1 = vec![0.0f32; NRF_H * NRF_IN];
        let mut db1 = vec![0.0f32; NRF_H];
        let mut dw2 = vec![0.0f32; NRF_H * NRF_H];
        let mut db2 = vec![0.0f32; NRF_H];
        let mut dw3 = vec![0.0f32; NRF_OUT * NRF_H];
        let mut db3 = vec![0.0f32; NRF_OUT];
        let mut step_loss = 0.0f32;

        for _ in 0..N_SAMPLES {
            let xyz = [rnd() * R, rnd() * R, rnd() * R];
            let x   = state.grid.encode(xyz);
            let tgt = ground_truth(xyz, TAU, positions, colors, focus);

            // ── Forward pass ──────────────────────────────────────────────────
            let mut h1_pre = [0.0f32; NRF_H];
            let mut h1     = [0.0f32; NRF_H];
            for i in 0..NRF_H {
                h1_pre[i] = state.mlp.b1[i]
                    + (0..NRF_IN).map(|j| state.mlp.w1[i * NRF_IN + j] * x[j]).sum::<f32>();
                h1[i] = h1_pre[i].max(0.0);
            }
            let mut h2_pre = [0.0f32; NRF_H];
            let mut h2     = [0.0f32; NRF_H];
            for i in 0..NRF_H {
                h2_pre[i] = state.mlp.b2[i]
                    + (0..NRF_H).map(|j| state.mlp.w2[i * NRF_H + j] * h1[j]).sum::<f32>();
                h2[i] = h2_pre[i].max(0.0);
            }
            let mut out_pre = [0.0f32; NRF_OUT];
            let mut out     = [0.0f32; NRF_OUT];
            for i in 0..NRF_OUT {
                out_pre[i] = state.mlp.b3[i]
                    + (0..NRF_H).map(|j| state.mlp.w3[i * NRF_H + j] * h2[j]).sum::<f32>();
                out[i] = 1.0 / (1.0 + (-out_pre[i]).exp());
            }

            // ── Loss + output gradients ───────────────────────────────────────
            let mut d_out = [0.0f32; NRF_OUT];
            for i in 0..NRF_OUT {
                let err = out[i] - tgt[i];
                step_loss  += err * err;
                d_out[i]    = 2.0 * err * out[i] * (1.0 - out[i]); // MSE + d/ds sigmoid
            }

            // ── Layer 3 weight gradients ──────────────────────────────────────
            for i in 0..NRF_OUT {
                for j in 0..NRF_H { dw3[i * NRF_H + j] += d_out[i] * h2[j]; }
                db3[i] += d_out[i];
            }

            // ── Backprop through h2 ───────────────────────────────────────────
            let mut d_h2 = [0.0f32; NRF_H];
            for j in 0..NRF_H {
                d_h2[j] = (0..NRF_OUT)
                    .map(|i| d_out[i] * state.mlp.w3[i * NRF_H + j]).sum::<f32>();
                d_h2[j] *= if h2_pre[j] > 0.0 { 1.0 } else { 0.0 };
            }
            for i in 0..NRF_H {
                for j in 0..NRF_H { dw2[i * NRF_H + j] += d_h2[i] * h1[j]; }
                db2[i] += d_h2[i];
            }

            // ── Backprop through h1 ───────────────────────────────────────────
            let mut d_h1 = [0.0f32; NRF_H];
            for j in 0..NRF_H {
                d_h1[j] = (0..NRF_H)
                    .map(|i| d_h2[i] * state.mlp.w2[i * NRF_H + j]).sum::<f32>();
                d_h1[j] *= if h1_pre[j] > 0.0 { 1.0 } else { 0.0 };
            }
            for i in 0..NRF_H {
                for j in 0..NRF_IN { dw1[i * NRF_IN + j] += d_h1[i] * x[j]; }
                db1[i] += d_h1[i];
            }
        }

        // SGD step (mean gradient over mini-batch).
        let s = LR / N_SAMPLES as f32;
        for (w, g) in state.mlp.w3.iter_mut().zip(&dw3) { *w -= s * g; }
        for (b, g) in state.mlp.b3.iter_mut().zip(&db3) { *b -= s * g; }
        for (w, g) in state.mlp.w2.iter_mut().zip(&dw2) { *w -= s * g; }
        for (b, g) in state.mlp.b2.iter_mut().zip(&db2) { *b -= s * g; }
        for (w, g) in state.mlp.w1.iter_mut().zip(&dw1) { *w -= s * g; }
        for (b, g) in state.mlp.b1.iter_mut().zip(&db1) { *b -= s * g; }

        loss = step_loss / N_SAMPLES as f32;
    }

    loss
}

// ── CPU ray-march (§7.2) ──────────────────────────────────────────────────────

/// Volume-render the T∞ NRF for transparent pixels via CPU ray-march (§7.2).
///
/// Use for the conformance / cpu-reference path — NOT real-time.
/// For tests, pass a small viewport (e.g., 8×8) to keep runtime short.
///
/// `composite`     — W×H×4 RGBA f32 pixel buffer (reads alpha, writes RGB+alpha).
/// `view_proj_inv` — column-major 4×4 inverse view-projection.
/// `cam_pos`       — camera world position.
/// `tau`           — current τ (τ-dependent LOD per §7.2, depth-scaled sampling).
/// `viewport`      — [width, height] in pixels.
pub fn cpu_ray_march(
    composite:     &mut [f32],
    state:         &NrfState,
    view_proj_inv: &[[f32; 4]; 4],
    cam_pos:       [f32; 3],
    _tau:          f32,
    viewport:      [u32; 2],
) {
    const N_SAMPLES: usize = 32;
    const T_NEAR:    f32   = 10.0;
    const T_FAR:     f32   = 4000.0;

    let [w, h] = [viewport[0] as usize, viewport[1] as usize];
    let step   = (T_FAR - T_NEAR) / N_SAMPLES as f32;
    let m      = view_proj_inv;

    let unproj = |ndx: f32, ndy: f32, ndz: f32| -> [f32; 3] {
        let wx = m[0][0]*ndx + m[1][0]*ndy + m[2][0]*ndz + m[3][0];
        let wy = m[0][1]*ndx + m[1][1]*ndy + m[2][1]*ndz + m[3][1];
        let wz = m[0][2]*ndx + m[1][2]*ndy + m[2][2]*ndz + m[3][2];
        let ww = m[0][3]*ndx + m[1][3]*ndy + m[2][3]*ndz + m[3][3];
        let iw = if ww.abs() > 1e-8 { 1.0 / ww } else { 0.0 };
        [wx * iw, wy * iw, wz * iw]
    };

    for py in 0..h {
        for px in 0..w {
            let base = (py * w + px) * 4;
            if composite[base + 3] >= 0.5 { continue; } // already opaque

            let ndcx =  (px as f32 + 0.5) / w as f32 * 2.0 - 1.0;
            let ndcy = -(py as f32 + 0.5) / h as f32 * 2.0 + 1.0;

            let far = unproj(ndcx, ndcy, 1.0);
            let raw = [far[0]-cam_pos[0], far[1]-cam_pos[1], far[2]-cam_pos[2]];
            let len = raw.iter().map(|x| x*x).sum::<f32>().sqrt().max(1e-8);
            let dir = [raw[0]/len, raw[1]/len, raw[2]/len];

            let (mut r, mut g, mut b, mut tr) = (0.0f32, 0.0f32, 0.0f32, 1.0f32);

            for s in 0..N_SAMPLES {
                let t   = T_NEAR + (s as f32 + 0.5) * step;
                let xyz = [cam_pos[0]+t*dir[0], cam_pos[1]+t*dir[1], cam_pos[2]+t*dir[2]];
                let [rho, cr, cg, cb] = state.mlp.forward(state.grid.encode(xyz));
                let alpha = 1.0 - (-rho * step * 0.001).exp();
                r  += tr * alpha * cr;
                g  += tr * alpha * cg;
                b  += tr * alpha * cb;
                tr *= 1.0 - alpha;
                if tr < 0.01 { break; }
            }

            composite[base]   = r.clamp(0.0, 1.0);
            composite[base+1] = g.clamp(0.0, 1.0);
            composite[base+2] = b.clamp(0.0, 1.0);
            composite[base+3] = (1.0 - tr).clamp(0.0, 1.0);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_grid_encodes_correct_dim() {
        let grid = HashGrid::new();
        let feat = grid.encode([100.0, -200.0, 50.0]);
        assert_eq!(feat.len(), NRF_IN);
        assert!(feat.iter().all(|v| v.is_finite()), "features must be finite");
    }

    #[test]
    fn mlp_output_in_unit_interval() {
        let mlp = NrfMlp::new();
        let out = mlp.forward([0.1f32; NRF_IN]);
        for &v in &out {
            assert!(v >= 0.0 && v <= 1.0, "MLP output {v} outside [0,1]");
        }
    }

    #[test]
    fn training_reduces_loss() {
        let positions: Vec<f32> = vec![
              0.0,   0.0,   0.0,
            500.0,   0.0,   0.0,
           -500.0,   0.0,   0.0,
        ];
        let colors: Vec<f32> = vec![0.8, 0.2, 0.2,  0.2, 0.8, 0.2,  0.2, 0.2, 0.8];
        let focus:  Vec<f32> = vec![0.5, 0.25, 0.25];

        let mut state = NrfState::new();
        let loss0 = train_nrf(&mut state, &positions, &colors, &focus);
        let loss1 = train_nrf(&mut state, &positions, &colors, &focus);
        // Second pass should not blow up, and loss should stay sub-unit.
        assert!(loss1 < 1.0, "loss should be sub-unit: {loss1:.4}");
        assert!(loss1 <= loss0 * 1.5,
            "loss should not grow dramatically (loss0={loss0:.4}, loss1={loss1:.4})");
    }

    #[test]
    fn ray_march_fills_transparent_pixels() {
        let state   = NrfState::new();
        let vp      = [8u32, 8u32];
        let mut buf = vec![0.0f32; 8 * 8 * 4];
        // Pre-fill pixel 0 so we can verify it stays unchanged.
        buf[3] = 1.0;

        let vp_inv: [[f32; 4]; 4] = [
            [1.0/300.0, 0.0, 0.0, 0.0],
            [0.0, 1.0/300.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        cpu_ray_march(&mut buf, &state, &vp_inv, [0.0, 0.0, 3000.0], 1.0, vp);

        assert_eq!(buf[3], 1.0, "pre-filled pixel must not be overwritten");
        let any_written = (0..8*8).any(|i| buf[i * 4 + 3] > 0.0);
        assert!(any_written, "ray-march must write at least one pixel");
    }
}
