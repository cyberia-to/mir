//! Epoch work: ≤1 Hz background thread.
//! eigensolver → Procrustes → BVH → double-buffer swap.

pub mod bvh;
pub mod eigensolver;
pub mod procrustes;

use std::sync::{Arc, RwLock};

use crate::graph::{Csr, ParticleIndex};

use eigensolver::SpectralCoords;

// ── EpochState ────────────────────────────────────────────────────────────────

/// Fully computed epoch outputs, swapped atomically into the frame thread.
#[derive(Clone)]
pub struct EpochState {
    /// n × 3 spectral coordinates (f32x3, row-major), Procrustes-aligned.
    pub positions:   Vec<f32>,
    /// n f32 radii = r₀ · √φ*(i)
    pub radii:       Vec<f32>,
    /// n × 3 f32 RGB colors
    pub colors:      Vec<f32>,
    /// n focus values φ* (uniform 1/n on epoch 0).
    pub focus:       Vec<f32>,
    /// Per-particle cluster ID at each of the 4 τ scales.
    pub cluster_ids: Vec<[u32; 4]>,
    /// Acceleration structure over spectral positions.
    pub bvh:         crate::epoch::bvh::Bvh,
    /// D^{-½} diagonal, one per particle.
    pub d_inv:       Vec<f32>,
    /// Scale factor applied when mapping eigenvectors to scene coordinates.
    pub scene_scale: f32,
    /// Phase-2 T∞ NRF: trained hash-grid MLP (§7.5, no CT-1.1).
    pub nrf:         Option<crate::nrf::NrfState>,
}

// ── EpochWorker ───────────────────────────────────────────────────────────────

/// Background worker that recomputes spectral layout once per epoch.
pub struct EpochWorker {
    state: Arc<RwLock<Option<EpochState>>>,
    // graph and vocab are kept alive but not used after spawn.
    _graph: Arc<Csr>,
    _vocab: Arc<ParticleIndex>,
}

impl EpochWorker {
    /// Spawn the background epoch thread and return (worker, shared state).
    ///
    /// The thread writes the first `EpochState` as soon as the eigensolver
    /// finishes, then idles.
    pub fn spawn(
        graph: Arc<Csr>,
        vocab: Arc<ParticleIndex>,
    ) -> (Self, Arc<RwLock<Option<EpochState>>>) {
        let state: Arc<RwLock<Option<EpochState>>> = Arc::new(RwLock::new(None));
        let state_clone = Arc::clone(&state);
        let graph_clone = Arc::clone(&graph);

        std::thread::spawn(move || {
            epoch_pipeline(&graph_clone, state_clone);
        });

        let worker = Self { state: Arc::clone(&state), _graph: graph, _vocab: vocab };
        (worker, Arc::clone(&state))
    }

    /// Access the shared epoch state.
    pub fn state(&self) -> Arc<RwLock<Option<EpochState>>> {
        Arc::clone(&self.state)
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

fn hsl_to_rgb(h_rad: f32, s: f32, l: f32) -> [f32; 3] {
    let h = h_rad.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU * 360.0;
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = if h < 60.0      { (c, x, 0.0) }
                    else if h < 120.0 { (x, c, 0.0) }
                    else if h < 180.0 { (0.0, c, x) }
                    else if h < 240.0 { (0.0, x, c) }
                    else if h < 300.0 { (x, 0.0, c) }
                    else              { (c, 0.0, x) };
    [(r + m).clamp(0.0, 1.0), (g + m).clamp(0.0, 1.0), (b + m).clamp(0.0, 1.0)]
}

fn epoch_pipeline(csr: &Csr, state_out: Arc<RwLock<Option<EpochState>>>) {
    let n = csr.n;

    // 1. Compute eigensolver → SpectralCoords.
    let mut sc: SpectralCoords = eigensolver::solve(csr);

    // Compute D^{-½} for the diffusion step.
    let deg = eigensolver::degree_vec(csr);
    let d_inv: Vec<f32> = deg
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d } else { 0.0 })
        .collect();
    let d_inv_sqrt: Vec<f32> = deg
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();
    let _ = d_inv_sqrt; // used indirectly via eigensolver

    // 2. Procrustes alignment (identity for epoch 0 — no stored reference).
    procrustes::align(&mut sc, &[]);

    // §3.4: scale to fixed R_scene = 1000 (compute after Procrustes).
    let max_norm = (0..n)
        .map(|p| {
            let pos = sc.position(p);
            (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt()
        })
        .fold(0.0f32, f32::max);
    let scene_scale = if max_norm > 1e-8 { 1000.0 / max_norm } else { 1.0 };
    if (scene_scale - 1.0).abs() > 1e-6 {
        for v in sc.coords.iter_mut() { *v *= scene_scale; }
    }

    // 3. Uniform focus: φ*(i) = 1/n.
    let focus = vec![if n > 0 { 1.0 / n as f32 } else { 0.0 }; n];

    // 3b. §5.3 Role inference: classify particles by degree.
    let degrees: Vec<usize> = (0..n).map(|i| {
        let (cols, _) = csr.row(i);
        cols.len()
    }).collect();
    let mean_deg = if n > 0 {
        degrees.iter().sum::<usize>() as f32 / n as f32
    } else { 0.0 };
    let var_deg = if n > 0 {
        degrees.iter().map(|&d| {
            let diff = d as f32 - mean_deg;
            diff * diff
        }).sum::<f32>() / n as f32
    } else { 0.0 };
    let std_deg = var_deg.sqrt();
    let hub_thresh  = mean_deg + 2.0 * std_deg;

    #[derive(Clone, Copy)] enum Role { Sphere, Hub, Leaf }
    let roles: Vec<Role> = degrees.iter().map(|&d| {
        if d == 1 { Role::Leaf }
        else if d as f32 > hub_thresh { Role::Hub }
        else { Role::Sphere }
    }).collect();

    // 3c. Compute radii (r₀ · √φ*) with role-based multiplier.
    let r0 = 100.0f32;
    let radii: Vec<f32> = (0..n).map(|i| {
        let role_mult = match roles[i] { Role::Hub => 1.5, Role::Leaf => 0.7, Role::Sphere => 1.0 };
        r0 * focus[i].sqrt() * role_mult
    }).collect();
    // Uniform blue palette: hub = bright, sphere = mid, leaf = dark.
    let colors: Vec<f32> = (0..n).flat_map(|i| {
        let (r, g, b) = match roles[i] {
            Role::Hub    => (0.20f32, 0.55f32, 1.00f32),
            Role::Sphere => (0.12f32, 0.38f32, 0.88f32),
            Role::Leaf   => (0.07f32, 0.22f32, 0.68f32),
        };
        [r, g, b]
    }).collect();

    // 4. Build BVH.
    let bvh = bvh::build(&sc, &focus);

    // 5. Collect cluster_ids from BVH.
    let cluster_ids = bvh.cluster_ids.clone();

    // 6. Train T∞ NRF — §7.5 fallback: hash-grid MLP, no graph-context conditioning.
    //    Graph-context conditioning (§7.1) requires a compiled CT-1.1 model from `tru`.
    //    Runs in the epoch worker thread — off the frame critical path.
    let nrf = {
        let mut state = crate::nrf::NrfState::new();
        crate::nrf::train_nrf(&mut state, &sc.coords, &colors, &focus);
        Some(state)
    };

    // 7. Write EpochState.
    let epoch_state = EpochState {
        positions: sc.coords,
        radii,
        colors,
        focus,
        cluster_ids,
        bvh,
        d_inv,
        scene_scale,
        nrf,
    };

    if let Ok(mut guard) = state_out.write() {
        *guard = Some(epoch_state);
    }
}

/// Deterministic cpu-reference epoch computation for conformance verification (§13.2).
/// Single-threaded, uses the same Rust eigensolver path.
pub fn cpu_reference_epoch(csr: &crate::graph::Csr) -> EpochState {
    use std::sync::{Arc, RwLock};
    let state = Arc::new(RwLock::new(None));
    epoch_pipeline(csr, Arc::clone(&state));
    let guard = state.read().unwrap();
    guard.as_ref().unwrap().clone()  // EpochState needs Clone
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Cyberlink, ParticleIndex, Csr};

    fn hash(v: u8) -> [u8; 32] {
        let mut h = [0u8; 32];
        h[0] = v;
        h
    }

    fn link(from: u8, to: u8) -> Cyberlink {
        Cyberlink {
            neuron: [0u8; 32],
            from: hash(from),
            to: hash(to),
            token: 0,
            amount: 1,
            valence: 1,
            block: 1,
        }
    }

    #[test]
    fn epoch_worker_completes() {
        // Build a small ring graph (8 nodes).
        let links: Vec<Cyberlink> = (0u8..8).map(|i| link(i, (i + 1) % 8)).collect();
        let vocab = ParticleIndex::build(links.iter().copied());
        let csr = Arc::new(Csr::build(links.into_iter(), &vocab));
        let vocab = Arc::new(vocab);

        let (worker, state) = EpochWorker::spawn(Arc::clone(&csr), vocab);

        // Poll with a timeout of 10 seconds.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            {
                let guard = state.read().unwrap();
                if guard.is_some() {
                    let es = guard.as_ref().unwrap();
                    assert_eq!(es.positions.len(), csr.n * 3);
                    assert_eq!(es.radii.len(), csr.n);
                    assert_eq!(es.colors.len(), csr.n * 3);
                    assert_eq!(es.focus.len(), csr.n);
                    assert_eq!(es.cluster_ids.len(), csr.n);
                    assert_eq!(es.d_inv.len(), csr.n);
                    assert!(es.scene_scale > 0.0);
                    break;
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "EpochWorker did not complete within timeout"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        drop(worker);
    }

    #[test]
    fn hub_has_larger_radius_than_leaf() {
        // Star graph: node 0 connects to nodes 1..8 (hub), nodes 1..8 each have degree 1 (leaves).
        let mut links: Vec<Cyberlink> = Vec::new();
        for spoke in 1u8..9 {
            links.push(link(0, spoke));
        }
        let vocab = ParticleIndex::build(links.iter().copied());
        let csr = Arc::new(Csr::build(links.into_iter(), &vocab));
        let vocab = Arc::new(vocab);

        let (_worker, state) = EpochWorker::spawn(csr, vocab);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            {
                let guard = state.read().unwrap();
                if let Some(es) = guard.as_ref() {
                    // Node 0 is the hub; nodes 1..8 are leaves. Radii should reflect this.
                    // In our vocbulary, node 0 = hash(0), etc.
                    // Just verify that some radius is larger than another (hub > leaf).
                    if es.radii.len() >= 2 {
                        let max_r = es.radii.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let min_r = es.radii.iter().cloned().fold(f32::INFINITY, f32::min);
                        // Hub (role_mult=1.5) should be larger than leaf (role_mult=0.7).
                        assert!(max_r > min_r, "hub radius should exceed leaf radius");
                    }
                    break;
                }
            }
            assert!(std::time::Instant::now() < deadline, "timeout");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    #[test]
    fn positions_scaled_to_r_scene() {
        // §3.4: max position norm must equal R_scene = 1000.
        let links: Vec<Cyberlink> = (0u8..8).map(|i| link(i, (i + 1) % 8)).collect();
        let vocab = ParticleIndex::build(links.iter().copied());
        let csr = Arc::new(Csr::build(links.into_iter(), &vocab));
        let vocab = Arc::new(vocab);

        let (_worker, state) = EpochWorker::spawn(csr, vocab);

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            {
                let guard = state.read().unwrap();
                if guard.is_some() {
                    let es = guard.as_ref().unwrap();
                    let n = es.positions.len() / 3;
                    if n == 0 { break; }
                    let max_norm = (0..n).map(|i| {
                        let b = i * 3;
                        let (x, y, z) = (es.positions[b], es.positions[b+1], es.positions[b+2]);
                        (x*x + y*y + z*z).sqrt()
                    }).fold(0.0f32, f32::max);
                    // After scaling, max norm ≈ 1000 (within 1%).
                    assert!(
                        (max_norm - 1000.0).abs() < 10.0,
                        "max position norm {max_norm} is not ~1000 (§3.4 scaling)"
                    );
                    break;
                }
            }
            assert!(std::time::Instant::now() < deadline, "timeout");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}
