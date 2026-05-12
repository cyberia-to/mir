//! Epoch work: ≤1 Hz background thread.
//! eigensolver → Procrustes → BVH → double-buffer swap.

pub mod bvh;
pub mod eigensolver;
pub mod procrustes;

use std::sync::{Arc, RwLock};

use crate::graph::{Csr, ParticleIndex};

use bvh::Bvh;
use eigensolver::SpectralCoords;

// ── EpochState ────────────────────────────────────────────────────────────────

/// Fully computed epoch outputs, swapped atomically into the frame thread.
pub struct EpochState {
    /// n × 3 spectral coordinates (f32x3, row-major), Procrustes-aligned.
    pub positions: Vec<f32>,
    /// n focus values φ* (uniform 1/n on epoch 0).
    pub focus: Vec<f32>,
    /// Per-particle cluster ID at each of the 4 τ scales.
    pub cluster_ids: Vec<[u32; 4]>,
    /// Acceleration structure over spectral positions.
    pub bvh: Bvh,
    /// D^{-½} diagonal, one per particle.
    pub d_inv: Vec<f32>,
    /// Scale factor applied when mapping eigenvectors to scene coordinates.
    pub scene_scale: f32,
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

    // Record scene scale from max position norm.
    let max_norm = (0..n)
        .map(|p| {
            let pos = sc.position(p);
            (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt()
        })
        .fold(0.0f32, f32::max);
    let scene_scale = if max_norm > 1e-8 { 1000.0 / max_norm } else { 1.0 };

    // 2. Procrustes alignment (identity for epoch 0 — no stored reference).
    procrustes::align(&mut sc, &[]);

    // 3. Uniform focus: φ*(i) = 1/n.
    let focus = vec![if n > 0 { 1.0 / n as f32 } else { 0.0 }; n];

    // 4. Build BVH.
    let bvh = bvh::build(&sc, &focus);

    // 5. Collect cluster_ids from BVH.
    let cluster_ids = bvh.cluster_ids.clone();

    // 6. Write EpochState.
    let epoch_state = EpochState {
        positions: sc.coords,
        focus,
        cluster_ids,
        bvh,
        d_inv,
        scene_scale,
    };

    if let Ok(mut guard) = state_out.write() {
        *guard = Some(epoch_state);
    }
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
}
