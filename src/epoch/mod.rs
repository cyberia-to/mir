//! Epoch work: ≤1 Hz background thread.
//! eigensolver → Procrustes → BVH → double-buffer swap.
//!
//! Step 2: eigensolver (needs acpu::sparse::csr_matvec)
//! Step 3: heat-kernel BVH (needs acpu::chebyshev)

pub mod eigensolver;
pub mod bvh;
pub mod procrustes;

use crate::graph::{Csr, ParticleIndex};

/// Epoch outputs — swapped atomically into the frame thread.
pub struct EpochState {
    /// n × 3 spectral coordinates (f32x3, row-major), Procrustes-aligned.
    pub positions: Vec<f32>,
    /// n focus values φ* (f32).
    pub focus: Vec<f32>,
    /// Per-particle cluster ID at each of the 4 τ scales.
    pub cluster_ids: Vec<[u32; 4]>,
}
