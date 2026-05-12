//! Heat-kernel BVH: four τ scales → 4-level hierarchy.
//!
//! Each level is a partition of particles via Chebyshev-approximated heat
//! diffusion H_τ on the graph Laplacian (R-1.0 §10). Same structure serves
//! both spatial frustum cull (AABB) and topological LOD (cluster ID).
//!
//! Depends on acpu::chebyshev (step 3). Placeholder until that kernel lands.

use crate::epoch::eigensolver::SpectralCoords;
use crate::graph::Csr;

pub const TAU_SCALES: [f32; 4] = [1.0, 10.0, 100.0, 1000.0];

#[derive(Default)]
pub struct BvhNode {
    pub aabb_min:   [f32; 3],
    pub aabb_max:   [f32; 3],
    pub focus_sum:  f32,
    pub child_start: u32, // index into nodes array
    pub child_count: u8,
}

pub struct Bvh {
    pub nodes: Vec<BvhNode>,
    /// Per-particle cluster ID at each τ scale (4 levels).
    pub cluster_ids: Vec<[u32; 4]>,
}

/// Placeholder — Chebyshev heat-kernel clustering follows in step 3.
pub fn build(_csr: &Csr, _coords: &SpectralCoords, _focus: &[f32]) -> Bvh {
    todo!("heat-kernel BVH — step 3")
}
