//! Heat-kernel BVH: four τ scales → 4-level spatial hierarchy.
//!
//! At each of 4 τ scales {1, 10, 100, 1000}, particles are clustered by
//! spatial grid on their spectral coordinates, dividing the [-1000,1000]³
//! bounding box into cells whose size scales as √τ. Each cell = one cluster.
//!
//! Current implementation: spatial grid clustering. §10.1 specifies heat-kernel
//! diffusion-distance clustering (Chebyshev), which requires the full tri-kernel.
//!
//! BvhNode tree is built bottom-up from the finest-scale clusters.

use crate::epoch::eigensolver::SpectralCoords;

pub const TAU_SCALES: [f32; 4] = [1.0, 10.0, 100.0, 1000.0];

/// One node in the BVH tree.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct BvhNode {
    pub aabb_min:    [f32; 3],  // 12 bytes
    pub focus_sum:   f32,        //  4 bytes  ← moved before aabb_max for GPU alignment
    pub aabb_max:    [f32; 3],  // 12 bytes
    pub child_start: u32,        //  4 bytes
    pub child_count: u32,        //  4 bytes  ← was u8, now u32
    pub is_leaf:     u32,        //  4 bytes  (1=leaf, 0=internal)
    pub leaf_start:  u32,        //  4 bytes  (first particle idx in leaf)
    pub leaf_count:  u32,        //  4 bytes  (particle count in leaf)
}

/// Acceleration structure with 4-level cluster hierarchy.
#[derive(Clone)]
pub struct Bvh {
    pub nodes: Vec<BvhNode>,
    /// Per-particle cluster ID at each τ scale (4 levels).
    pub cluster_ids: Vec<[u32; 4]>,
}

// ── Grid clustering ───────────────────────────────────────────────────────────

/// Assign cluster IDs via a regular 3D spatial grid on spectral coordinates.
///
/// Cell size at τ scale: cell_size ∝ √τ, calibrated so that τ=1 gives
/// a grid of roughly 20 cells per axis in the [-1000,1000]³ box.
fn grid_cluster(coords: &SpectralCoords, tau: f32) -> Vec<u32> {
    let n = coords.n;
    if n == 0 {
        return Vec::new();
    }

    // Cell size: 100 * sqrt(tau) units.  At τ=1 → 100 (20 cells/axis in 2000-unit box).
    let cell_size = 100.0_f32 * tau.sqrt();

    // Map each particle to a 3D grid cell index, then assign a linear cluster ID.
    // Grid is unbounded; we use a hash-map approach for generality.
    let mut cell_to_id: std::collections::HashMap<(i32, i32, i32), u32> =
        std::collections::HashMap::new();
    let mut ids = vec![0u32; n];
    let mut next_id = 0u32;

    for p in 0..n {
        let pos = coords.position(p);
        let ix = (pos[0] / cell_size).floor() as i32;
        let iy = (pos[1] / cell_size).floor() as i32;
        let iz = (pos[2] / cell_size).floor() as i32;
        let entry = cell_to_id.entry((ix, iy, iz)).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        ids[p] = *entry;
    }

    ids
}

// ── AABB helpers ──────────────────────────────────────────────────────────────

fn aabb_expand(mn: &mut [f32; 3], mx: &mut [f32; 3], pos: [f32; 3]) {
    for d in 0..3 {
        mn[d] = mn[d].min(pos[d]);
        mx[d] = mx[d].max(pos[d]);
    }
}

fn aabb_empty() -> ([f32; 3], [f32; 3]) {
    ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3])
}

// ── BVH builder ───────────────────────────────────────────────────────────────

// ── Canonical cluster label assignment (§10.2) ────────────────────────────────

fn canonicalize_ids(ids: &mut Vec<u32>, focus: &[f32]) -> u32 {
    let max_id = ids.iter().copied().max().unwrap_or(0);
    let n_clusters = (max_id as usize) + 1;

    let mut cluster_focus = vec![0.0f32; n_clusters];
    let mut cluster_min_p = vec![usize::MAX; n_clusters];

    for (p, &cid) in ids.iter().enumerate() {
        let c = cid as usize;
        if c < n_clusters {
            cluster_focus[c] += focus.get(p).copied().unwrap_or(0.0);
            cluster_min_p[c] = cluster_min_p[c].min(p);
        }
    }

    // Sort clusters: by -focus (descending), then by min_particle (ascending).
    let mut order: Vec<usize> = (0..n_clusters).collect();
    order.sort_by(|&a, &b| {
        cluster_focus[b].partial_cmp(&cluster_focus[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(cluster_min_p[a].cmp(&cluster_min_p[b]))
    });

    // Build remap: old_id → new_id
    let mut remap = vec![0u32; n_clusters];
    for (new_id, &old_id) in order.iter().enumerate() {
        remap[old_id] = new_id as u32;
    }

    // Apply remap.
    for cid in ids.iter_mut() {
        *cid = remap[*cid as usize];
    }

    max_id
}

/// Build the BVH from spectral coordinates and per-particle focus weights.
///
/// Returns a Bvh with nodes covering 4 τ scales and per-particle cluster IDs.
pub fn build(coords: &SpectralCoords, focus: &[f32]) -> Bvh {
    let n = coords.n;

    // Compute cluster IDs at each τ scale.
    let mut cluster_ids = vec![[0u32; 4]; n];
    let mut all_level_ids: [Vec<u32>; 4] = std::array::from_fn(|_| Vec::new());

    for (level, &tau) in TAU_SCALES.iter().enumerate() {
        let mut ids = grid_cluster(coords, tau);
        canonicalize_ids(&mut ids, focus);
        for p in 0..n {
            cluster_ids[p][level] = ids[p];
        }
        all_level_ids[level] = ids;
    }

    // Build BvhNodes: one node per cluster at each level, then parent nodes
    // that contain clusters from the previous finer level.
    //
    // Structure:
    //   nodes[0..n_c0]         — finest level τ=1 clusters (leaf nodes)
    //   nodes[n_c0..n_c0+n_c1] — τ=10 clusters (parent of τ=1 clusters)
    //   ...
    //
    // Each leaf cluster aggregates member particle positions + focus.
    // Parent nodes aggregate child cluster AABBs.

    let mut nodes: Vec<BvhNode> = Vec::new();

    // Level 0 (finest): one BvhNode per τ=1 cluster.
    let ids0 = &all_level_ids[0];
    let n_c0 = ids0.iter().copied().max().map(|m| m + 1).unwrap_or(0) as usize;
    let offset0 = 0usize;

    // Initialize level-0 nodes (leaves).
    let mut l0_nodes: Vec<BvhNode> = (0..n_c0)
        .map(|_| {
            let (mn, mx) = aabb_empty();
            BvhNode {
                aabb_min: mn, focus_sum: 0.0, aabb_max: mx,
                child_start: 0, child_count: 0,
                is_leaf: 1, leaf_start: 0, leaf_count: 0,
            }
        })
        .collect();

    for p in 0..n {
        let cid = ids0[p] as usize;
        if cid < n_c0 {
            let pos = coords.position(p);
            let foc = focus.get(p).copied().unwrap_or(0.0);
            let node = &mut l0_nodes[cid];
            aabb_expand(&mut node.aabb_min, &mut node.aabb_max, pos);
            node.focus_sum += foc;
        }
    }
    nodes.extend_from_slice(&l0_nodes);

    // Levels 1..3: cluster the previous level's cluster IDs.
    let mut prev_offset = offset0;
    let mut prev_ids = ids0.clone();

    for level in 1..4 {
        let cur_ids = &all_level_ids[level];
        let n_cur = cur_ids.iter().copied().max().map(|m| m + 1).unwrap_or(0) as usize;
        let cur_offset = nodes.len();

        let mut cur_nodes: Vec<BvhNode> = (0..n_cur)
            .map(|_| {
                let (mn, mx) = aabb_empty();
                BvhNode {
                    aabb_min: mn,
                    focus_sum: 0.0,
                    aabb_max: mx,
                    child_start: 0,
                    child_count: 0,
                    is_leaf: 0,
                    leaf_start: 0,
                    leaf_count: 0,
                }
            })
            .collect();

        // For each particle, expand the current-level node's AABB and add focus.
        for p in 0..n {
            let cid = cur_ids[p] as usize;
            if cid < n_cur {
                let pos = coords.position(p);
                let foc = focus.get(p).copied().unwrap_or(0.0);
                let node = &mut cur_nodes[cid];
                aabb_expand(&mut node.aabb_min, &mut node.aabb_max, pos);
                node.focus_sum += foc;
            }
        }

        // Link to children from previous level.
        // For each previous-level cluster id prev_c, find which current-level cluster
        // it belongs to. We pick the first particle from prev_c to determine cur cluster.
        // Build a mapping: prev_cluster_id → cur_cluster_id.
        let prev_count = nodes.len() - prev_offset;
        let mut prev_to_cur = vec![u32::MAX; prev_count];
        for p in 0..n {
            let pc = prev_ids[p] as usize;
            let cc = cur_ids[p] as usize;
            if pc < prev_count && cc < n_cur && prev_to_cur[pc] == u32::MAX {
                prev_to_cur[pc] = cc as u32;
            }
        }

        // For each current-level cluster, find the range of prev-level children.
        // Build cur_cluster → sorted list of prev_cluster children.
        let mut children_map: Vec<Vec<u32>> = vec![Vec::new(); n_cur];
        for (prev_c, &cur_c) in prev_to_cur.iter().enumerate() {
            if cur_c != u32::MAX {
                children_map[cur_c as usize].push((prev_offset + prev_c) as u32);
            }
        }

        // Store children as contiguous ranges: collect into a flat children array.
        // Store child_start as the first child's node index
        // and child_count as number of children (capped at 255).
        for cid in 0..n_cur {
            let children = &children_map[cid];
            if !children.is_empty() {
                cur_nodes[cid].child_start = children[0];
                cur_nodes[cid].child_count = children.len() as u32;
            }
        }

        nodes.extend_from_slice(&cur_nodes);
        prev_offset = cur_offset;
        prev_ids = cur_ids.clone();
    }

    Bvh { nodes, cluster_ids }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epoch::eigensolver::SpectralCoords;

    fn make_coords(pts: &[[f32; 3]]) -> SpectralCoords {
        let n = pts.len();
        let mut coords = Vec::with_capacity(n * 3);
        for p in pts {
            coords.extend_from_slice(p);
        }
        SpectralCoords { n, coords, extra: vec![0.0f32; n * 2] }
    }

    #[test]
    fn bvh_100_particles_all_assigned() {
        // Generate 100 points spread across the unit cube scaled to [-500, 500].
        let n = 100usize;
        let pts: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                let x = (t * 17.3).sin() * 500.0;
                let y = (t * 13.7).cos() * 500.0;
                let z = (t * 11.1).sin() * 300.0;
                [x, y, z]
            })
            .collect();
        let coords = make_coords(&pts);
        let focus = vec![1.0f32 / n as f32; n];

        let bvh = build(&coords, &focus);

        // Every particle should have a cluster ID assigned at all 4 scales.
        assert_eq!(bvh.cluster_ids.len(), n, "cluster_ids length mismatch");
        for (p, ids) in bvh.cluster_ids.iter().enumerate() {
            for level in 0..4 {
                // Cluster ID must be a valid index (not u32::MAX).
                assert_ne!(ids[level], u32::MAX, "particle {p} has no cluster at level {level}");
            }
        }

        // BVH should have at least one node.
        assert!(!bvh.nodes.is_empty(), "BVH has no nodes");

        // Every particle position should be contained in its level-0 cluster AABB.
        let ids0: Vec<u32> = bvh.cluster_ids.iter().map(|ids| ids[0]).collect();
        // Find node offset for level 0 (starts at 0 in our layout).
        // Compute node range at level 0: 0..n_c0.
        let n_c0 = ids0.iter().copied().max().map(|m| m + 1).unwrap_or(0) as usize;
        for p in 0..n {
            let cid = ids0[p] as usize;
            assert!(cid < n_c0, "cluster id {cid} out of range {n_c0}");
            let node = &bvh.nodes[cid];
            let pos = pts[p];
            for d in 0..3 {
                assert!(
                    node.aabb_min[d] <= pos[d] + 1e-3,
                    "particle {p} dim {d}: pos {} < aabb_min {}",
                    pos[d],
                    node.aabb_min[d]
                );
                assert!(
                    node.aabb_max[d] >= pos[d] - 1e-3,
                    "particle {p} dim {d}: pos {} > aabb_max {}",
                    pos[d],
                    node.aabb_max[d]
                );
            }
        }
    }
}
