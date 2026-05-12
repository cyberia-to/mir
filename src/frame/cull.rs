//! GPU BVH frustum cull: aruminium compute → VisibleSet + TierLevel.
//! R-1.0 §10.3. Step 4.

/// Tier thresholds in screen-pixel diameter (R-1.0 §6).
pub const S_T0: f32 = 200.0;
pub const S_T1: f32 = 40.0;
pub const S_T2: f32 = 8.0;
pub const S_T3: f32 = 1.0;

pub struct VisibleSet {
    /// Particle indices visible this frame, with assigned tier.
    pub entries: Vec<(u32, TierLevel)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum TierLevel { T0 = 0, T1 = 1, T2 = 2, T3 = 3, TInf = 4 }

/// Placeholder — aruminium BVH traversal compute shader follows in step 4.
pub fn cull_frame(_positions: &[f32], _focus: &[f32]) -> VisibleSet {
    todo!("GPU BVH traversal — step 4")
}

// MSL shader source (compiled at runtime via aruminium::Gpu::compile).
// Traverses the BVH and emits indirect-draw argument buffers per tier.
pub const BVH_CULL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BvhNode {
    float3 aabb_min;
    float  focus_sum;
    float3 aabb_max;
    uint   child_start;
    uint   child_count;
    uint   _pad[3];
};

struct Frustum {
    float4 planes[6]; // left, right, top, bottom, near, far
};

kernel void bvh_cull(
    device const float3  *positions    [[buffer(0)]],
    device const float   *focus        [[buffer(1)]],
    device const BvhNode *bvh          [[buffer(2)]],
    device const Frustum &frustum      [[buffer(3)]],
    device atomic_uint   *visible_count[[buffer(4)]],
    device uint2         *visible_out  [[buffer(5)]],  // (particle_idx, tier)
    constant uint        &n_particles  [[buffer(6)]],
    uint                  gid          [[thread_position_in_grid]])
{
    if (gid >= n_particles) return;

    float3 pos   = positions[gid];
    float  focus_ = focus[gid];

    // TODO step 4: BVH traversal + frustum cull + screen-size tier dispatch.
    // Placeholder: emit all particles at TInf tier.
    uint slot = atomic_fetch_add_explicit(visible_count, 1, memory_order_relaxed);
    visible_out[slot] = uint2(gid, 4); // TInf
}
"#;
