//! Edge rendering: bundled tubes + flow UV animation. Step 9.
//!
//! For each visible edge (p, q) where both endpoints are in the VisibleSet at
//! T1 or T2, a tube is generated with:
//!   • Radius proportional to A^eff_0(p, q) (effective affinity at scale τ=0)
//!   • UV along tube length driving the flow-strip texture offset
//!
//! The CPU side tracks per-edge flow offsets (updated each frame).
//! The MSL compute kernel writes instanced indirect-draw arguments for the GPU.

/// Per-frame UV offset update for flow-strip texture.
/// `flow_offsets[e]` advances by `sign * |P_pq| * dt` each frame, wrapping in [0, 1).
///
/// In a full integration:
///   sign     = sign(A^eff_2(p, q))  (direction of information flow)
///   |P_pq|   = |φ*(p) - φ*(q)|      (focus delta drives flow speed)
///   dt       = frame delta-time in seconds
pub struct EdgePass {
    /// Per-edge UV offset ∈ [0, 1).  Index matches the edge list passed to the GPU.
    flow_offsets: Vec<f32>,
}

impl EdgePass {
    /// Create a new EdgePass for a graph with `n_edges` visible edges.
    pub fn new(n_edges: usize) -> Self {
        Self {
            flow_offsets: vec![0.0f32; n_edges],
        }
    }

    /// Returns a read-only view of the current flow UV offsets.
    pub fn flow_offsets(&self) -> &[f32] {
        &self.flow_offsets
    }

    /// Resize the offset buffer when the visible edge set changes.
    pub fn resize(&mut self, n_edges: usize) {
        self.flow_offsets.resize(n_edges, 0.0);
    }

    /// Advance per-edge flow UV offsets by one frame.
    ///
    /// `edge_weights[e]` = signed flow speed P_pq (positive → p→q direction).
    /// The offset wraps in [0, 1) so the flow texture tiles seamlessly.
    pub fn update_flow_uvs(&mut self, edge_weights: &[f32], dt: f32) {
        debug_assert_eq!(
            self.flow_offsets.len(),
            edge_weights.len(),
            "EdgePass: offset buffer length != edge_weights length"
        );
        for (offset, &w) in self.flow_offsets.iter_mut().zip(edge_weights.iter()) {
            let delta = w * dt; // signed: w = sign(A^eff_2) * |P_pq|
            *offset = (*offset + delta).rem_euclid(1.0);
        }
    }
}

// ---------------------------------------------------------------------------
// MSL compute shader: generate instanced tube geometry for visible edges.
// ---------------------------------------------------------------------------

/// MSL compute kernel that writes instanced indirect-draw arguments for the
/// tube geometry pass.  One thread per visible edge.
///
/// Buffer layout:
///   0 — f32x3  positions[n_particles]  (spectral coordinates)
///   1 — uint2  edges[n_edges]          (from_idx, to_idx)
///   2 — f32    weights[n_edges]        (A^eff_0 edge weight → tube radius)
///   3 — f32    flow_uvs[n_edges]       (per-edge UV offset from EdgePass)
///   4 — uint   draw_args[]             (MTLDrawIndexedPrimitivesIndirectArguments)
///   5 — uint   n_edges (constant)
pub const EDGE_TUBES_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct DrawArgs {
    uint index_count;
    uint instance_count;
    uint index_start;
    int  base_vertex;
    uint base_instance;
};

kernel void edge_tubes(
    device const float3 *positions   [[buffer(0)]],
    device const uint2  *edges       [[buffer(1)]],
    device const float  *weights     [[buffer(2)]],
    device const float  *flow_uvs    [[buffer(3)]],
    device       DrawArgs *draw_args [[buffer(4)]],
    constant uint        &n_edges    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_edges) return;

    uint2  e       = edges[gid];
    float3 p       = positions[e.x];
    float3 q       = positions[e.y];
    float  weight  = weights[gid];
    float  uv_off  = flow_uvs[gid];

    // Tube radius proportional to edge weight (clamped to a visible range).
    float radius = clamp(weight * 0.05, 0.002, 0.05);

    // Length of the tube.
    float3 dir = q - p;
    float  len = length(dir);
    if (len < 1e-6) return;  // degenerate edge

    // Write one indirect-draw call per edge (tube = cylinder, 36 indices).
    // base_instance carries the edge index so the vertex shader can look up
    // p, q, radius, and uv_off per instance.
    draw_args[gid].index_count    = 36u;   // 12 tris × 3 verts (cylinder approx)
    draw_args[gid].instance_count = 1u;
    draw_args[gid].index_start    = 0u;
    draw_args[gid].base_vertex    = 0;
    draw_args[gid].base_instance  = gid;

    // Suppress unused-variable warnings in stub shader.
    (void)radius; (void)len; (void)uv_off;
}
"#;

/// MSL vertex shader for tube rendering — reads per-instance edge data.
pub const EDGE_TUBE_VERT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float3 normal;
};

// Per-instance edge payload written by the edge_tubes compute kernel.
struct EdgeInstance {
    float3 p;          // start position
    float3 q;          // end   position
    float  radius;
    float  uv_offset;
};

vertex VertexOut tube_vert(
    uint                        vid          [[vertex_id]],
    uint                        iid          [[instance_id]],
    device const EdgeInstance  *instances    [[buffer(0)]],
    constant     float4x4      &view_proj    [[buffer(1)]])
{
    EdgeInstance inst = instances[iid];

    // Build a local frame around the tube axis.
    float3 axis = normalize(inst.q - inst.p);
    float3 up   = abs(axis.y) < 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 side = normalize(cross(axis, up));
    up           = cross(side, axis);

    // 12-segment cylinder: vid ∈ [0, 35].
    uint   seg   = vid % 12;
    uint   end   = vid / 12;         // 0 = start cap, 1 = end cap, 2 = body
    float  theta = (float(seg) / 12.0) * 6.2831853;
    float3 radial = (cos(theta) * side + sin(theta) * up) * inst.radius;
    float3 base   = (end == 0u) ? inst.p : inst.q;
    float3 pos    = base + radial;

    float2 uv = float2(theta / 6.2831853, float(end) + inst.uv_offset);

    VertexOut out;
    out.position = view_proj * float4(pos, 1.0);
    out.uv       = uv;
    out.normal   = radial / inst.radius;
    return out;
}
"#;
