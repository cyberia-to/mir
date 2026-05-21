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

// ---------------------------------------------------------------------------
// Compute-based edge line rasterizer (§8, aruminium Metal compute).
// ---------------------------------------------------------------------------

/// MSL compute kernel: one thread per edge, rasterises as a thin anti-aliased
/// line into an RGBA f32 pixel buffer.
///
/// Buffer layout:
///   0 — float3  positions_all[n_particles]
///   1 — uint2   edges[n_edges]            (from, to)
///   2 — float   weights[n_edges]          (A^eff_0 — tube thickness)
///   3 — float   flow_uvs[n_edges]         (current UV offset)
///   4 — float4  out_pixels[W*H]           (read-write RGBA)
///   5 — float4x4 view_proj
///   6 — uint2   viewport
pub const EDGE_LINE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Project world point to screen pixel coords.
static float2 project(float3 p, float4x4 vp, float W, float H) {
    float4 clip = vp * float4(p, 1.0f);
    if (clip.w <= 0.0f) return float2(-1e9f, -1e9f);
    float2 ndc = clip.xy / clip.w;
    return float2((ndc.x * 0.5f + 0.5f) * W,
                  (1.0f - (ndc.y * 0.5f + 0.5f)) * H);
}

kernel void edge_line_rasterize(
    device const float3 *positions  [[buffer(0)]],
    device const uint2  *edges      [[buffer(1)]],
    device const float  *weights    [[buffer(2)]],
    device const float  *flow_uvs   [[buffer(3)]],
    device float4       *out_pixels [[buffer(4)]],
    constant float4x4   &view_proj  [[buffer(5)]],
    constant uint2      &viewport   [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint W = viewport.x;
    uint H = viewport.y;

    uint2  e   = edges[gid];
    float  w   = weights[gid];
    float3 p0  = positions[e.x];
    float3 p1  = positions[e.y];

    float2 s0 = project(p0, view_proj, float(W), float(H));
    float2 s1 = project(p1, view_proj, float(W), float(H));

    // Clip to screen.
    if (s0.x < 0 || s0.x >= float(W) || s0.y < 0 || s0.y >= float(H)) return;
    if (s1.x < 0 || s1.x >= float(W) || s1.y < 0 || s1.y >= float(H)) return;

    // Rasterise line: step along the longer axis.
    float2 delta = s1 - s0;
    int steps = int(max(abs(delta.x), abs(delta.y))) + 1;
    float2 step = delta / float(steps);

    float line_w = clamp(w * 3.0f, 0.5f, 4.0f);  // pixel width from edge weight
    float half_w = line_w * 0.5f;

    // Edge color: neon green with slight flow shimmer.
    float uv  = flow_uvs[gid];
    float3 edge_col = mix(float3(0.05f, 0.60f, 0.10f),
                          float3(0.30f, 1.00f, 0.30f), uv);

    for (int i = 0; i <= steps; ++i) {
        float2 px = s0 + step * float(i);
        int ix = int(px.x); int iy = int(px.y);

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = ix + dx; int ny = iy + dy;
                if (nx < 0 || nx >= int(W) || ny < 0 || ny >= int(H)) continue;
                float2 off = float2(nx, ny) - px;
                float dist = length(off);
                if (dist > half_w + 0.5f) continue;

                float alpha = clamp(1.0f - (dist - half_w), 0.0f, 1.0f) * 0.6f;
                uint idx = uint(ny) * W + uint(nx);
                float4 old = out_pixels[idx];
                // Additive blend for glowing edge look.
                out_pixels[idx] = float4(old.rgb + edge_col * alpha * (1.0f - old.a),
                                         min(old.a + alpha, 1.0f));
            }
        }
    }
}
"#;

/// Compute-based edge line rasteriser for use in the frame render loop.
pub struct EdgeLinePass {
    gpu:      aruminium::Gpu,
    pipeline: aruminium::Pipeline,
    queue:    aruminium::Queue,
}

unsafe impl Send for EdgeLinePass {}
unsafe impl Sync for EdgeLinePass {}

impl EdgeLinePass {
    pub fn new() -> Result<Self, aruminium::GpuError> {
        let gpu  = aruminium::Gpu::open()?;
        let lib  = gpu.compile(EDGE_LINE_MSL)?;
        let func = lib.function("edge_line_rasterize")?;
        let pipeline = gpu.pipeline(&func)?;
        let queue = gpu.new_command_queue()?;
        Ok(Self { gpu, pipeline, queue })
    }

    /// Rasterize edges into `pixels` (RGBA f32, W×H×4, modified in place).
    ///
    /// `edge_list` — (from_idx, to_idx) pairs for visible edges.
    /// `pos_buf`   — all-particle positions (n×3 f32).
    /// `weights`   — per-edge A^eff_0 weight.
    /// `flow_uvs`  — per-edge UV offsets from `EdgePass`.
    /// `view_proj` — column-major 4×4 VP matrix.
    #[allow(clippy::too_many_arguments)]
    pub fn rasterize(
        &self,
        pixels:    &mut Vec<f32>,
        edge_list: &[(u32, u32)],
        pos_buf:   &aruminium::Buffer,
        weights:   &[f32],
        flow_uvs:  &[f32],
        view_proj: &[[f32; 4]; 4],
        viewport:  [u32; 2],
    ) -> Result<(), aruminium::GpuError> {
        let n_edges = edge_list.len().min(weights.len()).min(flow_uvs.len());
        if n_edges == 0 { return Ok(()); }

        let [w, h] = viewport;
        let pixel_count = (w * h) as usize;
        if pixels.len() < pixel_count * 4 { return Ok(()); }

        // Pack edges as u32 pairs.
        let edge_data: Vec<u32> = edge_list[..n_edges].iter()
            .flat_map(|&(f, t)| [f, t])
            .collect();

        let pos_bytes = pos_buf.read_f32(|s| s.to_vec());
        let pos_buf_gpu = self.gpu.buffer_with_data(cast_u8_f32(&pos_bytes))?;
        let edge_buf   = self.gpu.buffer_with_data(cast_u8_u32(&edge_data))?;
        let w_buf      = self.gpu.buffer_with_data(cast_u8_f32(&weights[..n_edges]))?;
        let uv_buf     = self.gpu.buffer_with_data(cast_u8_f32(&flow_uvs[..n_edges]))?;
        let px_buf     = self.gpu.buffer_with_data(cast_u8_f32(pixels))?;

        let vp_bytes: [u8; 64] = unsafe { std::mem::transmute(*view_proj) };
        let viewport_bytes: [u8; 8] = unsafe { std::mem::transmute([w, h]) };

        let cmd = self.queue.commands()?;
        let enc = cmd.encoder()?;

        enc.bind(&self.pipeline);
        enc.bind_buffer(&pos_buf_gpu, 0, 0);
        enc.bind_buffer(&edge_buf,   0, 1);
        enc.bind_buffer(&w_buf,      0, 2);
        enc.bind_buffer(&uv_buf,     0, 3);
        enc.bind_buffer(&px_buf,     0, 4);
        enc.push(&vp_bytes,             5);
        enc.push(&viewport_bytes,       6);

        enc.launch((n_edges, 1, 1), (64, 1, 1));
        enc.finish();
        cmd.submit();
        cmd.wait();

        let result = px_buf.read_f32(|s| s.to_vec());
        let copy_len = pixels.len().min(result.len());
        pixels[..copy_len].copy_from_slice(&result[..copy_len]);
        Ok(())
    }
}

fn cast_u8_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn cast_u8_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

// ── §8.3 Edge bundling ────────────────────────────────────────────────────────

/// A bundled edge: represents one or more raw edges merged by cluster proximity.
#[derive(Clone)]
pub struct BundledEdge {
    pub from:   u32,
    pub to:     u32,
    pub weight: f32,  // sum of raw edge weights in the bundle
    pub count:  u32,  // number of edges merged into this bundle
}

/// §8.3: bundle visible edges by shared BVH cluster membership at level 2.
///
/// Edges sharing the same (level-2 cluster of `from`, level-2 cluster of `to`)
/// pair are merged into a single bundled edge with summed weight.  Reduces
/// edge instance count by 10–100× at typical camera distances.
///
/// `edge_list`   — (from_idx, to_idx) pairs (visible edges only).
/// `weights`     — per-edge A^eff_0 weight.
/// `cluster_ids` — per-particle cluster IDs at 4 τ scales (from EpochState).
pub fn bundle_edges(
    edge_list:   &[(u32, u32)],
    weights:     &[f32],
    cluster_ids: &[[u32; 4]],
) -> Vec<BundledEdge> {
    use std::collections::HashMap;
    let mut bundles: HashMap<(u32, u32), BundledEdge> = HashMap::new();

    for (i, &(from, to)) in edge_list.iter().enumerate() {
        let w = weights.get(i).copied().unwrap_or(0.0);
        let cf = cluster_ids.get(from as usize).map(|ids| ids[2]).unwrap_or(0);
        let ct = cluster_ids.get(to   as usize).map(|ids| ids[2]).unwrap_or(0);
        let key = if cf <= ct { (cf, ct) } else { (ct, cf) };

        bundles.entry(key)
            .and_modify(|b| { b.weight += w; b.count += 1; })
            .or_insert(BundledEdge { from, to, weight: w, count: 1 });
    }

    bundles.into_values().collect()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_uv_wraps() {
        let mut pass = EdgePass::new(3);
        // weight = 0.5 → offset += 0.5 each frame
        let weights = [0.5f32, -0.5, 1.0];
        pass.update_flow_uvs(&weights, 1.0);
        assert!((pass.flow_offsets[0] - 0.5).abs() < 1e-6, "forward advance");
        assert!((pass.flow_offsets[1] - 0.5).abs() < 1e-6, "backward wraps to 0.5");
        assert!((pass.flow_offsets[2] - 0.0).abs() < 1e-6, "1.0 wraps to 0.0");
    }

    #[test]
    fn resize_preserves_existing() {
        let mut pass = EdgePass::new(2);
        pass.flow_offsets[1] = 0.3;
        pass.resize(4);
        assert_eq!(pass.flow_offsets.len(), 4);
        assert!((pass.flow_offsets[1] - 0.3).abs() < 1e-6);
        assert_eq!(pass.flow_offsets[2], 0.0);
    }

    #[test]
    fn bundle_collapses_parallel_edges() {
        // Two edges share the same cluster pair at level 2 → should bundle.
        let edges   = vec![(0u32, 1u32), (2u32, 3u32)];
        let weights = vec![0.3f32, 0.5];
        // Both pairs belong to clusters [0, 0, 0, 0] and [0, 0, 1, 0].
        let cluster_ids: Vec<[u32; 4]> = vec![
            [0, 0, 0, 0],  // particle 0 → level-2 cluster 0
            [1, 1, 1, 0],  // particle 1 → level-2 cluster 1
            [2, 2, 0, 0],  // particle 2 → level-2 cluster 0
            [3, 3, 1, 0],  // particle 3 → level-2 cluster 1
        ];
        let bundled = bundle_edges(&edges, &weights, &cluster_ids);
        // Both edges map to cluster pair (0, 1) → should collapse into 1 bundle.
        assert_eq!(bundled.len(), 1, "expected 1 bundle, got {}", bundled.len());
        assert!((bundled[0].weight - 0.8).abs() < 1e-5, "bundled weight={}", bundled[0].weight);
        assert_eq!(bundled[0].count, 2);
    }

    #[test]
    fn bundle_distinct_clusters_stay_separate() {
        let edges   = vec![(0u32, 1u32), (2u32, 3u32)];
        let weights = vec![1.0f32, 1.0];
        let cluster_ids: Vec<[u32; 4]> = vec![
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [2, 2, 2, 0],  // level-2 cluster 2 (different from 0)
            [3, 3, 3, 0],  // level-2 cluster 3
        ];
        let bundled = bundle_edges(&edges, &weights, &cluster_ids);
        assert_eq!(bundled.len(), 2, "distinct cluster pairs should not bundle");
    }
}
