//! The paint pass: one kernel per frame.
//!
//! Per pixel — edge glow first (screen-space segments precomputed on the
//! CPU), Gaussian splats composited over (back-to-front, §6.4), T2 sphere
//! impostors ray-cast on top (§6.3), packed straight to RGBA8. One dispatch,
//! no intermediate frame buffer, no barriers between tiers — the whole
//! former t3 → t2 → edges → pack chain in registers.

use super::cull::{Camera, TierLevel};

pub const PAINT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Camera {
    float4x4 view_proj;
    float4   planes[6];
    float2   viewport;
    float    near;
    float    far;
    // The camera's own basis, in world space. A ray cannot be rebuilt from
    // view_proj's columns: those are the basis already multiplied through the
    // projection, so they carry the aspect and depth scales and are not unit
    // vectors in any frame. Reading them as a basis puts every ray-cast
    // sphere somewhere the rest of the scene is not.
    float4   cam_pos;
    float4   cam_right;
    float4   cam_up;
    float4   cam_fwd;
};

// The single light both tiers are lit by. Written normalized rather than
// normalize()d so the constant is identical in both languages.
constant float3 LIGHT = float3(0.36370f, 0.72739f, 0.58191f);

static float seg_dist(float2 p, float2 a, float2 b) {
    float2 ab = b - a;
    float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6f), 0.0f, 1.0f);
    return length(p - a - ab * t);
}

static float intersect_sphere(float3 O, float3 D, float3 C, float r) {
    float3 oc = O - C;
    float  a  = dot(D, D);
    float  hb = dot(D, oc);
    float  cc = dot(oc, oc) - r * r;
    float  disc = hb * hb - a * cc;
    if (disc < 0.0f) return -1.0f;
    float sq = sqrt(disc);
    float t0 = (-hb - sq) / a;
    if (t0 > 0.0f) return t0;
    float t1 = (-hb + sq) / a;
    return (t1 > 0.0f) ? t1 : -1.0f;
}

kernel void paint(
    device const float4 *splats    [[buffer(0)]],  // 2/splat: xyz r | rgb -, back-to-front
    device const float4 *spheres   [[buffer(1)]],  // 2/sphere: same layout, T2 subset
    device const float4 *segments  [[buffer(2)]],  // 2/edge: x0 y0 x1 y1 | hw r g b
    device uint         *dstbuf    [[buffer(3)]],  // packed RGBA8
    constant Camera     &camera    [[buffer(4)]],
    constant uint4      &counts    [[buffer(5)]],  // n_splats n_t2 n_edges -
    constant uint2      &viewport  [[buffer(6)]],
    uint2               gid        [[thread_position_in_grid]])
{
    uint W = viewport.x;
    uint H = viewport.y;
    if (gid.x >= W || gid.y >= H) return;
    float2 pix_f = float2(float(gid.x) + 0.5f, float(gid.y) + 0.5f);

    // ── edges: additive glow on black ──
    float3 rgb = float3(0.0f);
    float  a   = 0.0f;
    for (uint e = 0; e < counts.z; ++e) {
        float4 s0 = segments[e*2];
        float4 s1 = segments[e*2+1];
        float d = seg_dist(pix_f, s0.xy, s0.zw);
        // Analytic coverage across exactly one pixel: the edge of the line
        // lands where the line ends, instead of a pixel past it.
        float alpha = (1.0f - smoothstep(s1.x - 0.5f, s1.x + 0.5f, d)) * 0.85f;
        if (alpha <= 0.0f) continue;
        rgb += s1.yzw * alpha * (1.0f - a);
        a = min(a + alpha, 1.0f);
    }

    // ── gaussian splats over the glow (front-to-back accumulate) ──
    float3 g_rgb = float3(0.0f);
    float  g_a   = 0.0f;
    for (uint i = 0; i < counts.x; ++i) {
        if (g_a >= 0.9999f) break;
        float4 pr = splats[i*2];
        float4 clip = camera.view_proj * float4(pr.xyz, 1.0f);
        if (clip.w <= 0.0f) continue;
        float2 ndc = clip.xy / clip.w;
        float2 screen = float2((ndc.x * 0.5f + 0.5f) * float(W),
                               (1.0f - (ndc.y * 0.5f + 0.5f)) * float(H));
        float proj_r = max(pr.w * camera.cam_up.w
                           / clip.w * float(H) * 0.5f, 0.5f);
        float4 col = splats[i*2+1];
        // Solid particles belong to the sphere pass. Drawing them here as well
        // only puts a second, softer copy under the first.
        if (col.w > 0.5f) continue;
        float2 delta = pix_f - screen;
        float d = length(delta);
        // The silhouette the sphere pass would have given this particle, with
        // one pixel of coverage at the edge. A gaussian was about half this
        // wide, so a particle crossing the tier threshold changed size in one
        // frame — that step is what flickers while zooming.
        float alpha = clamp(proj_r + 0.5f - d, 0.0f, 1.0f);
        if (alpha <= 0.0f) continue;
        // Lit as the ball it stands for: the offset within the disc is the
        // surface normal, and the shading is the sphere pass's, term for term.
        float2 nd = delta / max(proj_r, 1e-4f);
        float nz = sqrt(max(0.0f, 1.0f - min(dot(nd, nd), 1.0f)));
        float3 nrm = normalize(float3(nd.x, -nd.y, nz));
        // Including the rim. The sphere pass has one, and a particle crossing
        // the tier threshold must not change brightness as it changes path —
        // during an orbit particles cross it constantly, and a step in
        // brightness on every crossing is what reads as flicker. nz is the
        // normal's component toward the camera, which is what the sphere pass
        // computes as dot(n, -ray).
        float3 lit = col.xyz * (0.2f + 0.8f * max(0.0f, dot(nrm, LIGHT)))
                   + pow(1.0f - nz, 3.0f) * 0.4f;
        g_rgb += lit * alpha * (1.0f - g_a);
        g_a   += alpha * (1.0f - g_a);
    }
    rgb = g_rgb + rgb * (1.0f - g_a);

    // ── T2 sphere impostors on top ──
    if (counts.y > 0) {
        float3 cam_origin = camera.cam_pos.xyz;
        float2 ndc;
        ndc.x =  (pix_f.x) / float(W) * 2.0f - 1.0f;
        ndc.y = -(pix_f.y) / float(H) * 2.0f + 1.0f;
        // The focal scales come from the camera, not from view_proj: those
        // entries carry right.x and up.y with them and are only the focal
        // length while the camera is unrotated. Taking them from the matrix
        // stretches x and y by different amounts the moment the graph is
        // turned, which is a sphere drawn as an ellipse.
        float3 ray_world = normalize(camera.cam_right.xyz * (ndc.x / camera.cam_right.w)
                                   + camera.cam_up.xyz    * (ndc.y / camera.cam_up.w)
                                   + camera.cam_fwd.xyz);

        float  t_min = 1e9f;
        float3 hit_n = float3(0.0f, 0.0f, 1.0f);
        float3 hit_col = float3(0.0f);
        bool   hit_any = false;
        for (uint i = 0; i < counts.y; ++i) {
            float4 pr = spheres[i*2];
            float t = intersect_sphere(cam_origin, ray_world, pr.xyz, pr.w);
            if (t > 0.0f && t < t_min) {
                t_min = t;
                hit_any = true;
                hit_n = normalize(cam_origin + t * ray_world - pr.xyz);
                hit_col = spheres[i*2+1].xyz;
            }
        }
        if (hit_any) {
            float  diff  = max(0.0f, dot(hit_n, LIGHT));
            float  rim   = pow(1.0f - max(0.0f, dot(hit_n, -ray_world)), 3.0f) * 0.4f;
            rgb = hit_col * (0.2f + 0.8f * diff) + rim;
        }
    }

    rgb = clamp(rgb, 0.0f, 1.0f);
    dstbuf[gid.y * W + gid.x] = uint(rgb.x * 255.0f)
                              | (uint(rgb.y * 255.0f) << 8)
                              | (uint(rgb.z * 255.0f) << 16)
                              | (255u << 24);
}
"#;

pub const PAINT_WGSL: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    planes: array<vec4<f32>, 6>,
    viewport: vec2<f32>,
    near: f32,
    far: f32,
    // The camera's own basis, in world space. A ray cannot be rebuilt from
    // view_proj's columns: those are the basis already multiplied through the
    // projection, so they carry the aspect and depth scales and are not unit
    // vectors in any frame. Reading them as a basis puts every ray-cast
    // sphere somewhere the rest of the scene is not.
    cam_pos: vec4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    cam_fwd: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> splats: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> spheres: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> segments: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dstbuf: array<u32>;
@group(0) @binding(4) var<uniform> camera: Camera;
@group(0) @binding(5) var<uniform> counts: vec4<u32>;
@group(0) @binding(6) var<uniform> viewport: vec2<u32>;

// The single light both tiers are lit by. Written normalized rather than
// normalize()d so the constant is identical in both languages.
const LIGHT = vec3<f32>(0.36370, 0.72739, 0.58191);

fn seg_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    return length(p - a - ab * t);
}

fn intersect_sphere(O: vec3<f32>, D: vec3<f32>, C: vec3<f32>, r: f32) -> f32 {
    let oc = O - C;
    let a = dot(D, D);
    let hb = dot(D, oc);
    let cc = dot(oc, oc) - r * r;
    let disc = hb * hb - a * cc;
    if (disc < 0.0) { return -1.0; }
    let sq = sqrt(disc);
    let t0 = (-hb - sq) / a;
    if (t0 > 0.0) { return t0; }
    let t1 = (-hb + sq) / a;
    if (t1 > 0.0) { return t1; }
    return -1.0;
}

@compute @workgroup_size(16, 16)
fn paint(@builtin(global_invocation_id) gid: vec3<u32>) {
    let W = viewport.x;
    let H = viewport.y;
    if (gid.x >= W || gid.y >= H) { return; }
    let pix_f = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

    // edges: additive glow on black
    var rgb = vec3<f32>(0.0);
    var a = 0.0;
    for (var e = 0u; e < counts.z; e++) {
        let s0 = segments[e*2u];
        let s1 = segments[e*2u+1u];
        let d = seg_dist(pix_f, s0.xy, s0.zw);
        // Analytic coverage across exactly one pixel: the edge of the line
        // lands where the line ends, instead of a pixel past it.
        let alpha = (1.0 - smoothstep(s1.x - 0.5, s1.x + 0.5, d)) * 0.85;
        if (alpha <= 0.0) { continue; }
        rgb += s1.yzw * alpha * (1.0 - a);
        a = min(a + alpha, 1.0);
    }

    // gaussian splats over the glow
    var g_rgb = vec3<f32>(0.0);
    var g_a = 0.0;
    for (var i = 0u; i < counts.x; i++) {
        if (g_a >= 0.9999) { break; }
        let pr = splats[i*2u];
        let clip = camera.view_proj * vec4<f32>(pr.xyz, 1.0);
        if (clip.w <= 0.0) { continue; }
        let ndc = clip.xy / clip.w;
        let screen = vec2<f32>((ndc.x * 0.5 + 0.5) * f32(W),
                               (1.0 - (ndc.y * 0.5 + 0.5)) * f32(H));
        let proj_r = max(pr.w * camera.cam_up.w
                         / clip.w * f32(H) * 0.5, 0.5);
        let col = splats[i*2u+1u];
        // Solid particles belong to the sphere pass. Drawing them here as well
        // only puts a second, softer copy under the first.
        if (col.w > 0.5) { continue; }
        let delta = pix_f - screen;
        let d = length(delta);
        // The silhouette the sphere pass would have given this particle, with
        // one pixel of coverage at the edge. A gaussian was about half this
        // wide, so a particle crossing the tier threshold changed size in one
        // frame — that step is what flickers while zooming.
        let alpha = clamp(proj_r + 0.5 - d, 0.0, 1.0);
        if (alpha <= 0.0) { continue; }
        // Lit as the ball it stands for: the offset within the disc is the
        // surface normal, and the shading is the sphere pass's, term for term.
        let nd = delta / max(proj_r, 1e-4);
        let nz = sqrt(max(0.0, 1.0 - min(dot(nd, nd), 1.0)));
        let nrm = normalize(vec3<f32>(nd.x, -nd.y, nz));
        // Including the rim. The sphere pass has one, and a particle crossing
        // the tier threshold must not change brightness as it changes path —
        // during an orbit particles cross it constantly, and a step in
        // brightness on every crossing is what reads as flicker. nz is the
        // normal's component toward the camera, which is what the sphere pass
        // computes as dot(n, -ray).
        let lit = col.xyz * (0.2 + 0.8 * max(0.0, dot(nrm, LIGHT)))
                + pow(1.0 - nz, 3.0) * 0.4;
        g_rgb += lit * alpha * (1.0 - g_a);
        g_a   += alpha * (1.0 - g_a);
    }
    rgb = g_rgb + rgb * (1.0 - g_a);

    // T2 sphere impostors on top
    if (counts.y > 0u) {
        let cam_origin = camera.cam_pos.xyz;
        let ndc2 = vec2<f32>(pix_f.x / f32(W) * 2.0 - 1.0,
                             -(pix_f.y / f32(H) * 2.0 - 1.0));
        // The focal scales come from the camera, not from view_proj: those
        // entries carry right.x and up.y with them and are only the focal
        // length while the camera is unrotated. Taking them from the matrix
        // stretches x and y by different amounts the moment the graph is
        // turned, which is a sphere drawn as an ellipse.
        let ray_world = normalize(camera.cam_right.xyz * (ndc2.x / camera.cam_right.w)
                                + camera.cam_up.xyz    * (ndc2.y / camera.cam_up.w)
                                + camera.cam_fwd.xyz);

        var t_min = 1e9;
        var hit_n = vec3<f32>(0.0, 0.0, 1.0);
        var hit_col = vec3<f32>(0.0);
        var hit_any = false;
        for (var i = 0u; i < counts.y; i++) {
            let pr = spheres[i*2u];
            let t = intersect_sphere(cam_origin, ray_world, pr.xyz, pr.w);
            if (t > 0.0 && t < t_min) {
                t_min = t;
                hit_any = true;
                hit_n = normalize(cam_origin + t * ray_world - pr.xyz);
                hit_col = spheres[i*2u+1u].xyz;
            }
        }
        if (hit_any) {
            let diff = max(0.0, dot(hit_n, LIGHT));
            let rim = pow(1.0 - max(0.0, dot(hit_n, -ray_world)), 3.0) * 0.4;
            rgb = hit_col * (0.2 + 0.8 * diff) + rim;
        }
    }

    rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    dstbuf[gid.y * W + gid.x] = u32(rgb.x * 255.0)
                              | (u32(rgb.y * 255.0) << 8u)
                              | (u32(rgb.z * 255.0) << 16u)
                              | (255u << 24u);
}
"#;

#[cfg(target_vendor = "apple")]
pub const PAINT_SRC: &str = PAINT_MSL;
#[cfg(not(target_vendor = "apple"))]
pub const PAINT_SRC: &str = PAINT_WGSL;

/// Sort particle indices back-to-front relative to the camera.
pub fn sort_by_depth(
    entries:   &[(u32, TierLevel)],
    positions: &[f32],
    camera:    &Camera,
) -> Vec<u32> {
    let mut all: Vec<u32> = entries.iter().map(|(idx, _)| *idx).collect();
    let fwd = [camera.view_proj[0][2], camera.view_proj[1][2], camera.view_proj[2][2]];
    let w_col = [camera.view_proj[0][3], camera.view_proj[1][3],
                 camera.view_proj[2][3], camera.view_proj[3][3]];
    let depth_of = |idx: u32| -> f32 {
        let base = idx as usize * 3;
        let (x, y, z) = (positions[base], positions[base + 1], positions[base + 2]);
        let clip_z = fwd[0] * x + fwd[1] * y + fwd[2] * z + camera.view_proj[3][2];
        let clip_w = w_col[0] * x + w_col[1] * y + w_col[2] * z + w_col[3];
        if clip_w.abs() < 1e-9 { 0.0 } else { clip_z / clip_w }
    };
    // Ties break on index, so two particles at the same depth keep the same
    // order every frame. Left to chance they swap, and swapping the order of
    // two alpha-composited splats changes the pixels.
    all.sort_unstable_by(|&a, &b| {
        depth_of(b)
            .partial_cmp(&depth_of(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    all
}

/// Project the cached edge list to screen-space glow segments:
/// x0 y0 x1 y1 half_width r g b — eight floats per edge.
pub fn edge_segments(
    edge_list: &[(u32, u32)],
    weights:   &[f32],
    positions: &[f32],
    camera:    &Camera,
    viewport:  [u32; 2],
) -> Vec<f32> {
    let (w, h) = (viewport[0] as f32, viewport[1] as f32);
    let vp = &camera.view_proj;
    let project = |i: u32| -> Option<[f32; 2]> {
        let b = i as usize * 3;
        let (x, y, z) = (positions[b], positions[b + 1], positions[b + 2]);
        let cw = vp[0][3] * x + vp[1][3] * y + vp[2][3] * z + vp[3][3];
        if cw <= 0.0 { return None; }
        let cx = (vp[0][0] * x + vp[1][0] * y + vp[2][0] * z + vp[3][0]) / cw;
        let cy = (vp[0][1] * x + vp[1][1] * y + vp[2][1] * z + vp[3][1]) / cw;
        let sx = (cx * 0.5 + 0.5) * w;
        let sy = (1.0 - (cy * 0.5 + 0.5)) * h;
        ((0.0..w).contains(&sx) && (0.0..h).contains(&sy)).then_some([sx, sy])
    };
    let mut segs = Vec::with_capacity(edge_list.len() * 8);
    for (k, &(p, q)) in edge_list.iter().enumerate() {
        let (Some(s0), Some(s1)) = (project(p), project(q)) else { continue };
        // Physical pixels now, and a link is a line rather than a bar: the
        // shader feathers it over one pixel, so sub-pixel widths still read.
        let half_w = (weights.get(k).copied().unwrap_or(0.0) * 0.8).clamp(0.25, 0.6);
        segs.extend_from_slice(&[s0[0], s0[1], s1[0], s1[1], half_w, 0.05, 0.60, 0.10]);
    }
    segs
}

pub struct PaintPass {
    gpu:      crate::gpu::Gpu,
    pipeline: crate::gpu::Pipeline,
}

unsafe impl Send for PaintPass {}
unsafe impl Sync for PaintPass {}

impl PaintPass {
    pub fn new() -> Result<Self, crate::gpu::GpuError> {
        let gpu      = crate::gpu::Gpu::open()?;
        let lib      = gpu.compile(PAINT_SRC)?;
        let func     = lib.function("paint")?;
        let pipeline = gpu.pipeline(&func)?;
        Ok(Self { gpu, pipeline })
    }

    /// Paint the whole frame. `sorted` is back-to-front; the T2 subset and
    /// segments are compact CPU-side gathers; `dst` receives packed RGBA8.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        sorted:    &[u32],
        visible:   &[(u32, TierLevel)],
        positions: &[f32],
        radii:     &[f32],
        colors:    &[f32],
        segments:  &[f32],
        camera:    &Camera,
        viewport:  [u32; 2],
        dst:       &crate::gpu::Buffer,
        cmd:       &crate::gpu::Commands,
    ) -> Result<(), crate::gpu::GpuError> {
        let [w, h] = viewport;
        let n = sorted.len() as u32;
        if n == 0 { return Ok(()); }

        // Interleave to two vec4s per particle: [x y z r] [cr cg cb 0].
        let interleave = |idxs: &[u32]| -> Vec<f32> {
            let mut d = Vec::with_capacity(idxs.len() * 8);
            for &idx in idxs {
                let b = idx as usize * 3;
                d.extend_from_slice(&positions[b..b + 3]);
                d.push(radii[idx as usize]);
                d.extend_from_slice(&colors[b..b + 3]);
                d.push(0.0);
            }
            d
        };
        // T2 is the *smallest* tier that earns a sphere, not the only one:
        // T0 and T1 are the nearest, largest particles on screen, and filtering
        // for equality dropped exactly them onto the flat-splat path. The
        // biggest thing in the frame was the one guaranteed to look flat.
        let is_solid = |t: &TierLevel| (*t as u8) <= (TierLevel::T2 as u8);
        let solid_idx: Vec<u32> = visible.iter()
            .filter(|(_, t)| is_solid(t))
            .map(|(i, _)| *i)
            .collect();
        // A splat under a sphere is a halo, not a surface: mark it so the
        // paint pass keeps it as glow instead of a competing flat disc.
        let solid_set: std::collections::HashSet<u32> = solid_idx.iter().copied().collect();
        let mut splat_data = interleave(sorted);
        for (k, &idx) in sorted.iter().enumerate() {
            if solid_set.contains(&idx) { splat_data[k * 8 + 7] = 1.0; }
        }
        let sphere_data = interleave(&solid_idx);
        let m = solid_idx.len() as u32;
        let n_edges = (segments.len() / 8) as u32;

        let b = |v: &[f32]| self.gpu.buffer_with_data(cast_f32(v));
        let splat_buf  = b(&splat_data)?;
        let sphere_buf = b(if sphere_data.is_empty() { &[0.0; 8] } else { &sphere_data })?;
        let seg_buf    = b(if segments.is_empty() { &[0.0; 8] } else { segments })?;

        let camera_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                camera as *const Camera as *const u8,
                std::mem::size_of::<Camera>(),
            )
        };
        let counts: [u32; 4] = [n, m, n_edges, 0];
        let counts_bytes: [u8; 16] = unsafe { std::mem::transmute(counts) };
        let vp_bytes: [u8; 8] = unsafe { std::mem::transmute([w, h]) };

        let enc = cmd.encoder()?;
        enc.bind(&self.pipeline);
        enc.bind_buffer(&splat_buf,  0, 0);
        enc.bind_buffer(&sphere_buf, 0, 1);
        enc.bind_buffer(&seg_buf,    0, 2);
        enc.bind_buffer(dst,         0, 3);
        enc.push(camera_bytes,          4);
        enc.push(&counts_bytes,         5);
        enc.push(&vp_bytes,             6);
        enc.launch((w as usize, h as usize, 1), (16, 16, 1));
        enc.finish();
        Ok(())
    }
}

fn cast_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
