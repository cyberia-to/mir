//! T3 Gaussian splat (§6.4): isotropic 3D Gaussian per Kerbl et al. 2023.
//! CPU depth sort (viable ≤10K particles), GPU compute rasterises into pixel buffer.

use super::super::cull::{Camera, TierLevel};

/// MSL compute shader: rasterise sorted isotropic Gaussian splats.
///
/// Each thread owns one output pixel.  For each splat (front-to-back after
/// CPU sort), project to screen, compute Gaussian weight, alpha-blend.
pub const SPLAT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Camera {
    float4x4 view_proj;
    float4   planes[6];
    float2   viewport;
    float    near;
    float    far;
};

kernel void gaussian_splat(
    device const float  *sorted_positions [[buffer(0)]],  // n*3 f32 xyz
    device const float  *sorted_radii     [[buffer(1)]],  // n   f32
    device const float  *sorted_colors    [[buffer(2)]],  // n*3 f32 rgb
    constant Camera     &camera           [[buffer(3)]],
    constant uint       &n_splats         [[buffer(4)]],
    constant uint2      &viewport         [[buffer(5)]],
    device float4       *out_pixels      [[buffer(6)]],  // RGBA f32, row-major
    uint2               gid               [[thread_position_in_grid]])
{
    uint W = viewport.x;
    uint H = viewport.y;
    if (gid.x >= W || gid.y >= H) return;

    // Accumulator for front-to-back alpha blending.
    float3 color_acc = float3(0.0f);
    float  alpha_acc = 0.0f;

    float2 pix_f = float2(float(gid.x) + 0.5f, float(gid.y) + 0.5f);

    for (uint i = 0; i < n_splats; ++i) {
        if (alpha_acc >= 0.9999f) break;  // early exit: pixel fully opaque

        float3 center  = float3(sorted_positions[i*3],
                                sorted_positions[i*3+1],
                                sorted_positions[i*3+2]);
        float  r       = sorted_radii[i];
        float3 col     = float3(sorted_colors[i*3],
                                sorted_colors[i*3+1],
                                sorted_colors[i*3+2]);
        float  opacity = 1.0f;

        // Project center to clip space.
        float4 clip = camera.view_proj * float4(center, 1.0f);
        if (clip.w <= 0.0f) continue;

        float2 ndc;
        ndc.x = clip.x / clip.w;
        ndc.y = clip.y / clip.w;

        // NDC → screen pixels.
        float2 screen;
        screen.x = (ndc.x * 0.5f + 0.5f) * float(W);
        screen.y = (1.0f - (ndc.y * 0.5f + 0.5f)) * float(H);

        // Projected radius in pixels.
        float proj_r = r * abs(camera.view_proj[1][1]) / clip.w * float(H) * 0.5f;
        if (proj_r < 0.5f) proj_r = 0.5f;

        float2 delta = pix_f - screen;
        float  dist2 = dot(delta, delta);
        float  sigma2 = proj_r * proj_r * 0.18f;  // tighter splat for crisp dots

        if (dist2 > 9.0f * sigma2) continue;  // skip if > 3σ away

        float  g     = exp(-0.5f * dist2 / sigma2);
        float  alpha = opacity * g;

        // Front-to-back blend: src over dst.
        float3 blend = col * alpha * (1.0f - alpha_acc);
        color_acc += blend;
        alpha_acc += alpha * (1.0f - alpha_acc);
    }

    out_pixels[gid.y * W + gid.x] = float4(color_acc, alpha_acc);
}
"#;

/// Sort particle indices back-to-front relative to the camera.
pub fn sort_by_depth(
    entries:   &[(u32, TierLevel)],
    positions: &[f32],
    camera:    &Camera,
) -> Vec<u32> {
    // Include all tiers: TInf sub-pixel particles still get the 0.5px minimum splat.
    let mut t3: Vec<u32> = entries
        .iter()
        .map(|(idx, _)| *idx)
        .collect();

    // Compute view-space depth (dot with forward direction from view_proj row 2).
    // view_proj is column-major [[f32;4];4], so row 2 = [vp[0][2], vp[1][2], vp[2][2], vp[3][2]].
    let fwd = [
        camera.view_proj[0][2],
        camera.view_proj[1][2],
        camera.view_proj[2][2],
    ];
    let w_col = [
        camera.view_proj[0][3],
        camera.view_proj[1][3],
        camera.view_proj[2][3],
        camera.view_proj[3][3],
    ];

    let depth_of = |idx: u32| -> f32 {
        let base = idx as usize * 3;
        let x = positions[base];
        let y = positions[base + 1];
        let z = positions[base + 2];
        // clip.z / clip.w as depth proxy (perspective-correct).
        let clip_z = fwd[0] * x + fwd[1] * y + fwd[2] * z + camera.view_proj[3][2];
        let clip_w = w_col[0] * x + w_col[1] * y + w_col[2] * z + w_col[3];
        if clip_w.abs() < 1e-9 { 0.0 } else { clip_z / clip_w }
    };

    // Sort back-to-front: largest depth first (furthest away).
    t3.sort_unstable_by(|&a, &b| {
        let da = depth_of(a);
        let db = depth_of(b);
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });

    t3
}

/// T3 Gaussian splat pass (compute).
#[allow(dead_code)]
pub struct T3Pass {
    gpu:      crate::gpu::Gpu,
    pipeline: crate::gpu::Pipeline,
    queue:    crate::gpu::Queue,
}

unsafe impl Send for T3Pass {}
unsafe impl Sync for T3Pass {}

impl T3Pass {
    pub fn new() -> Result<Self, crate::gpu::GpuError> {
        let gpu      = crate::gpu::Gpu::open()?;
        let lib      = gpu.compile(SPLAT_SRC)?;
        let func     = lib.function("gaussian_splat")?;
        let pipeline = gpu.pipeline(&func)?;
        let queue    = gpu.new_command_queue()?;
        Ok(Self { gpu, pipeline, queue })
    }

    /// Render Gaussian splats into the frame buffer.
    ///
    /// `sorted_indices` — back-to-front (from `sort_by_depth`); `positions` /
    /// `radii` / `colors` — the CPU epoch mirrors, gathered here without
    /// touching a GPU buffer.
    pub fn draw(
        &self,
        sorted_indices: &[u32],
        positions:      &[f32],
        radii:          &[f32],
        colors:         &[f32],
        camera:         &Camera,
        viewport:       [u32; 2],
        out_buf:        &crate::gpu::Buffer,
        cmd:            &crate::gpu::Commands,
    ) -> Result<(), crate::gpu::GpuError> {
        let [w, h] = viewport;
        let n = sorted_indices.len() as u32;

        // Splats write every pixel, so this pass also clears the frame for
        // the ones layered on top of it. With nothing to draw there is
        // nothing to clear either — the caller allocated a zeroed buffer.
        if n == 0 {
            return Ok(());
        }

        let mut pos_data = Vec::with_capacity(n as usize * 3);
        let mut col_data = Vec::with_capacity(n as usize * 3);
        let mut rad_data = Vec::with_capacity(n as usize);
        for &idx in sorted_indices {
            let base = idx as usize * 3;
            pos_data.extend_from_slice(&positions[base..base + 3]);
            col_data.extend_from_slice(&colors[base..base + 3]);
            rad_data.push(radii[idx as usize]);
        }

        let pos_buf = self.gpu.buffer_with_data(cast_f32(&pos_data))?;
        let rad_buf = self.gpu.buffer_with_data(cast_f32(&rad_data))?;
        let col_buf = self.gpu.buffer_with_data(cast_f32(&col_data))?;

        let camera_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                camera as *const Camera as *const u8,
                std::mem::size_of::<Camera>(),
            )
        };

        let n_bytes:  [u8; 4] = n.to_le_bytes();
        let vp_bytes: [u8; 8] = unsafe { std::mem::transmute([w, h]) };

        let enc = cmd.encoder()?;

        enc.bind(&self.pipeline);
        enc.bind_buffer(&pos_buf, 0, 0);
        enc.bind_buffer(&rad_buf, 0, 1);
        enc.bind_buffer(&col_buf, 0, 2);
        enc.push(camera_bytes,      3);
        enc.push(&n_bytes,          4);
        enc.push(&vp_bytes,         5);
        enc.bind_buffer(out_buf,  0, 6);

        enc.launch((w as usize, h as usize, 1), (16, 16, 1));
        enc.finish();
        Ok(())
    }
}

/// Cast f32 slice to byte slice (safe: u8 has no alignment requirements).
fn cast_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// Platform kernel source: MSL under Metal, WGSL under wgpu.
#[cfg(target_vendor = "apple")]
pub const SPLAT_SRC: &str = SPLAT_MSL;
#[cfg(not(target_vendor = "apple"))]
pub const SPLAT_SRC: &str = SPLAT_WGSL;

#[allow(dead_code)]
pub const SPLAT_WGSL: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    planes: array<vec4<f32>, 6>,
    viewport: vec2<f32>,
    near: f32,
    far: f32,
};

@group(0) @binding(0) var<storage, read> sorted_positions: array<f32>;
@group(0) @binding(1) var<storage, read> sorted_radii: array<f32>;
@group(0) @binding(2) var<storage, read> sorted_colors: array<f32>;
@group(0) @binding(3) var<uniform> camera: Camera;
@group(0) @binding(4) var<uniform> n_splats: u32;
@group(0) @binding(5) var<uniform> viewport: vec2<u32>;
@group(0) @binding(6) var<storage, read_write> out_pixels: array<vec4<f32>>;

@compute @workgroup_size(16, 16)
fn gaussian_splat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let W = viewport.x;
    let H = viewport.y;
    if (gid.x >= W || gid.y >= H) { return; }

    var color_acc = vec3<f32>(0.0);
    var alpha_acc = 0.0;

    let pix_f = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

    for (var i = 0u; i < n_splats; i++) {
        if (alpha_acc >= 0.9999) { break; }

        let center = vec3<f32>(sorted_positions[i * 3u],
                               sorted_positions[i * 3u + 1u],
                               sorted_positions[i * 3u + 2u]);
        let r = sorted_radii[i];
        let col = vec3<f32>(sorted_colors[i * 3u],
                            sorted_colors[i * 3u + 1u],
                            sorted_colors[i * 3u + 2u]);
        let opacity = 1.0;

        let clip = camera.view_proj * vec4<f32>(center, 1.0);
        if (clip.w <= 0.0) { continue; }

        let ndc = clip.xy / clip.w;
        var screen: vec2<f32>;
        screen.x = (ndc.x * 0.5 + 0.5) * f32(W);
        screen.y = (1.0 - (ndc.y * 0.5 + 0.5)) * f32(H);

        var proj_r = r * abs(camera.view_proj[1][1]) / clip.w * f32(H) * 0.5;
        proj_r = max(proj_r, 0.5);

        let delta = pix_f - screen;
        let dist2 = dot(delta, delta);
        let sigma2 = proj_r * proj_r * 0.18;

        if (dist2 > 9.0 * sigma2) { continue; }

        let g = exp(-0.5 * dist2 / sigma2);
        let alpha = opacity * g;

        color_acc += col * alpha * (1.0 - alpha_acc);
        alpha_acc += alpha * (1.0 - alpha_acc);
    }

    out_pixels[gid.y * W + gid.x] = vec4<f32>(color_acc, alpha_acc);
}
"#;
