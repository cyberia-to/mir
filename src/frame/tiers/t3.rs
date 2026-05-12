//! T3 Gaussian splat: isotropic 3D Gaussian per Kerbl et al. 2023 (Phase 1).
//! CPU depth sort (viable ≤10K particles), GPU compute rasterises into pixel buffer.
//! Step 6.

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
    device const float  *sorted_opacity   [[buffer(3)]],  // n   f32
    constant Camera     &camera           [[buffer(4)]],
    constant uint       &n_splats         [[buffer(5)]],
    constant uint2      &viewport         [[buffer(6)]],
    device float4       *out_pixels       [[buffer(7)]],  // RGBA f32, row-major
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
        float  opacity = sorted_opacity[i];

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
        float  sigma2 = proj_r * proj_r * 0.5f;  // Gaussian σ² = (proj_r/√2)²

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
    // Only sort T3 entries.
    let mut t3: Vec<u32> = entries
        .iter()
        .filter(|(_, t)| *t == TierLevel::T3)
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
    gpu:      aruminium::Gpu,
    pipeline: aruminium::Pipeline,
    queue:    aruminium::Queue,
}

impl T3Pass {
    pub fn new() -> Result<Self, aruminium::GpuError> {
        let gpu      = aruminium::Gpu::open()?;
        let lib      = gpu.compile(SPLAT_MSL)?;
        let func     = lib.function("gaussian_splat")?;
        let pipeline = gpu.pipeline(&func)?;
        let queue    = gpu.new_command_queue()?;
        Ok(Self { gpu, pipeline, queue })
    }

    /// Render Gaussian splats and return an RGBA f32 pixel buffer.
    ///
    /// # Arguments
    /// * `sorted_indices` — particle indices sorted back-to-front (from `sort_by_depth`)
    /// * `positions`      — all-particle position buffer (n×3 f32)
    /// * `radii`          — all-particle radius buffer (n f32)
    /// * `colors`         — all-particle color buffer (n×3 f32)
    /// * `camera`         — camera uniforms
    /// * `viewport`       — [width, height] in pixels
    ///
    /// Returns an RGBA f32 pixel buffer (width×height×4 f32 values).
    pub fn draw(
        &self,
        sorted_indices: &[u32],
        positions:      &aruminium::Buffer,
        radii:          &aruminium::Buffer,
        colors:         &aruminium::Buffer,
        camera:         &Camera,
        viewport:       [u32; 2],
    ) -> Result<Vec<f32>, aruminium::GpuError> {
        let [w, h] = viewport;
        let n = sorted_indices.len() as u32;

        if n == 0 {
            return Ok(vec![0.0f32; (w * h * 4) as usize]);
        }

        // Gather sorted compact buffers on CPU.
        let pos_data = positions.read_f32(|s| {
            let mut d = Vec::with_capacity(n as usize * 3);
            for &idx in sorted_indices {
                let base = idx as usize * 3;
                d.push(s[base]);
                d.push(s[base + 1]);
                d.push(s[base + 2]);
            }
            d
        });
        let rad_data = radii.read_f32(|s| {
            sorted_indices.iter().map(|&i| s[i as usize]).collect::<Vec<_>>()
        });
        let col_data = colors.read_f32(|s| {
            let mut d = Vec::with_capacity(n as usize * 3);
            for &idx in sorted_indices {
                let base = idx as usize * 3;
                d.push(s[base]);
                d.push(s[base + 1]);
                d.push(s[base + 2]);
            }
            d
        });
        // Opacity: uniform 1.0 for Phase 1 (no opacity buffer in spec yet).
        let opacity_data: Vec<f32> = vec![1.0f32; n as usize];

        let pos_buf = self.gpu.buffer_with_data(cast_f32(&pos_data))?;
        let rad_buf = self.gpu.buffer_with_data(cast_f32(&rad_data))?;
        let col_buf = self.gpu.buffer_with_data(cast_f32(&col_data))?;
        let opa_buf = self.gpu.buffer_with_data(cast_f32(&opacity_data))?;

        let pixel_count = (w * h) as usize;
        let out_buf = self.gpu.buffer(pixel_count * 16)?;

        let camera_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                camera as *const Camera as *const u8,
                std::mem::size_of::<Camera>(),
            )
        };

        let n_bytes:  [u8; 4] = n.to_le_bytes();
        let vp_bytes: [u8; 8] = unsafe { std::mem::transmute([w, h]) };

        let cmd = self.queue.commands()?;
        let enc = cmd.encoder()?;

        enc.bind(&self.pipeline);
        enc.bind_buffer(&pos_buf, 0, 0);
        enc.bind_buffer(&rad_buf, 0, 1);
        enc.bind_buffer(&col_buf, 0, 2);
        enc.bind_buffer(&opa_buf, 0, 3);
        enc.push(camera_bytes,      4);
        enc.push(&n_bytes,          5);
        enc.push(&vp_bytes,         6);
        enc.bind_buffer(&out_buf, 0, 7);

        enc.launch((w as usize, h as usize, 1), (16, 16, 1));
        enc.finish();
        cmd.submit();
        cmd.wait();

        let pixels = out_buf.read_f32(|s| s.to_vec());
        Ok(pixels)
    }
}

/// Cast f32 slice to byte slice (safe: u8 has no alignment requirements).
fn cast_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
