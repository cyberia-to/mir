//! T2 analytic sphere impostor (§6.3): compute shader ray-casts analytic spheres,
//! writes RGBA f32 pixels into a shared output buffer.
//! aruminium provides compute-only Metal access; vertex+fragment pipelines are not used.

use super::super::cull::{Camera, TierLevel};

/// MSL compute shader: ray-sphere intersection, writes RGBA into out_pixels.
///
/// Layout:  one thread per output pixel.  For each pixel, march through all
/// T2-visible particles sorted front-to-back, test ray–sphere intersection,
/// shade the nearest hit, and alpha-composite into the accumulator.
pub const IMPOSTOR_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Must match Rust `Camera`.
struct Camera {
    float4x4 view_proj;
    float4   planes[6];
    float2   viewport;
    float    near;
    float    far;
};

// Reconstruct view-space ray direction for pixel (u,v) in NDC.
static float3 ray_dir_view(float2 ndc, float4x4 inv_proj) {
    float4 clip = float4(ndc, -1.0f, 1.0f);
    float4 view = inv_proj * clip;
    return normalize(view.xyz / view.w);
}

// Simple 4×4 matrix inversion for projection matrix (column-major).
// Only correct for perspective projection (column 2 and 3 pattern).
static float4x4 inverse_proj(float4x4 m) {
    // For a typical perspective matrix the inverse is analytic.
    // General path: adjugate / determinant.
    float a = m[0][0], b = m[1][1];
    float c = m[2][2], d = m[2][3];
    float e = m[3][2];
    float4x4 inv = float4x4(0);
    inv[0][0] = 1.0f / a;
    inv[1][1] = 1.0f / b;
    inv[2][3] = 1.0f / e;
    inv[3][2] = 1.0f / d;
    inv[3][3] = -c / (d * e);
    return inv;
}

// Ray–sphere intersection.  Ray: origin O + t*D.  Sphere: center C, radius r.
// Returns t of nearest positive hit, or -1 if no hit.
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

kernel void sphere_impostor(
    // Particle data for T2-visible particles (pre-gathered, compact).
    device const float  *positions  [[buffer(0)]],  // n*3 f32 xyz
    device const float  *radii      [[buffer(1)]],  // n   f32
    device const float  *colors     [[buffer(2)]],  // n*3 f32 rgb
    constant Camera     &cam        [[buffer(3)]],
    constant uint       &n_spheres  [[buffer(4)]],
    constant uint2      &viewport   [[buffer(5)]],  // width, height
    // Output: RGBA f32 pixels, row-major.
    device float4       *out_pixels [[buffer(6)]],
    uint2               gid         [[thread_position_in_grid]])
{
    uint W = viewport.x;
    uint H = viewport.y;
    if (gid.x >= W || gid.y >= H) return;

    // NDC of pixel centre.
    float2 ndc;
    ndc.x =  (float(gid.x) + 0.5f) / float(W) * 2.0f - 1.0f;
    ndc.y = -(float(gid.y) + 0.5f) / float(H) * 2.0f + 1.0f;

    // Reconstruct view-space ray from camera position (origin = 0 in view space).
    // Use inverse of view_proj to reconstruct world-space ray.
    float4 near_h = float4(ndc, 0.0f, 1.0f);
    float4 far_h  = float4(ndc, 1.0f, 1.0f);

    // We do ray–sphere in world space: ray origin is the camera world position.
    // Extract camera origin from the inverse of the view matrix embedded in view_proj.
    // For ortho-correct result: unproject two depths.
    // (Phase-1 approximation: treat view_proj[3] column as camera position negated.)
    float3 cam_origin = float3(-cam.view_proj[3][0],
                                -cam.view_proj[3][1],
                                -cam.view_proj[3][2]);

    // Unproject far point to get ray direction.
    float4x4 vp = cam.view_proj;
    // Row 2 gives clip-Z, row 3 gives clip-W.  Unproject by adjoint.
    // Simple approach: step from near (0,0) to far (0,1) in NDC, unproject both.
    // Because this is a projection matrix we can do it analytically for the direction.
    // Approximate: direction = normalize(inv(view_proj) * (ndc,1,1) - cam_origin).
    // Use the view-proj to bring a far-plane point back to world.
    float4 ws_far = float4(ndc.x, ndc.y, 1.0f, 1.0f);
    // We need the inverse of view_proj.  For Phase-1 we derive from column 3.
    // Fallback: compute approximate ray dir from the projection-matrix columns.
    // For standard perspective: ray = normalize( R^T * (ndc/proj_scale, -1) )
    // where R is the rotation block of the view matrix.
    float fx = cam.view_proj[0][0];  // 2*near/(right-left)
    float fy = cam.view_proj[1][1];  // 2*near/(top-bottom)
    // View-space ray direction (before applying camera rotation).
    float3 ray_view = normalize(float3(ndc.x / fx, ndc.y / fy, -1.0f));

    // Rotate to world space using the upper-left 3×3 of view_proj (it encodes R*S).
    // The view matrix is stored column-major.  Columns 0,1,2 are right, up, -forward.
    float3 right   = float3(cam.view_proj[0][0], cam.view_proj[0][1], cam.view_proj[0][2]);
    float3 up      = float3(cam.view_proj[1][0], cam.view_proj[1][1], cam.view_proj[1][2]);
    float3 forward = float3(cam.view_proj[2][0], cam.view_proj[2][1], cam.view_proj[2][2]);

    float3 ray_world = normalize(
        ray_view.x * right +
        ray_view.y * up    +
        ray_view.z * forward);

    // Find nearest sphere hit.
    float  t_min   = 1e9f;
    float3 hit_n   = float3(0, 0, 1);
    float3 hit_col = float3(0, 0, 0);
    bool   hit_any = false;

    for (uint i = 0; i < n_spheres; ++i) {
        float3 center = float3(positions[i*3], positions[i*3+1], positions[i*3+2]);
        float  r      = radii[i];
        float3 col    = float3(colors[i*3], colors[i*3+1], colors[i*3+2]);

        float t = intersect_sphere(cam_origin, ray_world, center, r);
        if (t > 0.0f && t < t_min) {
            t_min   = t;
            hit_any = true;
            float3 hit_pos = cam_origin + t * ray_world;
            hit_n          = normalize(hit_pos - center);
            hit_col        = col;
        }
    }

    float4 result = float4(0, 0, 0, 0);
    if (hit_any) {
        // Diffuse + rim lighting.
        float3 light = normalize(float3(0.5f, 1.0f, 0.8f));
        float  diff  = max(0.0f, dot(hit_n, light));
        float  rim   = 1.0f - max(0.0f, dot(hit_n, -ray_world));
        rim = pow(rim, 3.0f) * 0.4f;
        float3 shade = hit_col * (0.2f + 0.8f * diff) + rim;
        result = float4(shade, 1.0f);
    }

    out_pixels[gid.y * W + gid.x] = result;
}
"#;

/// T2 sphere-impostor pass (compute).
#[allow(dead_code)]
pub struct T2Pass {
    gpu:      crate::gpu::Gpu,
    pipeline: crate::gpu::Pipeline,
    queue:    crate::gpu::Queue,
}

unsafe impl Send for T2Pass {}
unsafe impl Sync for T2Pass {}

impl T2Pass {
    pub fn new() -> Result<Self, crate::gpu::GpuError> {
        let gpu      = crate::gpu::Gpu::open()?;
        let lib      = gpu.compile(IMPOSTOR_SRC)?;
        let func     = lib.function("sphere_impostor")?;
        let pipeline = gpu.pipeline(&func)?;
        let queue    = gpu.new_command_queue()?;
        Ok(Self { gpu, pipeline, queue })
    }

    /// Render T2-tier spheres into a pixel buffer.
    ///
    /// # Arguments
    /// * `visible`   — (particle_idx, tier) pairs from VisibleSet (only T2 entries used)
    /// * `positions` — all-particle position buffer (n×3 f32)
    /// * `radii`     — all-particle radius buffer (n f32)
    /// * `colors`    — all-particle color buffer (n×3 f32)
    /// * `camera`    — camera uniforms
    /// * `viewport`  — [width, height] in pixels
    ///
    /// Returns an RGBA f32 pixel buffer (width×height×4 f32 values).
    pub fn draw(
        &self,
        visible:   &[(u32, TierLevel)],
        positions: &crate::gpu::Buffer,
        radii:     &crate::gpu::Buffer,
        colors:    &crate::gpu::Buffer,
        camera:    &Camera,
        viewport:  [u32; 2],
    ) -> Result<Vec<f32>, crate::gpu::GpuError> {
        let [w, h] = viewport;

        // Gather T2-only particles into compact GPU buffers.
        let t2_indices: Vec<u32> = visible
            .iter()
            .filter(|(_, t)| *t == TierLevel::T2)
            .map(|(idx, _)| *idx)
            .collect();

        let n = t2_indices.len() as u32;
        if n == 0 {
            return Ok(vec![0.0f32; (w * h * 4) as usize]);
        }

        // Compact positions, radii, colors by gathering from CPU side.
        let pos_data = positions.read_f32(|s| {
            let mut d = Vec::with_capacity(n as usize * 3);
            for &idx in &t2_indices {
                let base = idx as usize * 3;
                d.push(s[base]);
                d.push(s[base + 1]);
                d.push(s[base + 2]);
            }
            d
        });
        let rad_data = radii.read_f32(|s| {
            t2_indices.iter().map(|&i| s[i as usize]).collect::<Vec<_>>()
        });
        let col_data = colors.read_f32(|s| {
            let mut d = Vec::with_capacity(n as usize * 3);
            for &idx in &t2_indices {
                let base = idx as usize * 3;
                d.push(s[base]);
                d.push(s[base + 1]);
                d.push(s[base + 2]);
            }
            d
        });

        let pos_buf = self.gpu.buffer_with_data(bytemuck_cast_f32(&pos_data))?;
        let rad_buf = self.gpu.buffer_with_data(bytemuck_cast_f32(&rad_data))?;
        let col_buf = self.gpu.buffer_with_data(bytemuck_cast_f32(&col_data))?;

        // Output pixel buffer.
        let pixel_count = (w * h) as usize;
        let out_buf = self.gpu.buffer(pixel_count * 16)?; // 4 f32 × 4 bytes

        let camera_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                camera as *const Camera as *const u8,
                std::mem::size_of::<Camera>(),
            )
        };

        let n_bytes  = n.to_le_bytes();
        let vp_bytes = [w, h];
        let vp_raw: [u8; 8] = unsafe { std::mem::transmute(vp_bytes) };

        let cmd = self.queue.commands()?;
        let enc = cmd.encoder()?;

        enc.bind(&self.pipeline);
        enc.bind_buffer(&pos_buf, 0, 0);
        enc.bind_buffer(&rad_buf, 0, 1);
        enc.bind_buffer(&col_buf, 0, 2);
        enc.push(camera_bytes,      3);
        enc.push(&n_bytes,          4);
        enc.push(&vp_raw,           5);
        enc.bind_buffer(&out_buf, 0, 6);

        enc.launch((w as usize, h as usize, 1), (16, 16, 1));
        enc.finish();
        cmd.submit();
        cmd.wait();

        let pixels = out_buf.read_f32(|s| s.to_vec());
        Ok(pixels)
    }
}

/// Cast a f32 slice to a byte slice (for buffer_with_data).
fn bytemuck_cast_f32(v: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
    }
}

/// Platform kernel source: MSL under Metal, WGSL under wgpu.
#[cfg(target_vendor = "apple")]
pub const IMPOSTOR_SRC: &str = IMPOSTOR_MSL;
#[cfg(not(target_vendor = "apple"))]
pub const IMPOSTOR_SRC: &str = IMPOSTOR_WGSL;

#[allow(dead_code)]
pub const IMPOSTOR_WGSL: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    planes: array<vec4<f32>, 6>,
    viewport: vec2<f32>,
    near: f32,
    far: f32,
};

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> radii: array<f32>;
@group(0) @binding(2) var<storage, read> colors: array<f32>;
@group(0) @binding(3) var<uniform> cam: Camera;
@group(0) @binding(4) var<uniform> n_spheres: u32;
@group(0) @binding(5) var<uniform> viewport: vec2<u32>;
@group(0) @binding(6) var<storage, read_write> out_pixels: array<vec4<f32>>;

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
    return select(-1.0, t1, t1 > 0.0);
}

@compute @workgroup_size(16, 16)
fn sphere_impostor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let W = viewport.x;
    let H = viewport.y;
    if (gid.x >= W || gid.y >= H) { return; }

    var ndc: vec2<f32>;
    ndc.x = (f32(gid.x) + 0.5) / f32(W) * 2.0 - 1.0;
    ndc.y = -(f32(gid.y) + 0.5) / f32(H) * 2.0 + 1.0;

    // Phase-1 approximation, same as the MSL: view_proj column 3 negated.
    let cam_origin = vec3<f32>(-cam.view_proj[3][0], -cam.view_proj[3][1], -cam.view_proj[3][2]);

    let fx = cam.view_proj[0][0];
    let fy = cam.view_proj[1][1];
    let ray_view = normalize(vec3<f32>(ndc.x / fx, ndc.y / fy, -1.0));

    let right = vec3<f32>(cam.view_proj[0][0], cam.view_proj[0][1], cam.view_proj[0][2]);
    let up = vec3<f32>(cam.view_proj[1][0], cam.view_proj[1][1], cam.view_proj[1][2]);
    let forward = vec3<f32>(cam.view_proj[2][0], cam.view_proj[2][1], cam.view_proj[2][2]);

    let ray_world = normalize(ray_view.x * right + ray_view.y * up + ray_view.z * forward);

    var t_min = 1e9;
    var hit_n = vec3<f32>(0.0, 0.0, 1.0);
    var hit_col = vec3<f32>(0.0);
    var hit_any = false;

    for (var i = 0u; i < n_spheres; i++) {
        let center = vec3<f32>(positions[i * 3u], positions[i * 3u + 1u], positions[i * 3u + 2u]);
        let r = radii[i];
        let col = vec3<f32>(colors[i * 3u], colors[i * 3u + 1u], colors[i * 3u + 2u]);

        let t = intersect_sphere(cam_origin, ray_world, center, r);
        if (t > 0.0 && t < t_min) {
            t_min = t;
            hit_any = true;
            let hit_pos = cam_origin + t * ray_world;
            hit_n = normalize(hit_pos - center);
            hit_col = col;
        }
    }

    var result = vec4<f32>(0.0);
    if (hit_any) {
        let light = normalize(vec3<f32>(0.5, 1.0, 0.8));
        let diff = max(0.0, dot(hit_n, light));
        var rim = 1.0 - max(0.0, dot(hit_n, -ray_world));
        rim = pow(rim, 3.0) * 0.4;
        let shade = hit_col * (0.2 + 0.8 * diff) + vec3<f32>(rim);
        result = vec4<f32>(shade, 1.0);
    }

    out_pixels[gid.y * W + gid.x] = result;
}
"#;
