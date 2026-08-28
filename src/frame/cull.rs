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
pub enum TierLevel {
    T0   = 0,
    T1   = 1,
    T2   = 2,
    T3   = 3,
    TInf = 4,
}

/// Camera uniforms passed to the cull shader.
#[repr(C)]
pub struct Camera {
    /// Column-major 4×4 view-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Six frustum planes, each (nx, ny, nz, d) in world space.
    pub planes:    [[f32; 4]; 6],
    /// Viewport dimensions in pixels.
    pub viewport:  [f32; 2],
    /// Near / far clip distances (used for depth normalisation).
    pub near:      f32,
    pub far:       f32,
    /// Camera position and world-space basis, w unused. The paint pass casts
    /// rays through these; see the shader-side comment for why it cannot take
    /// them out of `view_proj`.
    pub cam_pos:   [f32; 4],
    pub cam_right: [f32; 4],
    pub cam_up:    [f32; 4],
    pub cam_fwd:   [f32; 4],
}

/// GPU BVH frustum-cull + tier-assignment pass.
#[allow(dead_code)]
pub struct CullPass {
    gpu:      crate::gpu::Gpu,
    pipeline: crate::gpu::Pipeline,
    queue:    crate::gpu::Queue,
}

// SAFETY: Metal objects are thread-safe per Metal documentation.
unsafe impl Send for CullPass {}
unsafe impl Sync for CullPass {}

impl CullPass {
    pub fn new() -> Result<Self, crate::gpu::GpuError> {
        let gpu   = crate::gpu::Gpu::open()?;
        let lib   = gpu.compile(BVH_CULL_SRC)?;
        let func  = lib.function("bvh_cull")?;
        let pipeline = gpu.pipeline(&func)?;
        let queue = gpu.new_command_queue()?;
        Ok(Self { gpu, pipeline, queue })
    }

    /// Run the cull pass and return the visible set.
    ///
    /// # Arguments
    /// * `positions`   — GPU buffer, n×3 f32 (xyz per particle)
    /// * `radii`       — GPU buffer, n f32   (r₀·√focus per particle)
    /// * `bvh_nodes`   — GPU buffer, BvhNode array (48 bytes/node)
    /// * `camera`      — camera uniforms (view_proj, planes, viewport)
    /// * `n_particles` — number of particles
    pub fn run(
        &self,
        positions:   &crate::gpu::Buffer,
        radii:       &crate::gpu::Buffer,
        bvh_nodes:   &crate::gpu::Buffer,
        camera:      &Camera,
        n_particles: u32,
    ) -> Result<VisibleSet, crate::gpu::GpuError> {
        // Output buffers:  visible_count (1 u32) + visible_out (n * 8 bytes).
        let count_buf = self.gpu.buffer(4)?;
        count_buf.write(|b| b[..4].fill(0));

        let out_size = (n_particles as usize) * 8; // u32 pair per particle
        let out_buf  = self.gpu.buffer(out_size.max(8))?;

        // Pack camera + n_particles as inline push constants.
        let camera_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                camera as *const Camera as *const u8,
                std::mem::size_of::<Camera>(),
            )
        };

        let cmd = self.queue.commands()?;
        let enc = cmd.encoder()?;

        enc.bind(&self.pipeline);
        enc.bind_buffer(positions,  0, 0);
        enc.bind_buffer(radii,      0, 1);
        enc.bind_buffer(bvh_nodes,  0, 2);
        enc.push(camera_bytes,         3);
        enc.bind_buffer(&count_buf, 0, 4);
        enc.bind_buffer(&out_buf,   0, 5);

        let n_bytes = n_particles.to_le_bytes();
        enc.push(&n_bytes, 6);

        enc.launch((n_particles as usize, 1, 1), (64, 1, 1));
        enc.finish();
        cmd.submit();
        cmd.wait();

        // Read back results.
        let count = count_buf.read(|b| {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        }) as usize;

        let entries = out_buf.read(|b| {
            let mut v = Vec::with_capacity(count);
            for i in 0..count {
                let off = i * 8;
                if off + 8 > b.len() { break; }
                let idx  = u32::from_le_bytes([b[off],   b[off+1], b[off+2], b[off+3]]);
                let tier = u32::from_le_bytes([b[off+4], b[off+5], b[off+6], b[off+7]]);
                let tl = match tier {
                    0 => TierLevel::T0,
                    1 => TierLevel::T1,
                    2 => TierLevel::T2,
                    3 => TierLevel::T3,
                    _ => TierLevel::TInf,
                };
                v.push((idx, tl));
            }
            v
        });

        Ok(VisibleSet { entries })
    }
}

// ── MSL shader ──────────────────────────────────────────────────────────────

pub const BVH_CULL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Must match Rust `Camera` layout (column-major).
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

// Must match `epoch::bvh::BvhNode` plus the is_leaf / leaf fields added here.
// Padding makes total = 48 bytes (matches spec).
struct BvhNode {
    float3 aabb_min;    // 12
    float  focus_sum;   //  4
    float3 aabb_max;    // 12
    uint   child_start; //  4
    uint   child_count; //  4
    uint   is_leaf;     //  4  (1 → leaf node)
    uint   leaf_start;  //  4  (first particle index)
    uint   leaf_count;  //  4  (particles in leaf)
};                      // 48 bytes total

// Tier thresholds (screen-space diameter in pixels).
constant float S_T0 = 200.0f;
constant float S_T1 =  40.0f;
constant float S_T2 =   8.0f;
constant float S_T3 =   1.0f;

// Test AABB (min, max) against one frustum plane.
// Returns true if ALL 8 corners are on the negative side → node culled.
static bool aabb_outside_plane(float3 aabb_min, float3 aabb_max, float4 plane) {
    // p-vertex: corner maximising dot(normal, corner)
    float3 p;
    p.x = (plane.x >= 0.0f) ? aabb_max.x : aabb_min.x;
    p.y = (plane.y >= 0.0f) ? aabb_max.y : aabb_min.y;
    p.z = (plane.z >= 0.0f) ? aabb_max.z : aabb_min.z;
    return dot(plane.xyz, p) + plane.w < 0.0f;
}

// Test AABB against all 6 frustum planes.
// Returns true if the AABB is fully outside any plane → cull.
static bool aabb_culled(float3 aabb_min, float3 aabb_max, constant Camera &cam) {
    for (int i = 0; i < 6; ++i) {
        if (aabb_outside_plane(aabb_min, aabb_max, cam.planes[i])) return true;
    }
    return false;
}

// Project world-space sphere (center, radius) to screen-space diameter (pixels).
static float screen_diameter(float3 center, float radius, constant Camera &cam) {
    float4 clip = cam.view_proj * float4(center, 1.0f);
    if (clip.w <= 0.0f) return 0.0f;
    // Projected radius estimate: use column[1][1] (fov scale in Y).
    // cam_up.w is the focal length; view_proj[1][1] is that times up.y, so
    // using it would shrink every particle as the camera pitches and drop
    // whole tiers below the solid threshold for no reason but the angle.
    float proj_r = radius * cam.cam_up.w / clip.w;
    // Convert to pixels (proj_r is in NDC halves, viewport.y is full height).
    return proj_r * cam.viewport.y;
}

static uint diameter_to_tier(float diam) {
    if (diam >= S_T0) return 0;
    if (diam >= S_T1) return 1;
    if (diam >= S_T2) return 2;
    if (diam >= S_T3) return 3;
    return 4; // TInf
}

kernel void bvh_cull(
    device const float  *positions_x   [[buffer(0)]],   // interleaved xyz: 3*n floats
    device const float  *radii         [[buffer(1)]],
    device const BvhNode *bvh          [[buffer(2)]],
    constant Camera     &cam           [[buffer(3)]],
    device atomic_uint  *visible_count [[buffer(4)]],
    device uint2        *visible_out   [[buffer(5)]],    // (particle_idx, tier)
    constant uint       &n_particles   [[buffer(6)]],
    uint                 gid           [[thread_position_in_grid]])
{
    if (gid >= n_particles) return;

    // positions buffer is interleaved xyz.
    device const float *p = positions_x + gid * 3;
    float3 pos   = float3(p[0], p[1], p[2]);
    float  radius = radii[gid];

    // Fast point-cull first (degenerate AABB = point).
    float3 pmin = pos - radius;
    float3 pmax = pos + radius;

    if (aabb_culled(pmin, pmax, cam)) return;

    float  diam = screen_diameter(pos, radius, cam);
    uint   tier = diameter_to_tier(diam);

    uint slot = atomic_fetch_add_explicit(visible_count, 1u, memory_order_relaxed);
    visible_out[slot] = uint2(gid, tier);
}
"#;

/// The kernel for the platform's device layer: MSL under Metal, WGSL under
/// wgpu. Same math, same buffer indices; WebGPU dispatches whole workgroups,
/// so the WGSL side leans on the same `gid >= n` guard Metal already had.
#[cfg(target_vendor = "apple")]
pub const BVH_CULL_SRC: &str = BVH_CULL_MSL;
#[cfg(not(target_vendor = "apple"))]
pub const BVH_CULL_SRC: &str = BVH_CULL_WGSL;

#[allow(dead_code)]
pub const BVH_CULL_WGSL: &str = r#"
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

struct BvhNode {
    aabb_min: vec3<f32>,
    focus_sum: f32,
    aabb_max: vec3<f32>,
    child_start: u32,
    child_count: u32,
    is_leaf: u32,
    leaf_start: u32,
    leaf_count: u32,
};

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> radii: array<f32>;
@group(0) @binding(2) var<storage, read> bvh: array<BvhNode>;
@group(0) @binding(3) var<uniform> cam: Camera;
@group(0) @binding(4) var<storage, read_write> visible_count: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> visible_out: array<vec2<u32>>;
@group(0) @binding(6) var<uniform> n_particles: u32;

const S_T0: f32 = 200.0;
const S_T1: f32 = 40.0;
const S_T2: f32 = 8.0;
const S_T3: f32 = 1.0;

fn aabb_outside_plane(aabb_min: vec3<f32>, aabb_max: vec3<f32>, plane: vec4<f32>) -> bool {
    var p: vec3<f32>;
    p.x = select(aabb_min.x, aabb_max.x, plane.x >= 0.0);
    p.y = select(aabb_min.y, aabb_max.y, plane.y >= 0.0);
    p.z = select(aabb_min.z, aabb_max.z, plane.z >= 0.0);
    return dot(plane.xyz, p) + plane.w < 0.0;
}

fn aabb_culled(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    for (var i = 0; i < 6; i++) {
        if (aabb_outside_plane(aabb_min, aabb_max, cam.planes[i])) { return true; }
    }
    return false;
}

fn screen_diameter(center: vec3<f32>, radius: f32) -> f32 {
    let clip = cam.view_proj * vec4<f32>(center, 1.0);
    if (clip.w <= 0.0) { return 0.0; }
    // cam_up.w is the focal length; view_proj[1][1] is that times up.y, so
    // using it would shrink every particle as the camera pitches and drop
    // whole tiers below the solid threshold for no reason but the angle.
    let proj_r = radius * cam.cam_up.w / clip.w;
    return proj_r * cam.viewport.y;
}

fn diameter_to_tier(diam: f32) -> u32 {
    if (diam >= S_T0) { return 0u; }
    if (diam >= S_T1) { return 1u; }
    if (diam >= S_T2) { return 2u; }
    if (diam >= S_T3) { return 3u; }
    return 4u;
}

@compute @workgroup_size(64)
fn bvh_cull(@builtin(global_invocation_id) gid3: vec3<u32>) {
    // Keeps the BVH binding live for the auto layout; the per-particle
    // point-cull below does not walk it (same as the Metal kernel).
    _ = arrayLength(&bvh);

    let gid = gid3.x;
    if (gid >= n_particles) { return; }

    let pos = vec3<f32>(positions[gid * 3u], positions[gid * 3u + 1u], positions[gid * 3u + 2u]);
    let radius = radii[gid];

    let pmin = pos - vec3<f32>(radius);
    let pmax = pos + vec3<f32>(radius);
    if (aabb_culled(pmin, pmax)) { return; }

    let diam = screen_diameter(pos, radius);
    let tier = diameter_to_tier(diam);

    let slot = atomicAdd(&visible_count, 1u);
    visible_out[slot] = vec2<u32>(gid, tier);
}
"#;
