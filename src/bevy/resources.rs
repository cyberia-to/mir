//! Bevy ECS resources for the graph world.

use std::sync::{Arc, RwLock};
use bevy::prelude::*;

use crate::epoch::EpochState;
use crate::frame::cull::CullPass;
use crate::frame::tiers::t2::T2Pass;
use crate::frame::tiers::t3::T3Pass;
use crate::frame::tiers::tinf::TInfPass;
use crate::frame::edges::{EdgePass, EdgeLinePass};
use crate::graph::Csr;

#[derive(Resource)]
pub struct EpochStateRes {
    pub inner: Arc<RwLock<Option<EpochState>>>,
}

/// Graph data for spawning EpochWorker.
#[derive(Resource, Clone)]
pub struct GraphWorldConfig {
    pub graph: Arc<Csr>,
}

/// 6DOF camera with heat-kernel τ zoom.
#[derive(Resource)]
pub struct GraphCamera {
    pub position: [f32; 3],
    pub yaw:      f32,
    pub pitch:    f32,
    pub fov:      f32,
    pub near:     f32,
    pub far:      f32,
    pub tau:      f32,
    pub tau_target: f32,
    pub viewport: [f32; 2],
    /// Active §9.2 warp animation (None if free-fly).
    pub warp:     Option<WarpAnim>,
}

impl Default for GraphCamera {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 3000.0],
            yaw: 0.0, pitch: 0.0,
            fov: std::f32::consts::FRAC_PI_3,
            near: 1.0, far: 100_000.0,
            tau: 1.0, tau_target: 1.0,
            viewport: [1280.0, 720.0],
            warp: None,
        }
    }
}

impl GraphCamera {
    pub fn forward(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        [cp * sy, sp, -cp * cy]
    }
    pub fn right(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        [cy, 0.0, sy]
    }
    fn up(&self) -> [f32; 3] {
        let r = self.right(); let f = self.forward();
        [r[1]*f[2]-r[2]*f[1], r[2]*f[0]-r[0]*f[2], r[0]*f[1]-r[1]*f[0]]
    }

    /// Column-major 4×4 view matrix. In MSL: m[col][row].
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        let r = self.right(); let u = self.up(); let f = self.forward(); let p = self.position;
        [
            [r[0], u[0], -f[0], 0.0],
            [r[1], u[1], -f[1], 0.0],
            [r[2], u[2], -f[2], 0.0],
            [-(r[0]*p[0]+r[1]*p[1]+r[2]*p[2]),
             -(u[0]*p[0]+u[1]*p[1]+u[2]*p[2]),
              f[0]*p[0]+f[1]*p[1]+f[2]*p[2],
             1.0],
        ]
    }

    /// Column-major perspective projection (OpenGL: z ∈ [-1,1]).
    pub fn proj_matrix(&self) -> [[f32; 4]; 4] {
        let aspect = self.viewport[0] / self.viewport[1].max(1.0);
        let f = 1.0 / (self.fov * 0.5).tan();
        let (n, fa) = (self.near, self.far);
        let range = n - fa;
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (fa + n) / range, -1.0],
            [0.0, 0.0, 2.0 * fa * n / range, 0.0],
        ]
    }

    /// Column-major view-projection matrix.
    pub fn view_proj(&self) -> [[f32; 4]; 4] {
        mat4_mul(&self.proj_matrix(), &self.view_matrix())
    }

    /// 6 frustum planes [nx,ny,nz,d]: dot(n,p)+d≥0 = inside (Gribb-Hartmann).
    pub fn frustum_planes(&self) -> [[f32; 4]; 6] {
        let m = self.view_proj();
        let row = |i: usize| -> [f32; 4] { [m[0][i], m[1][i], m[2][i], m[3][i]] };
        let (r0,r1,r2,r3) = (row(0), row(1), row(2), row(3));
        let add = |a:[f32;4], b:[f32;4]| [a[0]+b[0],a[1]+b[1],a[2]+b[2],a[3]+b[3]];
        let sub = |a:[f32;4], b:[f32;4]| [a[0]-b[0],a[1]-b[1],a[2]-b[2],a[3]-b[3]];
        [add(r3,r0), sub(r3,r0), add(r3,r1), sub(r3,r1), add(r3,r2), sub(r3,r2)]
    }

    /// Build Camera struct for GPU shaders.
    pub fn to_gpu_camera(&self) -> crate::frame::cull::Camera {
        crate::frame::cull::Camera {
            view_proj: self.view_proj(),
            planes:    self.frustum_planes(),
            viewport:  self.viewport,
            near:      self.near,
            far:       self.far,
        }
    }
}

fn mat4_mul(a: &[[f32;4];4], b: &[[f32;4];4]) -> [[f32;4];4] {
    let mut c = [[0.0f32;4];4];
    for col in 0..4 { for row in 0..4 { for k in 0..4 { c[col][row] += a[k][row] * b[col][k]; } } }
    c
}

/// §9.2 warp target: set this resource to trigger a camera warp to a particle.
#[derive(Resource, Default)]
pub struct WarpTarget {
    /// None = no pending warp. Set to Some(particle_idx) to trigger warp next frame.
    pub particle_idx: Option<u32>,
}

/// Warp-to-particle animation state (§9.2).
pub struct WarpAnim {
    pub from_pos:   [f32; 3],
    pub to_pos:     [f32; 3],
    pub to_yaw:     f32,
    pub to_pitch:   f32,
    pub elapsed:    f32,
    /// Total duration (500 ms per §9.2).
    pub duration:   f32,
}

/// GPU buffers and render passes.
/// SAFETY: Metal objects are thread-safe per Metal documentation.
pub struct GpuBuffers {
    pub n_particles: usize,
    pub viewport:    [u32; 2],
    pub gpu:         Option<aruminium::Gpu>,
    pub pos_buf:     Option<aruminium::Buffer>,
    pub rad_buf:     Option<aruminium::Buffer>,
    pub col_buf:     Option<aruminium::Buffer>,
    pub bvh_buf:     Option<aruminium::Buffer>,  // BvhNode array for cull pass
    pub dummy_buf:   Option<aruminium::Buffer>,  // fallback when BVH not ready
    pub cull:        Option<CullPass>,
    pub t2:          Option<T2Pass>,
    pub t3:          Option<T3Pass>,
    pub tinf:        Option<TInfPass>,
    pub edge:        EdgePass,
    pub edge_line:   Option<EdgeLinePass>,
    pub focus:       Vec<f32>,
    pub csr:         Option<Arc<Csr>>,
    pub d_inv:       Vec<f32>,
    pub visible:     Vec<(u32, crate::frame::cull::TierLevel)>,
    pub last_pixels: Option<Vec<f32>>,  // RGBA f32, W×H×4
    pub output_image: Option<Handle<Image>>,
}

unsafe impl Send for GpuBuffers {}
unsafe impl Sync for GpuBuffers {}

impl Resource for GpuBuffers {}

impl Default for GpuBuffers {
    fn default() -> Self {
        Self {
            n_particles: 0, viewport: [1280, 720],
            gpu: None, pos_buf: None, rad_buf: None, col_buf: None,
            bvh_buf: None, dummy_buf: None,
            cull: None, t2: None, t3: None, tinf: None, edge_line: None,
            edge: EdgePass::new(0),
            focus: Vec::new(), csr: None, d_inv: Vec::new(),
            visible: Vec::new(), last_pixels: None, output_image: None,
        }
    }
}

impl GpuBuffers {
    pub fn new() -> Self {
        let mut s = Self::default();
        match aruminium::Gpu::open() {
            Ok(gpu) => {
                s.cull      = CullPass::new().ok();
                s.t2        = T2Pass::new().ok();
                s.t3        = T3Pass::new().ok();
                s.tinf      = TInfPass::new().ok();
                s.edge_line = EdgeLinePass::new().ok();
                s.dummy_buf = gpu.buffer(4).ok();
                s.gpu       = Some(gpu);
            }
            Err(e) => { warn!("mir: GPU init failed: {e}. Rendering disabled."); }
        }
        s
    }

    pub fn upload_epoch(&mut self, epoch: &EpochState) {
        let Some(gpu) = &self.gpu else { return };
        let n = epoch.positions.len() / 3;
        self.n_particles = n;
        self.focus  = epoch.focus.clone();
        self.d_inv  = epoch.d_inv.clone();
        self.pos_buf = gpu.buffer_with_data(cast_f32(&epoch.positions)).ok();
        self.rad_buf = gpu.buffer_with_data(cast_f32(&epoch.radii)).ok();
        self.col_buf = gpu.buffer_with_data(cast_f32(&epoch.colors)).ok();

        // Upload BVH nodes for the cull pass (§10.3).
        if !epoch.bvh.nodes.is_empty() {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    epoch.bvh.nodes.as_ptr() as *const u8,
                    epoch.bvh.nodes.len()
                        * std::mem::size_of::<crate::epoch::bvh::BvhNode>(),
                )
            };
            self.bvh_buf = gpu.buffer_with_data(bytes).ok();
        }
    }
}

fn cast_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
