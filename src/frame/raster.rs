//! Spec T2/T1 raster: instanced sphere impostors + edge quads.
//!
//! One quad per visible particle, ray-cast in the fragment. One quad per
//! kept link, camera-facing. Cost follows coverage, not |P| × pixels.
//! Color target is Shared so the CPU can read it; depth stays Private.

use aruminium::ffi::{MTLPixelFormatDepth32Float, MTLPixelFormatRGBA8Unorm};
use aruminium::{
    BlendState, Buffer, ColorAttachmentDesc, CullMode, DepthAttachmentDesc, DepthStencil, Gpu,
    GpuError, PrimitiveType, RenderPassDescriptor, RenderPipeline, RenderPipelineSpec, Texture,
};

use super::cull::{Camera, TierLevel};

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Uni {
    float4x4 view_proj;
    float4   cam_pos;
    float4   cam_right;
    float4   cam_up;
    float4   cam_fwd;
    float2   viewport;
    float    near;
    float    far;
};

// float4 pairs — Metal float3 is 16-byte aligned and would not match Rust.
struct Sphere {
    float4 pos_r;
    float4 rgb_a;
};

struct Edge {
    float4 p_w;
    float4 q;
    float4 rgb_a;
};

struct VOut {
    float4 position [[position]];
    float3 world;
    float3 center;
    float  radius;
    float3 color;
};

struct FragOut {
    float4 color [[color(0)]];
    float  depth [[depth(any)]];
};

constant float3 LIGHT = float3(0.36370f, 0.72739f, 0.58191f);

// GraphCamera::proj_matrix is OpenGL (z ∈ [-1,1]). Metal clip z is [0,1].
static float4 to_metal(float4 clip) {
    clip.z = 0.5f * (clip.z + clip.w);
    return clip;
}

vertex VOut v_sphere(uint vid [[vertex_id]], uint iid [[instance_id]],
                     constant Sphere *sph [[buffer(0)]],
                     constant Uni    &u   [[buffer(1)]]) {
    float2 c[4] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,1) };
    float2 q = c[vid];
    Sphere s = sph[iid];
    float3 rgt = u.cam_right.xyz;
    float3 up  = u.cam_up.xyz;
    float3 pos = s.pos_r.xyz + (rgt * q.x + up * q.y) * s.pos_r.w * 1.15f;
    VOut o;
    o.position = to_metal(u.view_proj * float4(pos, 1.0f));
    o.world = pos;
    o.center = s.pos_r.xyz;
    o.radius = s.pos_r.w;
    o.color = s.rgb_a.xyz;
    return o;
}

fragment FragOut f_sphere(VOut in [[stage_in]], constant Uni &u [[buffer(1)]]) {
    float3 O = u.cam_pos.xyz;
    float3 D = normalize(in.world - O);
    float3 oc = O - in.center;
    float a = dot(D, D);
    float hb = dot(D, oc);
    float cc = dot(oc, oc) - in.radius * in.radius;
    float disc = hb * hb - a * cc;
    FragOut o;
    if (disc < 0.0f) discard_fragment();
    float t = (-hb - sqrt(disc)) / a;
    if (t < 0.0f) t = (-hb + sqrt(disc)) / a;
    if (t < 0.0f) discard_fragment();
    float3 hit = O + t * D;
    float3 n = normalize(hit - in.center);
    float diff = max(0.0f, dot(n, LIGHT));
    float rim = pow(1.0f - max(0.0f, dot(n, -D)), 3.0f) * 0.4f;
    float3 rgb = in.color * (0.18f + 0.82f * diff) + rim;
    float4 clip = to_metal(u.view_proj * float4(hit, 1.0f));
    o.color = float4(rgb, 1.0f);
    o.depth = clip.z / max(clip.w, 1e-5f);
    return o;
}

vertex VOut v_edge(uint vid [[vertex_id]], uint iid [[instance_id]],
                   constant Edge *ed [[buffer(0)]],
                   constant Uni  &u  [[buffer(1)]]) {
    float2 c[4] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,1) };
    float2 q = c[vid];
    Edge e = ed[iid];
    float3 p = e.p_w.xyz;
    float3 r = e.q.xyz;
    float3 axis = r - p;
    float len = length(axis);
    float3 dir = len > 1e-5f ? axis / len : float3(0,1,0);
    float3 eye = normalize(u.cam_pos.xyz - (p + r) * 0.5f);
    float3 side = cross(dir, eye);
    float sl = length(side);
    side = sl > 1e-4f ? side / sl : u.cam_right.xyz;
    float3 along = dir * (q.y * 0.5f + 0.5f) * len;
    float3 pos = p + along + side * q.x * e.p_w.w;
    VOut o;
    o.position = to_metal(u.view_proj * float4(pos, 1.0f));
    o.world = pos;
    o.center = p;
    o.radius = e.p_w.w;
    o.color = e.rgb_a.xyz;
    return o;
}

fragment FragOut f_edge(VOut in [[stage_in]], constant Uni &u [[buffer(1)]]) {
    float4 clip = to_metal(u.view_proj * float4(in.world, 1.0f));
    FragOut o;
    o.color = float4(in.color, 0.55f);
    o.depth = clip.z / max(clip.w, 1e-5f);
    return o;
}
"#;

#[repr(C)]
struct Uni {
    view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    cam_fwd: [f32; 4],
    viewport: [f32; 2],
    near: f32,
    far: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Sphere {
    pos_r: [f32; 4],
    rgb_a: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Edge {
    p_w: [f32; 4],
    q: [f32; 4],
    rgb_a: [f32; 4],
}

const MAX_SPHERES: usize = 8192;
const MAX_EDGES: usize = 384;

pub struct RasterPass {
    gpu: Gpu,
    sphere_pipe: RenderPipeline,
    edge_pipe: RenderPipeline,
    depth_state: aruminium::DepthStencilState,
    edge_depth: aruminium::DepthStencilState,
    color: Option<Texture>,
    depth: Option<Texture>,
    size: [u32; 2],
    sph_buf: Option<Buffer>,
    edge_buf: Option<Buffer>,
    queue: aruminium::Queue,
}

impl RasterPass {
    pub fn new() -> Result<Self, GpuError> {
        let gpu = Gpu::open()?;
        let lib = gpu.compile(SHADER)?;
        let vs = lib.function("v_sphere")?;
        let fs = lib.function("f_sphere")?;
        let ve = lib.function("v_edge")?;
        let fe = lib.function("f_edge")?;
        let sphere_pipe = gpu.render_pipeline(
            &vs,
            &fs,
            &RenderPipelineSpec::color(MTLPixelFormatRGBA8Unorm)
                .with_depth(MTLPixelFormatDepth32Float),
        )?;
        let edge_spec = RenderPipelineSpec::color(MTLPixelFormatRGBA8Unorm)
            .with_depth(MTLPixelFormatDepth32Float)
            .with_blend(0, BlendState::additive());
        let edge_pipe = gpu.render_pipeline(&ve, &fe, &edge_spec)?;
        let depth_state = gpu.depth_stencil_state(DepthStencil::less_write())?;
        let edge_depth = gpu.depth_stencil_state(DepthStencil::less_no_write())?;
        let queue = gpu.new_command_queue()?;
        Ok(Self {
            gpu,
            sphere_pipe,
            edge_pipe,
            depth_state,
            edge_depth,
            color: None,
            depth: None,
            size: [0, 0],
            sph_buf: None,
            edge_buf: None,
            queue,
        })
    }

    pub fn draw(
        &mut self,
        visible: &[(u32, TierLevel)],
        positions: &[f32],
        radii: &[f32],
        colors: &[f32],
        edges: &[(u32, u32)],
        weights: &[f32],
        camera: &Camera,
        viewport: [u32; 2],
        dst: &mut [u8],
    ) -> Result<(), GpuError> {
        let [w, h] = viewport;
        if w == 0 || h == 0 {
            return Ok(());
        }
        if self.size != viewport {
            self.color = Some(
                self.gpu
                    .render_target_shared(w, h, MTLPixelFormatRGBA8Unorm)?,
            );
            self.depth = Some(self.gpu.depth_target(w, h, MTLPixelFormatDepth32Float)?);
            self.size = viewport;
        }
        let color = self.color.as_ref().unwrap();
        let depth = self.depth.as_ref().unwrap();

        let focal = camera.cam_up[3].max(1.0);
        let cam = [camera.cam_pos[0], camera.cam_pos[1], camera.cam_pos[2]];

        let mut spheres: Vec<Sphere> = Vec::with_capacity(visible.len().min(MAX_SPHERES));
        for &(idx, tier) in visible.iter().take(MAX_SPHERES) {
            let b = idx as usize * 3;
            if b + 2 >= positions.len() {
                continue;
            }
            let pos = [positions[b], positions[b + 1], positions[b + 2]];
            let r = radii.get(idx as usize).copied().unwrap_or(1.0);
            let dist = {
                let dx = pos[0] - cam[0];
                let dy = pos[1] - cam[1];
                let dz = pos[2] - cam[2];
                (dx * dx + dy * dy + dz * dz).sqrt().max(1.0)
            };
            let radius = if (tier as u8) <= (TierLevel::T2 as u8) {
                r.max(dist / focal * 0.6)
            } else {
                (1.2 * dist / focal).max(r)
            };
            let col = if b + 2 < colors.len() {
                [colors[b], colors[b + 1], colors[b + 2]]
            } else {
                [0.15, 0.85, 0.35]
            };
            spheres.push(Sphere {
                pos_r: [pos[0], pos[1], pos[2], radius],
                rgb_a: [col[0], col[1], col[2], 1.0],
            });
        }

        let cap = if visible.len() > 1600 {
            MAX_EDGES / 2
        } else {
            MAX_EDGES
        };
        let mut ranked: Vec<(f32, usize)> = edges
            .iter()
            .enumerate()
            .map(|(i, _)| (weights.get(i).copied().unwrap_or(0.2), i))
            .collect();
        if ranked.len() > cap {
            ranked.select_nth_unstable_by(cap, |a, b| b.0.total_cmp(&a.0));
            ranked.truncate(cap);
        }
        let mut edge_inst: Vec<Edge> = Vec::with_capacity(ranked.len());
        for &(wgt, k) in &ranked {
            let (p, q) = edges[k];
            let bp = p as usize * 3;
            let bq = q as usize * 3;
            if bp + 2 >= positions.len() || bq + 2 >= positions.len() {
                continue;
            }
            let pp = [positions[bp], positions[bp + 1], positions[bp + 2]];
            let qq = [positions[bq], positions[bq + 1], positions[bq + 2]];
            let mx = (pp[0] + qq[0]) * 0.5 - cam[0];
            let my = (pp[1] + qq[1]) * 0.5 - cam[1];
            let mz = (pp[2] + qq[2]) * 0.5 - cam[2];
            let dist = (mx * mx + my * my + mz * mz).sqrt().max(1.0);
            let px = (wgt.sqrt() * 2.4).clamp(1.1, 4.5);
            let width = px * dist / focal;
            let c0 = if bp + 2 < colors.len() {
                [colors[bp], colors[bp + 1], colors[bp + 2]]
            } else {
                [0.05, 0.5, 0.15]
            };
            let c1 = if bq + 2 < colors.len() {
                [colors[bq], colors[bq + 1], colors[bq + 2]]
            } else {
                c0
            };
            edge_inst.push(Edge {
                p_w: [pp[0], pp[1], pp[2], width],
                q: [qq[0], qq[1], qq[2], 0.0],
                rgb_a: [
                    (c0[0] + c1[0]) * 0.18,
                    (c0[1] + c1[1]) * 0.18,
                    (c0[2] + c1[2]) * 0.18,
                    1.0,
                ],
            });
        }

        upload(&self.gpu, &mut self.sph_buf, &spheres)?;
        upload(&self.gpu, &mut self.edge_buf, &edge_inst)?;

        let uni = Uni {
            view_proj: camera.view_proj,
            cam_pos: camera.cam_pos,
            cam_right: camera.cam_right,
            cam_up: camera.cam_up,
            cam_fwd: camera.cam_fwd,
            viewport: [w as f32, h as f32],
            near: camera.near,
            far: camera.far,
        };
        let uni_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(&uni as *const Uni as *const u8, std::mem::size_of::<Uni>())
        };

        let mut pass = RenderPassDescriptor::new();
        pass.color_attachment(
            0,
            ColorAttachmentDesc::clear(color, [0.02, 0.03, 0.02, 1.0]),
        );
        pass.depth_attachment(DepthAttachmentDesc::clear(depth));

        let cmd = self.queue.commands()?;
        let enc = cmd.render_encoder(&pass)?;
        enc.set_viewport(0.0, 0.0, w as f64, h as f64, 0.0, 1.0);
        enc.set_cull_mode(CullMode::None);

        if !spheres.is_empty() {
            enc.bind(&self.sphere_pipe);
            enc.set_depth_stencil_state(&self.depth_state);
            enc.set_vertex_buffer(0, self.sph_buf.as_ref().unwrap(), 0);
            enc.push_vertex(uni_bytes, 1);
            enc.push_fragment(uni_bytes, 1);
            enc.draw_instanced(PrimitiveType::TriangleStrip, 0, 4, spheres.len() as u32);
        }
        if !edge_inst.is_empty() {
            enc.bind(&self.edge_pipe);
            enc.set_depth_stencil_state(&self.edge_depth);
            enc.set_vertex_buffer(0, self.edge_buf.as_ref().unwrap(), 0);
            enc.push_vertex(uni_bytes, 1);
            enc.push_fragment(uni_bytes, 1);
            enc.draw_instanced(PrimitiveType::TriangleStrip, 0, 4, edge_inst.len() as u32);
        }
        enc.end();
        cmd.submit();
        cmd.wait();

        let need = (w as usize) * (h as usize) * 4;
        if dst.len() < need {
            return Ok(());
        }
        unsafe {
            color.get_bytes(
                dst.as_mut_ptr() as *mut _,
                (w as usize) * 4,
                aruminium::ffi::MTLRegion {
                    origin: aruminium::ffi::MTLOrigin { x: 0, y: 0, z: 0 },
                    size: aruminium::ffi::MTLSize {
                        width: w as usize,
                        height: h as usize,
                        depth: 1,
                    },
                },
                0,
            );
        }
        Ok(())
    }
}

fn upload<T: Copy>(gpu: &Gpu, slot: &mut Option<Buffer>, data: &[T]) -> Result<(), GpuError> {
    let bytes = as_bytes(data);
    let need = bytes.len().max(64).next_power_of_two();
    if slot.as_ref().map(|b| b.size()).unwrap_or(0) < need {
        *slot = Some(gpu.buffer(need)?);
    }
    let buf = slot.as_ref().unwrap();
    buf.write(|dst| {
        if !bytes.is_empty() {
            dst[..bytes.len()].copy_from_slice(bytes);
        }
    });
    Ok(())
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
