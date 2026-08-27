//! Pack pass: RGBA f32 frame → packed RGBA8 bytes, background fill included.
//! The one buffer the CPU ever maps is the packed output — a quarter of the
//! float frame — and both per-pixel loops (bg fill, f32→u8) run on the GPU.

pub const PACK_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void pack_rgba8(
    device const float4 *frame   [[buffer(0)]],
    device uint         *dst     [[buffer(1)]],
    constant uint       &n_pix   [[buffer(2)]],
    uint                gid      [[thread_position_in_grid]])
{
    if (gid >= n_pix) return;
    float4 p = frame[gid];
    if (p.w < 0.5f) p = float4(0.0f, 0.0f, 0.0f, 1.0f);
    p = clamp(p, 0.0f, 1.0f);
    dst[gid] = uint(p.x * 255.0f)
             | (uint(p.y * 255.0f) << 8)
             | (uint(p.z * 255.0f) << 16)
             | (255u << 24);
}
"#;

pub const PACK_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> frame: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> n_pix: u32;

@compute @workgroup_size(256)
fn pack_rgba8(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= n_pix) { return; }
    var p = frame[gid.x];
    if (p.w < 0.5) { p = vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    p = clamp(p, vec4<f32>(0.0), vec4<f32>(1.0));
    dst[gid.x] = u32(p.x * 255.0)
               | (u32(p.y * 255.0) << 8u)
               | (u32(p.z * 255.0) << 16u)
               | (255u << 24u);
}
"#;

#[cfg(target_vendor = "apple")]
pub const PACK_SRC: &str = PACK_MSL;
#[cfg(not(target_vendor = "apple"))]
pub const PACK_SRC: &str = PACK_WGSL;

pub struct PackPass {
    pipeline: crate::gpu::Pipeline,
    queue:    crate::gpu::Queue,
}

unsafe impl Send for PackPass {}
unsafe impl Sync for PackPass {}

impl PackPass {
    pub fn new() -> Result<Self, crate::gpu::GpuError> {
        let gpu      = crate::gpu::Gpu::open()?;
        let lib      = gpu.compile(PACK_SRC)?;
        let func     = lib.function("pack_rgba8")?;
        let pipeline = gpu.pipeline(&func)?;
        let queue    = gpu.new_command_queue()?;
        Ok(Self { pipeline, queue })
    }

    pub fn run(
        &self,
        frame: &crate::gpu::Buffer,
        out:   &crate::gpu::Buffer,
        n_pix: u32,
        cmd:   &crate::gpu::Commands,
    ) -> Result<(), crate::gpu::GpuError> {
        let enc = cmd.encoder()?;
        enc.bind(&self.pipeline);
        enc.bind_buffer(frame, 0, 0);
        enc.bind_buffer(out,   0, 1);
        enc.push(&n_pix.to_le_bytes(), 2);
        enc.launch((n_pix as usize, 1, 1), (256, 1, 1));
        enc.finish();
        Ok(())
    }
}
