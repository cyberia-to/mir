//! T∞: background fill for sub-pixel particles (§6.5).
//!
//! Current implementation: τ-tinted radial gradient using dominant cluster
//! color from BVH focus sums (§10). This is the GPU background pass that
//! runs every frame.
//!
//! For neural background, see `nrf`: hash-grid MLP (§7.5 fallback, no
//! graph-context conditioning). Full §7.1 NRF requires a compiled CT-1.1
//! model produced by the `tru` crate.

/// MSL compute kernel: fill pixels where alpha < threshold with a τ-tinted
/// background.  Background color is a radial gradient from cluster-0 color
/// (centre) to deep-space black (edges), modulated by the τ parameter.
///
/// Buffer layout:
///   0 — float4 in/out pixels  (RGBA f32, row-major, W×H)
///   1 — float4 cluster_color  (dominant cluster tint from BVH focus-sum)
///   2 — float  tau            (current heat-kernel scale, drives fog density)
///   3 — uint2  viewport       (width, height)
pub const TINF_BG_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void tinf_background(
    device float4       *pixels        [[buffer(0)]],
    constant float4     &cluster_color [[buffer(1)]],
    constant float      &tau           [[buffer(2)]],
    constant uint2      &viewport      [[buffer(3)]],
    uint2               gid            [[thread_position_in_grid]])
{
    uint W = viewport.x;
    uint H = viewport.y;
    if (gid.x >= W || gid.y >= H) return;

    uint idx = gid.y * W + gid.x;
    float4 px = pixels[idx];

    // Only fill transparent pixels (alpha < 0.5 → not covered by T2/T3).
    if (px.w >= 0.5f) return;

    // Radial gradient from centre: r ∈ [0, 1].
    float2 uv;
    uv.x = (float(gid.x) + 0.5f) / float(W) * 2.0f - 1.0f;
    uv.y = (float(gid.y) + 0.5f) / float(H) * 2.0f - 1.0f;
    float r = min(length(uv), 1.0f);

    // τ drives fog density: larger τ = more diffuse, lighter background.
    float fog_t = clamp(log(tau + 1.0f) / log(101.0f), 0.0f, 1.0f);

    // Centre: cluster tint at 10% brightness. Edge: black.
    float3 centre_col = cluster_color.rgb * (0.08f + 0.12f * fog_t);
    float3 bg = centre_col * (1.0f - r * r);

    // Blend with any partial transparency from T3 splats.
    float src_a = 1.0f - px.w;
    pixels[idx] = float4(bg * src_a + px.rgb * px.w, 1.0f);
}
"#;

/// T∞ background pass (compute).
pub struct TInfPass {
    gpu:      aruminium::Gpu,
    pipeline: aruminium::Pipeline,
    queue:    aruminium::Queue,
}

unsafe impl Send for TInfPass {}
unsafe impl Sync for TInfPass {}

impl TInfPass {
    pub fn new() -> Result<Self, aruminium::GpuError> {
        let gpu      = aruminium::Gpu::open()?;
        let lib      = gpu.compile(TINF_BG_MSL)?;
        let func     = lib.function("tinf_background")?;
        let pipeline = gpu.pipeline(&func)?;
        let queue    = gpu.new_command_queue()?;
        Ok(Self { gpu, pipeline, queue })
    }

    /// Fill background pixels in-place in `pixels` (RGBA f32, W×H×4).
    ///
    /// `cluster_color` — dominant cluster tint [R, G, B, 1].
    /// `tau`           — current heat-kernel scale.
    /// `viewport`      — [width, height].
    pub fn fill_background(
        &self,
        pixels:        &mut Vec<f32>,
        cluster_color: [f32; 4],
        tau:           f32,
        viewport:      [u32; 2],
    ) -> Result<(), aruminium::GpuError> {
        let [w, h] = viewport;
        let pixel_count = (w * h) as usize;
        if pixels.len() < pixel_count * 4 {
            return Ok(());
        }

        let px_buf = self.gpu.buffer_with_data(cast_f32(pixels))?;
        let col_bytes: [u8; 16] = unsafe { std::mem::transmute(cluster_color) };
        let tau_bytes: [u8; 4]  = tau.to_le_bytes();
        let vp_bytes:  [u8; 8]  = unsafe { std::mem::transmute([w, h]) };

        let cmd = self.queue.commands()?;
        let enc = cmd.encoder()?;

        enc.bind(&self.pipeline);
        enc.bind_buffer(&px_buf, 0, 0);
        enc.push(&col_bytes,       1);
        enc.push(&tau_bytes,       2);
        enc.push(&vp_bytes,        3);

        enc.launch((w as usize, h as usize, 1), (16, 16, 1));
        enc.finish();
        cmd.submit();
        cmd.wait();

        // Read result back into pixels.
        let result = px_buf.read_f32(|s| s.to_vec());
        let copy_len = pixels.len().min(result.len());
        pixels[..copy_len].copy_from_slice(&result[..copy_len]);
        Ok(())
    }
}

impl Default for TInfPass {
    fn default() -> Self {
        Self::new().expect("TInfPass: GPU unavailable")
    }
}

fn cast_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tinf_msl_source_nonempty() {
        assert!(!TINF_BG_MSL.is_empty());
        assert!(TINF_BG_MSL.contains("tinf_background"));
    }
}
