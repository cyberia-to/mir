//! T1 surface + label: analytic impostor sphere + world-space text. Step 8.
//!
//! Labels are rendered in world-space via sugarloaf text rendering.
//! For Phase 1 this is a stub ready to wire in once the graph world
//! surface (Bevy 3D scene) is available.
//!
//! R-1.0 §6.1: T1 fires when screen-pixel diameter ∈ [S_T2, S_T1) = [8, 40) px.
//! Each T1 particle shows:
//!   • Analytic impostor sphere (fragment shader SDF)
//!   • World-space text label at center + radius offset (sugarloaf)
//!   • Luminance driven by φ* (focus value)

use crate::frame::cull::{Camera, TierLevel};

pub struct T1Pass;

impl T1Pass {
    pub fn new() -> Self { Self }

    /// Draw T1 impostors and labels for all visible T1 particles.
    ///
    /// `visible`   — slice of (particle_idx, tier) from the cull pass.
    /// `positions` — flat f32 array, stride 3 (x, y, z per particle).
    /// `titles`    — human-readable name / CID per particle.
    /// `camera`    — camera matrices for world→screen projection.
    pub fn draw_labels(
        &self,
        visible:   &[(u32, TierLevel)],
        positions: &[f32],
        titles:    &[String],
        camera:    &Camera,
    ) {
        // Filter to T1 particles only.
        let t1_particles: Vec<u32> = visible
            .iter()
            .filter(|(_, tier)| *tier == TierLevel::T1)
            .map(|(idx, _)| *idx)
            .collect();

        if t1_particles.is_empty() {
            return;
        }

        // TODO: integrate sugarloaf text rendering when the Bevy graph world
        // surface is available.  Each label should be:
        //   1. A world-space text entity parented to the particle.
        //   2. Positioned at center + radius offset along camera-facing axis.
        //   3. Culled automatically when the particle leaves T1 range.
        //
        // For now, log a trace so the path is exercised in tests.
        let _ = (positions, titles, camera, t1_particles.len());
    }
}

impl Default for T1Pass {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// MSL impostor sphere shader (compiled via aruminium at runtime — step 8+).
// ---------------------------------------------------------------------------

/// MSL fragment shader: analytic sphere SDF impostor at T1 LOD.
/// Receives billboard quad UVs; ray-sphere intersects to compute depth + normal.
pub const T1_IMPOSTOR_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float3 center_world;
    float  radius;
    float  focus;      // φ* luminance driver
};

fragment float4 t1_impostor_frag(
    VertexOut in [[stage_in]],
    constant float4x4 &view_proj [[buffer(0)]])
{
    // Ray-sphere SDF in clip-quad space.
    float2 d = in.uv * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard_fragment();

    // Normal from SDF.
    float3 normal = float3(d, sqrt(1.0 - r2));

    // Diffuse + focus luminance.
    float diffuse = max(0.0, dot(normal, float3(0.577, 0.577, 0.577)));
    float luma    = mix(0.2, 1.0, in.focus) * diffuse;

    return float4(luma, luma, luma, 1.0);
}
"#;

