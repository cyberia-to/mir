//! T1 surface + label (§6.2): analytic impostor sphere + world-space text.
//!
//! §6.2: T1 fires when screen-pixel diameter ∈ [S_T2, S_T1) = [8, 40) px.
//! Each T1 particle shows:
//!   • Analytic impostor sphere (fragment shader SDF)
//!   • World-space text label at center + radius offset
//!   • Luminance driven by φ* (focus value)
//!
//! Text rendering via sugarloaf requires a 3D scene surface; `draw_labels`
//! is a stub until that surface is wired in.

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
        let _ = (positions, titles, camera, t1_particles.len());
    }
}

impl Default for T1Pass {
    fn default() -> Self { Self::new() }
}

/// Screen-space label for a T1 edge (§8.4).
/// Rendered at the tube midpoint with billboard orientation.
#[derive(Debug, Clone)]
pub struct EdgeLabel {
    /// Screen-space position (pixels) of the label anchor.
    pub screen_pos:  [f32; 2],
    /// Display text: predicate particle title when present, else CID prefix.
    pub text:        String,
    /// Depth value (clip-space z/w) for sorting.
    pub depth:       f32,
}

/// §8.4: compute screen-space positions and text for visible T1 edge labels.
///
/// `edge_list`  — (from_idx, to_idx) pairs for T1-visible edges.
/// `positions`  — flat n×3 f32 particle positions.
/// `titles`     — per-particle display name / CID prefix.
/// `view_proj`  — column-major 4×4 VP matrix (same as Camera::view_proj).
/// `viewport`   — [width, height] in pixels.
pub fn compute_edge_labels(
    edge_list:  &[(u32, u32)],
    positions:  &[f32],
    titles:     &[String],
    view_proj:  &[[f32; 4]; 4],
    viewport:   [f32; 2],
) -> Vec<EdgeLabel> {
    let [w, h] = viewport;
    edge_list.iter().filter_map(|&(from, to)| {
        let fb = from as usize * 3;
        let tb = to   as usize * 3;
        if fb + 2 >= positions.len() || tb + 2 >= positions.len() { return None; }

        // Midpoint in world space.
        let mid = [
            (positions[fb]   + positions[tb])   * 0.5,
            (positions[fb+1] + positions[tb+1]) * 0.5,
            (positions[fb+2] + positions[tb+2]) * 0.5,
        ];

        // Project to clip space.
        let m = view_proj;
        let cx = m[0][0]*mid[0] + m[1][0]*mid[1] + m[2][0]*mid[2] + m[3][0];
        let cy = m[0][1]*mid[0] + m[1][1]*mid[1] + m[2][1]*mid[2] + m[3][1];
        let cz = m[0][2]*mid[0] + m[1][2]*mid[1] + m[2][2]*mid[2] + m[3][2];
        let cw = m[0][3]*mid[0] + m[1][3]*mid[1] + m[2][3]*mid[2] + m[3][3];

        if cw <= 0.0 { return None; }
        let ndcx = cx / cw;
        let ndcy = cy / cw;
        if ndcx < -1.0 || ndcx > 1.0 || ndcy < -1.0 || ndcy > 1.0 { return None; }

        let sx = (ndcx * 0.5 + 0.5) * w;
        let sy = (1.0 - (ndcy * 0.5 + 0.5)) * h;

        // Label text: "from→to" using titles or CID indices (8-char limit, char-safe).
        let truncate = |s: &str| s.chars().take(8).collect::<String>();
        let from_name = titles.get(from as usize).map(|s| truncate(s)).unwrap_or_else(|| "?".into());
        let to_name   = titles.get(to   as usize).map(|s| truncate(s)).unwrap_or_else(|| "?".into());
        let text = format!("{}→{}", from_name, to_name);

        Some(EdgeLabel { screen_pos: [sx, sy], text, depth: cz / cw })
    }).collect()
}

#[cfg(test)]
mod tests_t1 {
    use super::*;

    #[test]
    fn edge_label_projects_midpoint() {
        let edges = vec![(0u32, 1u32)];
        let positions = vec![
            -100.0f32, 0.0, 0.0,  // particle 0
             100.0f32, 0.0, 0.0,  // particle 1
        ];
        let titles = vec!["alpha".to_string(), "beta".to_string()];
        // Identity view-projection (orthographic).
        let vp: [[f32; 4]; 4] = [
            [0.001, 0.0, 0.0, 0.0],
            [0.0, 0.001, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let labels = compute_edge_labels(&edges, &positions, &titles, &vp, [1280.0, 720.0]);
        assert_eq!(labels.len(), 1, "should produce one label");
        // Midpoint is [0, 0, 0] → NDC [0, 0] → screen [640, 360].
        assert!((labels[0].screen_pos[0] - 640.0).abs() < 1.0, "x={}", labels[0].screen_pos[0]);
        assert!((labels[0].screen_pos[1] - 360.0).abs() < 1.0, "y={}", labels[0].screen_pos[1]);
        assert!(labels[0].text.contains("alpha"), "text={}", labels[0].text);
    }

    #[test]
    fn edge_label_behind_camera_excluded() {
        let edges = vec![(0u32, 1u32)];
        let positions = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 10.0];
        let titles = vec!["a".to_string(), "b".to_string()];
        // Projection that puts everything behind camera (w < 0).
        let vp: [[f32; 4]; 4] = [
            [1.0, 0.0,  0.0, 0.0],
            [0.0, 1.0,  0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, -1.0],  // w = -z - 1 < 0 for z=5
        ];
        let labels = compute_edge_labels(&edges, &positions, &titles, &vp, [1280.0, 720.0]);
        assert_eq!(labels.len(), 0, "behind-camera edge should produce no label");
    }
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

