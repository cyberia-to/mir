//! T2 analytic impostor: one quad per particle, ray-cast in fragment shader.
//! Indirect draw, tile-shaded deferred (aruminium). Step 5.

pub const IMPOSTOR_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float3 world_pos;
    float  radius;
    float3 color;
};

vertex VertexOut impostor_vert(
    uint vid [[vertex_id]],
    device const float3 *positions [[buffer(0)]],
    device const float  *radii     [[buffer(1)]],
    device const float3 *colors    [[buffer(2)]],
    constant float4x4  &mvp        [[buffer(3)]])
{
    // Quad billboard: vid encodes (particle_idx, corner [0..3])
    uint  pid    = vid >> 2;
    uint  corner = vid & 3;
    float3 center = positions[pid];
    float  r      = radii[pid];

    // Screen-space quad offsets.
    float2 offsets[4] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,1) };
    float2 off = offsets[corner] * r;

    VertexOut out;
    out.world_pos = center;
    out.radius    = r;
    out.color     = colors[pid];
    out.position  = mvp * float4(center + float3(off, 0.0), 1.0);
    return out;
}

fragment float4 impostor_frag(VertexOut in [[stage_in]]) {
    // TODO step 5: analytic sphere ray-cast in fragment, depth write, normal.
    return float4(in.color, 1.0);
}
"#;
