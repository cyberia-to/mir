//! T3 Gaussian splat: 3D anisotropic Gaussian per Kerbl et al. 2023.
//! Front-to-back alpha blending. Step 6.

pub const SPLAT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct SplatParams {
    float3 mean;
    float  opacity;
    float3 color;
    float  _pad;
    // 3×3 covariance (packed 6 floats, upper-triangular)
    float  cov[6];
};

kernel void splat_sort(
    device const SplatParams *splats    [[buffer(0)]],
    device const float4x4    &view_proj [[buffer(1)]],
    device       uint        *order     [[buffer(2)]],
    device       float       *depths    [[buffer(3)]],
    constant     uint        &n         [[buffer(4)]],
    uint                      gid       [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float4 clip = view_proj * float4(splats[gid].mean, 1.0);
    depths[gid] = clip.z / clip.w;
    order[gid]  = gid;
    // TODO step 6: GPU radix sort by depth (front-to-back), then rasterize.
}
"#;
