// HDR bloom composite + warm-preserving AgX filmic tone mapping + linear-to-sRGB output.

#include "common/constants.wgsl"

@group(0) @binding(0) var<uniform> u: FrameUniforms;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
@group(0) @binding(3) var lin_sampler: sampler;
@group(0) @binding(4) var display_tex: texture_storage_2d<rgba8unorm, write>;

fn agx_default_contrast_approx(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2
        - 40.14 * x4 * x
        + 31.96 * x4
        - 6.868 * x2 * x
        + 0.4298 * x2
        + 0.1191 * x
        - 0.00232;
}

fn agx_tonemap_warm(col: vec3<f32>, warmth_param: f32) -> vec3<f32> {
    // AgX inset matrix (column-major in WGSL)
    let agx_mat = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
    );
    // AgX outset matrix (column-major in WGSL)
    let agx_mat_inv = mat3x3<f32>(
        vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
    );

    let min_ev: f32 = -11.2;
    let max_ev: f32 = 4.0;

    var v = agx_mat * max(col, vec3<f32>(1e-6));
    v = clamp(log2(v), vec3<f32>(min_ev), vec3<f32>(max_ev));
    v = (v - min_ev) / (max_ev - min_ev);
    v = agx_default_contrast_approx(v);
    v = max(agx_mat_inv * v, vec3<f32>(0.0));

    // Warm-preserving filmic look: protect deep shadows (luma < 0.05) and golden midtones
    let punch = clamp(warmth_param, 0.5, 1.8);
    let contrasted = pow(max(v, vec3<f32>(0.0)), vec3<f32>(1.05 + 0.12 * punch));
    let luma = dot(contrasted, vec3<f32>(0.2126, 0.7152, 0.0722));
    let sat = 1.0 + 0.18 * punch;
    return clamp(mix(vec3<f32>(luma), contrasted, sat), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let clamped = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = clamped * 12.92;
    let hi = 1.055 * pow(clamped, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, clamped <= vec3<f32>(0.0031308));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u.resolution.x;
    let h = u.resolution.y;
    if gid.x >= w || gid.y >= h {
        return;
    }

    let hdr = textureLoad(hdr_tex, vec2<i32>(gid.xy), 0).rgb;
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(f32(w), f32(h));
    let bloom = textureSampleLevel(bloom_tex, lin_sampler, uv, 0.0).rgb;

    let bloom_w = select(0.0, max(u.render_params.y, 0.0) * 0.48, has_flag(u.flags, FLAG_ENABLE_BLOOM));
    let exposed = (hdr + bloom * bloom_w) * max(u.render_params.x, 0.01);

    let mapped = agx_tonemap_warm(exposed, u.render_params.z);
    let srgb = linear_to_srgb(mapped);
    textureStore(display_tex, vec2<i32>(gid.xy), vec4<f32>(srgb, 1.0));
}
