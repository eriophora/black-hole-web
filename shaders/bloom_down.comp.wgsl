// 13-tap Jimenez HDR bloom downsample pass.
// Pass 0 applies firefly clamp, Karis averaging across 5 overlapping 2x2 sub-boxes, and soft-knee threshold.

struct BloomParams {
    src_texel: vec2<f32>,
    dst_size: vec2<u32>,
    is_first_pass: u32,
    threshold: f32,
    knee: f32,
    scatter: f32,
};

@group(0) @binding(0) var<uniform> params: BloomParams;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var lin_sampler: sampler;
@group(0) @binding(3) var dst_tex: texture_storage_2d<rgba16float, write>;

fn luma_rec709(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn soft_knee_threshold(c: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    let clamped = min(max(c, vec3<f32>(0.0)), vec3<f32>(24.0));
    let br = max(clamped.r, max(clamped.g, clamped.b));
    let rq = clamp(br - threshold + knee, 0.0, 2.0 * knee);
    let soft = (rq * rq) / max(4.0 * knee, 1e-4);
    let contrib = max(soft, br - threshold) / max(br, 1e-5);
    return clamped * max(contrib, 0.0);
}

fn sample_tap(uv: vec2<f32>) -> vec3<f32> {
    let c = textureSampleLevel(src_tex, lin_sampler, uv, 0.0).rgb;
    if params.is_first_pass != 0u {
        return soft_knee_threshold(c, params.threshold, params.knee);
    }
    return max(c, vec3<f32>(0.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.dst_size.x || gid.y >= params.dst_size.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(params.dst_size);
    let tx = params.src_texel;

    // 13-tap Jimenez filter layout
    let a = sample_tap(uv + vec2<f32>(-2.0, -2.0) * tx);
    let b = sample_tap(uv + vec2<f32>( 0.0, -2.0) * tx);
    let c = sample_tap(uv + vec2<f32>( 2.0, -2.0) * tx);
    let d = sample_tap(uv + vec2<f32>(-1.0, -1.0) * tx);
    let e = sample_tap(uv + vec2<f32>( 1.0, -1.0) * tx);
    let f = sample_tap(uv + vec2<f32>(-2.0,  0.0) * tx);
    let g = sample_tap(uv);
    let h = sample_tap(uv + vec2<f32>( 2.0,  0.0) * tx);
    let i = sample_tap(uv + vec2<f32>(-1.0,  1.0) * tx);
    let j = sample_tap(uv + vec2<f32>( 1.0,  1.0) * tx);
    let k = sample_tap(uv + vec2<f32>(-2.0,  2.0) * tx);
    let l = sample_tap(uv + vec2<f32>( 0.0,  2.0) * tx);
    let m = sample_tap(uv + vec2<f32>( 2.0,  2.0) * tx);

    var out_col: vec3<f32>;
    if params.is_first_pass != 0u {
        let g0 = 0.25 * (d + e + i + j);
        let g1 = 0.25 * (a + b + f + g);
        let g2 = 0.25 * (b + c + g + h);
        let g3 = 0.25 * (f + g + k + l);
        let g4 = 0.25 * (g + h + l + m);

        let w0 = 0.5 / (1.0 + luma_rec709(g0));
        let w1 = 0.125 / (1.0 + luma_rec709(g1));
        let w2 = 0.125 / (1.0 + luma_rec709(g2));
        let w3 = 0.125 / (1.0 + luma_rec709(g3));
        let w4 = 0.125 / (1.0 + luma_rec709(g4));
        let w_sum = max(w0 + w1 + w2 + w3 + w4, 1e-5);
        out_col = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / w_sum;
    } else {
        out_col = 0.5 * (0.25 * (d + e + i + j))
            + 0.125 * (0.25 * (a + b + f + g))
            + 0.125 * (0.25 * (b + c + g + h))
            + 0.125 * (0.25 * (f + g + k + l))
            + 0.125 * (0.25 * (g + h + l + m));
    }

    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(max(out_col, vec3<f32>(0.0)), 1.0));
}
