// 9-tap tent filter HDR bloom upsample pass.

struct BloomParams {
    src_texel: vec2<f32>,
    dst_size: vec2<u32>,
    is_first_pass: u32,
    threshold: f32,
    knee: f32,
    scatter: f32,
};

@group(0) @binding(0) var<uniform> params: BloomParams;
@group(0) @binding(1) var low_tex: texture_2d<f32>;
@group(0) @binding(2) var base_tex: texture_2d<f32>;
@group(0) @binding(3) var lin_sampler: sampler;
@group(0) @binding(4) var dst_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.dst_size.x || gid.y >= params.dst_size.y {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(params.dst_size);
    let d = params.src_texel * params.scatter;

    let s0 = textureSampleLevel(low_tex, lin_sampler, uv + vec2<f32>(-d.x, -d.y), 0.0).rgb;
    let s1 = textureSampleLevel(low_tex, lin_sampler, uv + vec2<f32>( 0.0, -d.y), 0.0).rgb * 2.0;
    let s2 = textureSampleLevel(low_tex, lin_sampler, uv + vec2<f32>( d.x, -d.y), 0.0).rgb;

    let s3 = textureSampleLevel(low_tex, lin_sampler, uv + vec2<f32>(-d.x,  0.0), 0.0).rgb * 2.0;
    let s4 = textureSampleLevel(low_tex, lin_sampler, uv, 0.0).rgb * 4.0;
    let s5 = textureSampleLevel(low_tex, lin_sampler, uv + vec2<f32>( d.x,  0.0), 0.0).rgb * 2.0;

    let s6 = textureSampleLevel(low_tex, lin_sampler, uv + vec2<f32>(-d.x,  d.y), 0.0).rgb;
    let s7 = textureSampleLevel(low_tex, lin_sampler, uv + vec2<f32>( 0.0,  d.y), 0.0).rgb * 2.0;
    let s8 = textureSampleLevel(low_tex, lin_sampler, uv + vec2<f32>( d.x,  d.y), 0.0).rgb;

    let tent = (s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8) * (1.0 / 16.0);
    let base = textureLoad(base_tex, vec2<i32>(gid.xy), 0).rgb;
    let out_col = max(base + tent * 0.65, vec3<f32>(0.0));

    textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(out_col, 1.0));
}
