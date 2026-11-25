struct BlitParams {
    swapchain_is_srgb: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var display_tex: texture_2d<f32>;
@group(0) @binding(1) var display_sampler: sampler;
@group(0) @binding(2) var<uniform> blit_params: BlitParams;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    out.uv = vec2<f32>(x, y);
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow(max((c + 0.055) / 1.055, vec3<f32>(0.0)), vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let col = textureSample(display_tex, display_sampler, in.uv).rgb;
    if blit_params.swapchain_is_srgb != 0u {
        return vec4<f32>(srgb_to_linear(col), 1.0);
    }
    return vec4<f32>(col, 1.0);
}
