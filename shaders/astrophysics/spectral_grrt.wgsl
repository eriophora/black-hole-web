// Relativistic radiative transfer state, Doppler/gravitational shift, and emission color helpers.

#include "common/constants.wgsl"
#include "metrics/kerr_schild.wgsl"

struct RadianceState {
    rgb: vec3<f32>,
    transmittance: f32,
    transmittance_rgb: vec3<f32>,
    weighted_g: f32,
    weight_sum: f32,
    n_crossings: f32,
    prev_z: f32,
};

fn init_radiance_state() -> RadianceState {
    return RadianceState(
        vec3<f32>(0.0),
        1.0,
        vec3<f32>(1.0),
        0.0,
        0.0,
        0.0,
        0.0,
    );
}

fn doppler_factor(
    x: vec4<f32>,
    p: vec4<f32>,
    u_emit: vec4<f32>,
    uniforms: FrameUniforms,
) -> f32 {
    let mass = uniforms.bh_mass;
    let spin_clamped = clamp(uniforms.bh_spin, -0.999, 0.999);
    let a = spin_clamped * mass;
    let r_plus = ks_event_horizon(mass, spin_clamped);
    let enable_doppler = has_flag(uniforms.flags, FLAG_ENABLE_DOPPLER);
    return ks_doppler_factor(x, p, u_emit, mass, a, r_plus, enable_doppler);
}

fn blackbody_rgb(temp_k: f32) -> vec3<f32> {
    let lo = planck_norm_lo(temp_k);
    let hi = planck_norm_hi(temp_k);
    return spectral_to_linear_rgb(lo, hi);
}

fn normalize_luma(c: vec3<f32>) -> vec3<f32> {
    let lum = max(dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)), 1e-4);
    return c / lum;
}

// Pure unit-luminance chromaticity with relativistic g * T_emit shift anchored to the warm Gargantua palette.
fn gargantua_emission_chroma(temp_emit_k: f32, g: f32, uniforms: FrameUniforms) -> vec3<f32> {
    let t_obs = clamp(temp_emit_k * max(g, 0.05), 1100.0, 9500.0);
    let bb_chroma = normalize_luma(blackbody_rgb(t_obs));

    let c_copper = normalize_luma(vec3<f32>(1.0, 0.18, 0.015));
    let c_gold = normalize_luma(vec3<f32>(1.0, 0.48, 0.08));
    let c_ivory = normalize_luma(vec3<f32>(1.0, 0.89, 0.68));

    let t_low = smoothstep(1300.0, 3100.0, t_obs);
    let t_high = smoothstep(3100.0, 5600.0, t_obs);
    let pal = mix(mix(c_copper, c_gold, t_low), c_ivory, t_high);

    let warmth = clamp(uniforms.render_params.z, 0.0, 2.0);
    let mix_amt = clamp(0.78 * warmth, 0.0, 0.94);
    return normalize_luma(mix(bb_chroma, pal, mix_amt));
}

fn gargantua_emission_color(temp_k: f32, g: f32, uniforms: FrameUniforms) -> vec3<f32> {
    let chroma = gargantua_emission_chroma(temp_k, g, uniforms);
    return chroma * pow(clamp(g, 0.08, 2.8), 2.2);
}
