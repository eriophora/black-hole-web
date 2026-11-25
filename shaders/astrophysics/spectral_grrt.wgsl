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

// Fast branch-light linear-RGB blackbody chromaticity fit (1100K..30000K, zero exp() calls)
fn blackbody_rgb(temp_k: f32) -> vec3<f32> {
    let t = clamp(temp_k, 1000.0, 30000.0) * 0.01;
    var r: f32;
    var g: f32;
    var b: f32;
    if t <= 66.0 {
        r = 1.0;
        g = clamp(0.39008158 * log(max(t, 1.0)) - 0.63184144, 0.0, 1.0);
        if t <= 19.0 {
            b = 0.0;
        } else {
            b = clamp(0.54320679 * log(max(t - 10.0, 1.0)) - 1.19625409, 0.0, 1.0);
        }
    } else {
        let tm = max(t - 60.0, 1.0);
        r = clamp(1.29293619 * pow(tm, -0.13320476), 0.0, 1.0);
        g = clamp(1.12989086 * pow(tm, -0.07551485), 0.0, 1.0);
        b = 1.0;
    }
    // Convert perceptual sRGB fit to linear RGB
    let c = vec3<f32>(r, g, b);
    return max(c * c * (0.72 + 0.28 * c), vec3<f32>(1e-4));
}

fn normalize_luma(c: vec3<f32>) -> vec3<f32> {
    let lum = max(dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)), 1e-4);
    return c / lum;
}

// Pure unit-luminance chromaticity with relativistic g * T_emit shift anchored to the warm Gargantua palette.
fn gargantua_emission_chroma(temp_emit_k: f32, g: f32, uniforms: FrameUniforms) -> vec3<f32> {
    let t_obs = clamp(temp_emit_k * max(g, 0.05), 1100.0, 9500.0);

    let c_copper = vec3<f32>(2.92, 0.526, 0.044);
    let c_gold = vec3<f32>(1.78, 0.855, 0.142);
    let c_ivory = vec3<f32>(1.11, 0.990, 0.756);

    let t_low = smoothstep(1300.0, 3100.0, t_obs);
    let t_high = smoothstep(3100.0, 5600.0, t_obs);
    let pal = mix(mix(c_copper, c_gold, t_low), c_ivory, t_high);

    let bb_chroma = normalize_luma(blackbody_rgb(t_obs));
    let warmth = clamp(uniforms.render_params.z, 0.0, 2.0);
    let mix_amt = clamp(0.78 * warmth, 0.0, 0.94);
    return normalize_luma(mix(bb_chroma, pal, mix_amt));
}

fn gargantua_emission_color(temp_k: f32, g: f32, uniforms: FrameUniforms) -> vec3<f32> {
    let chroma = gargantua_emission_chroma(temp_k, g, uniforms);
    return chroma * pow(clamp(g, 0.08, 2.8), 2.2);
}
