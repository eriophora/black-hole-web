// Shared uniforms, flags, mathematical constants, and 8-band spectral CIE/Planck helpers.

struct FrameUniforms {
    cam_pos: vec4<f32>,
    cam_fwd: vec4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    resolution: vec2<u32>,
    frame_index: u32,
    flags: u32,
    bh_spin: f32,
    bh_mass: f32,
    fov_y_rad: f32,
    sim_time: f32,
    disk_r_in: f32,
    disk_r_out: f32,
    disk_half_height: f32,
    disk_temp_peak_k: f32,
    planet_pos_radius: vec4<f32>,
    planet_params: vec4<f32>,
    render_params: vec4<f32>,
};

const FLAG_ENABLE_DISK: u32 = 1u;
const FLAG_ENABLE_PLANET: u32 = 2u;
const FLAG_ENABLE_STARS: u32 = 4u;
const FLAG_ENABLE_DOPPLER: u32 = 8u;
const FLAG_ENABLE_BLOOM: u32 = 16u;
const FLAG_ENABLE_PHOTON_SSAA: u32 = 32u;

fn has_flag(flags: u32, bit: u32) -> bool {
    return (flags & bit) != 0u;
}

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const INV_PI: f32 = 0.31830988618;
const R_ESCAPE_TAIL: f32 = 75.0;

// Covector-vector contraction p_mu u^mu.
fn dot4_cov_con(p: vec4<f32>, u: vec4<f32>) -> f32 {
    return dot(p, u);
}

// Interleaved Gradient Noise (Jimenez 2014) for clean step-size jittering.
fn ign_noise(pixel: vec2<u32>, frame_index: u32) -> f32 {
    let p = vec2<f32>(pixel) + 5.588238 * f32(frame_index & 63u);
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// 8 spectral channels spanning 400-750 nm (50 nm bins)
const PLANCK_C2_LO: vec4<f32> = vec4<f32>(35969.425, 31972.822, 28775.54, 26159.582);
const PLANCK_C2_HI: vec4<f32> = vec4<f32>(23979.617, 22135.031, 20553.957, 19183.693);
const PLANCK_SCALE_LO: vec4<f32> = vec4<f32>(270.03593, 149.86076, 88.49430, 54.948);
const PLANCK_SCALE_HI: vec4<f32> = vec4<f32>(35.56386, 23.83393, 16.45415, 11.65474);

const DUST_EXT_LO: vec4<f32> = vec4<f32>(1.4422546, 1.2595817, 1.1158384, 1.0);
const DUST_EXT_HI: vec4<f32> = vec4<f32>(0.9047311, 0.8252431, 0.7576531, 0.6997101);

const CIE_X_LO: vec4<f32> = vec4<f32>(0.0143, 0.3362, 0.0049, 0.4334) * 0.4765;
const CIE_X_HI: vec4<f32> = vec4<f32>(1.0622, 0.2835, 0.0114, 0.0003) * 0.4765;
const CIE_Y_LO: vec4<f32> = vec4<f32>(0.0004, 0.0380, 0.3230, 0.9950) * 0.4765;
const CIE_Y_HI: vec4<f32> = vec4<f32>(0.6310, 0.1070, 0.0041, 0.0001) * 0.4765;
const CIE_Z_LO: vec4<f32> = vec4<f32>(0.0679, 1.7721, 0.2720, 0.0087) * 0.4765;
const CIE_Z_HI: vec4<f32> = vec4<f32>(0.0008, 0.0000, 0.0000, 0.0000) * 0.4765;

fn planck_norm_lo(temp_k: f32) -> vec4<f32> {
    let inv_t = 1.0 / max(temp_k, 40.0);
    let u = clamp(PLANCK_C2_LO * inv_t, vec4<f32>(1e-4), vec4<f32>(55.0));
    return PLANCK_SCALE_LO / max(exp(u) - vec4<f32>(1.0), vec4<f32>(1e-5));
}

fn planck_norm_hi(temp_k: f32) -> vec4<f32> {
    let inv_t = 1.0 / max(temp_k, 40.0);
    let u = clamp(PLANCK_C2_HI * inv_t, vec4<f32>(1e-4), vec4<f32>(55.0));
    return PLANCK_SCALE_HI / max(exp(u) - vec4<f32>(1.0), vec4<f32>(1e-5));
}

fn planck_norm_scalar(lambda_nm: f32, temp_k: f32) -> f32 {
    let t_safe = max(temp_k, 40.0);
    let u = clamp(14387770.0 / (lambda_nm * t_safe), 1e-4, 55.0);
    let ratio = 550.0 / lambda_nm;
    let ratio5 = ratio * ratio * ratio * ratio * ratio;
    return (54.948 * ratio5) / max(exp(u) - 1.0, 1e-5);
}

fn spectral_to_linear_rgb(spec_lo: vec4<f32>, spec_hi: vec4<f32>) -> vec3<f32> {
    let X = dot(spec_lo, CIE_X_LO) + dot(spec_hi, CIE_X_HI);
    let Y = dot(spec_lo, CIE_Y_LO) + dot(spec_hi, CIE_Y_HI);
    let Z = dot(spec_lo, CIE_Z_LO) + dot(spec_hi, CIE_Z_HI);

    let r =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
    let g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
    let b =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
    return max(vec3<f32>(r, g, b), vec3<f32>(0.0));
}
