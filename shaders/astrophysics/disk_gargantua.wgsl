// Two-component Gargantua self-occluding volumetric accretion disk:
// (A) Hot incandescent plasma with multi-scale ln(r) ringlets x seam-free sheared 3D filaments
// (B) Cool high-opacity wavelength-selective dust lanes with 2-3 tap inward radial self-shadowing

#include "common/constants.wgsl"
#include "astrophysics/spectral_grrt.wgsl"

@group(0) @binding(4) var disk_noise_tex: texture_3d<f32>;
@group(0) @binding(5) var disk_noise_samp: sampler;

fn ks_isco_radius(mass: f32, a_star: f32) -> f32 {
    let a_s = clamp(a_star, -0.999, 0.999);
    let z1 = 1.0 + pow(max(1.0 - a_s * a_s, 0.0), 1.0 / 3.0)
        * (pow(1.0 + a_s, 1.0 / 3.0) + pow(max(1.0 - a_s, 0.0), 1.0 / 3.0));
    let z2 = sqrt(3.0 * a_s * a_s + z1 * z1);
    let sgn = select(-1.0, 1.0, a_s >= 0.0);
    return mass * (3.0 + z2 - sgn * sqrt(max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0)));
}

// Prograde Kerr Keplerian + smooth plunging-region 4-velocity using closed-form Kerr-Schild norm.
fn ks_disk_four_velocity_fast(
    x: vec4<f32>,
    mass: f32,
    a: f32,
    r_plus: f32,
    r_isco: f32,
    omega_isco: f32,
) -> vec4<f32> {
    let rho2 = max(x.y * x.y + x.z * x.z, 1e-8);
    let rho = sqrt(rho2);
    let sqrt_m = sqrt(max(mass, 1e-4));
    let omega_k = sqrt_m / max(rho * sqrt(rho) + a * sqrt_m, 1e-4);

    let s = ks_sample(x, mass, a);
    let a2 = a * a;
    let r2_plus_a2 = s.r * s.r + a2;
    let denom_zamo = max(r2_plus_a2 + (s.f * a2 * rho2) / max(r2_plus_a2, 1e-12), 1e-12);
    let omega_zamo = (s.f * a) / denom_zamo;

    let t_plunge = smoothstep(r_plus * 1.02, r_isco, rho);
    let omega = select(mix(omega_zamo, omega_isco, t_plunge), omega_k, rho >= r_isco);

    let lu = 1.0 - (omega * a * rho2) / max(r2_plus_a2, 1e-12);
    let norm2 = 1.0 - omega * omega * rho2 - s.f * lu * lu;
    if norm2 <= 1e-5 {
        return ks_local_rest_u(x, mass, a, r_plus);
    }
    return vec4<f32>(1.0, -omega * x.z, omega * x.y, 0.0) * inverseSqrt(norm2);
}

fn ks_disk_four_velocity(x: vec4<f32>, mass: f32, a: f32, r_plus: f32) -> vec4<f32> {
    let a_star = clamp(a / max(mass, 1e-4), -0.999, 0.999);
    let r_isco = ks_isco_radius(mass, a_star);
    let sqrt_m = sqrt(max(mass, 1e-4));
    let omega_isco = sqrt_m / max(pow(r_isco, 1.5) + a * sqrt_m, 1e-4);
    return ks_disk_four_velocity_fast(x, mass, a, r_plus, r_isco, omega_isco);
}

fn eval_disk_step_fast(
    x: vec4<f32>,
    p: vec4<f32>,
    z_a: f32,
    z_b: f32,
    ds: f32,
    n_half: f32,
    mass: f32,
    a: f32,
    r_plus: f32,
    r_isco: f32,
    omega_isco: f32,
    uniforms: FrameUniforms,
    state: ptr<function, RadianceState>,
) {
    if ds <= 1e-6 || (*state).transmittance < 1e-3 {
        return;
    }

    let r_in = uniforms.disk_r_in;
    let r_out = uniforms.disk_r_out;
    let pos = x.yzw;
    let rho = length(pos.xy);
    if rho < r_in * 0.94 || rho > r_out * 1.02 {
        return;
    }

    let r_span = max(r_out - r_in, 1e-3);
    let r_norm = clamp((rho - r_in) / r_span, 0.0, 1.0);
    let h_local = uniforms.disk_half_height * mix(0.28, 1.30, pow(r_norm, 0.70));
    let inv_h = 1.0 / max(h_local, 1e-4);
    let z_norm = pos.z * inv_h;
    let za_n = z_a * inv_h;
    let zb_n = z_b * inv_h;

    if min(abs(za_n), min(abs(z_norm), abs(zb_n))) > 2.5 {
        return;
    }

    // Track equatorial crossings for unified photon-ring weighting
    if (*state).prev_z * pos.z <= 0.0 && rho >= r_in * 0.95 && rho <= r_out {
        (*state).n_crossings += 1.0;
    }

    // Crisp inner disk border preserves the dark plunging gap between the accretion disk and inner photon ring
    let outer_taper = exp(-pow(max(rho - (r_in + 1.1 * mass), 0.0) / (8.4 * mass), 1.45));
    let rad_env = smoothstep(r_in * 0.97, r_in + 0.70, rho)
        * smoothstep(r_out * 0.97, r_out * 0.56, rho)
        * outer_taper;
    // 3-point Simpson vertical Gaussian quadrature with C1 zero-tail window eliminates slab-edge shelves at max exposure
    let va = exp(-1.12 * za_n * za_n) * smoothstep(2.45, 1.45, abs(za_n));
    let vm = exp(-1.12 * z_norm * z_norm) * smoothstep(2.45, 1.45, abs(z_norm));
    let vb = exp(-1.12 * zb_n * zb_n) * smoothstep(2.45, 1.45, abs(zb_n));
    let v_env = (1.0 / 6.0) * (va + 4.0 * vm + vb);
    if rad_env * v_env < 6e-4 {
        return;
    }

    let ln_r = log(max(rho, 0.5) / max(r_in, 0.5));
    let phi = atan2(pos.y, pos.x);
    let omega_shear = 1.0 / (rho * sqrt(max(rho, 1.0)) + abs(a));
    let theta = phi - omega_shear * uniforms.sim_time * 0.35 - 4.2 * ln_r;

    let uvw = vec3<f32>(
        3.2 * ln_r,
        0.62 * cos(theta),
        0.62 * sin(theta) + 0.16 * z_norm,
    );
    let n1 = textureSampleLevel(disk_noise_tex, disk_noise_samp, uvw, 0.0);
    let uvw2 = uvw * vec3<f32>(2.5, 2.5, 1.9) + vec3<f32>(0.37, 0.61, 0.19);
    let n2_raw = textureSampleLevel(disk_noise_tex, disk_noise_samp, uvw2, 0.0);

    // Analytic radial pixel-footprint band-limiting (lens_mag == 1.0 in direct view n_half <= 0.85)
    let pixel_ang = (2.0 * tan(0.5 * uniforms.fov_y_rad)) / f32(max(uniforms.resolution.y, 1u));
    let cam_dist = max(length(pos - uniforms.cam_pos.xyz), 1.0);
    let lens_mag = exp(min(PI * max(n_half - 0.85, 0.0), 6.0));
    let sigma_lnr = (cam_dist * pixel_ang * lens_mag) / max(rho, 0.5);

    let nyq_macro = exp(-0.5 * pow((14.0 * sigma_lnr) / 1.1, 2.0));
    let nyq_mid   = exp(-0.5 * pow((38.0 * sigma_lnr) / 1.1, 2.0));
    let nyq_fine  = exp(-0.5 * pow((86.0 * sigma_lnr) / 1.1, 2.0));
    let nyq_n2    = exp(-0.5 * pow((26.0 * sigma_lnr), 2.0));
    let n2 = mix(vec4<f32>(0.5), n2_raw, nyq_n2);

    // Multi-scale concentric ln(r) ringlets broken organically by sheared turbulence (zero phi seam).
    // Energy-preserving band-limiting ensures demagnified n>=2 photon sub-rings keep true integrated disk radiance.
    let inner_arch_sharp = mix(3.4, 2.2, smoothstep(r_in, r_in + 5.0, rho));
    let band_macro = 0.5 + 0.5 * nyq_macro * sin(ln_r * 14.0 + inner_arch_sharp * (n1.r - 0.5));
    let band_mid = 0.5 + 0.5 * nyq_mid * sin(ln_r * 38.0 + 3.2 * (n1.g - 0.5));
    let band_fine = 0.5 + 0.5 * nyq_fine * sin(ln_r * 86.0 + 3.6 * (n2.g - 0.5));
    let m_macro = mix(0.44, pow(band_macro, 1.55), nyq_macro);
    let m_mid   = mix(0.48, pow(band_mid, 1.18), nyq_mid);
    let ringlets = mix(0.14, 1.0, m_macro)
        * mix(0.36, 1.0, m_mid)
        * mix(0.58, 1.0, band_fine);

    // Keep physical n=1, n=2 geodesic photon-ring passes luminous; attenuate only dust opacity on higher passes
    let dust_sub_atten = 1.0 - 0.65 * smoothstep(1.45, 2.25, n_half);

    // Feathery wispy outer edge modulation so strands dissolve gracefully into space
    let outer_wisp = mix(1.0, smoothstep(0.18, 0.78, n1.g * 0.55 + n2.g * 0.45), smoothstep(0.42, 0.92, r_norm));

    // Component A: Hot incandescent plasma density with crisp fine wispy filaments (reuses n2.a 64-period channel)
    let inner_flare = 0.68 + 5.2 * exp(-pow((rho - (r_in + 0.65)) / 1.75, 2.0));
    let fine_wisps = 0.58 + 0.90 * pow(clamp(1.0 - abs(2.0 * n2.a - 1.0), 0.0, 1.0), 1.5);
    let fil_plasma = pow(mix(n1.g, n2.g, 0.48), 1.95) * (0.44 + 1.16 * pow(n2.a, 1.25)) * fine_wisps;
    let rho_plasma = rad_env * outer_wisp * v_env * ringlets * fil_plasma * inner_flare;

    // Component B: Cool high-opacity self-occluding dust lanes (sublimates right at scorching inner rim)
    let dust_rad_env = smoothstep(r_in + 0.55, r_in + 2.4, rho)
        * smoothstep(r_out, r_out * 0.58, rho)
        * mix(1.0, 0.55, r_norm);
    let dza = exp(-1.25 * za_n * za_n) * smoothstep(2.45, 1.45, abs(za_n));
    let dzm = exp(-1.25 * z_norm * z_norm) * smoothstep(2.45, 1.45, abs(z_norm));
    let dzb = exp(-1.25 * zb_n * zb_n) * smoothstep(2.45, 1.45, abs(zb_n));
    let dust_z_env = (1.0 / 6.0) * (dza + 4.0 * dzm + dzb);
    let dust_lane = smoothstep(0.20, 0.68, (1.0 - band_macro * 0.72) * 0.48 + n1.b * 0.58 + (n2.b - 0.5) * 0.20);
    let dust_clump = pow(smoothstep(0.16, 0.76, n1.b * 0.56 + n2.r * 0.44), 1.18);
    let rho_dust = dust_rad_env * outer_wisp * dust_z_env * dust_lane * dust_clump * 1.95 * dust_sub_atten;

    if rho_plasma < 1e-4 && rho_dust < 0.01 {
        return;
    }

    // Relativistic Doppler & gravitational shift
    let u_emit = ks_disk_four_velocity_fast(x, mass, a, r_plus, r_isco, omega_isco);
    let g = doppler_factor(x, p, u_emit, uniforms);
    let beam = pow(clamp(g, 0.08, 2.8), 3.0);

    var j_plasma = vec3<f32>(0.0);
    var alpha_plasma = 0.0;
    if rho_plasma >= 1e-4 {
        let t_rad = pow(r_in / max(rho, r_in), 0.76);
        let t_peak = clamp(uniforms.disk_temp_peak_k * 0.82, 4200.0, 5800.0);
        let t_emit = mix(1450.0, t_peak, t_rad) * mix(0.78, 1.32, n1.g * 0.72 + n2.g * 0.28);
        let chroma = gargantua_emission_chroma(t_emit, g, uniforms);

        let plasma_gain: f32 = 3.5;
        j_plasma = chroma * beam * rho_plasma * plasma_gain;
        alpha_plasma = 0.20 * rho_plasma;
    }

    var j_dust = vec3<f32>(0.0);
    var alpha_dust_rgb = vec3<f32>(0.0);
    if rho_dust > 0.01 {
        // Fast single inward radial core self-shadowing tap (cuts 3D texture reads in half on WebGPU)
        let d_ln = log(max(rho - 1.05, r_in) / max(r_in, 0.5)) - ln_r;
        let occ = textureSampleLevel(
            disk_noise_tex,
            disk_noise_samp,
            uvw + vec3<f32>(3.2 * d_ln, 0.0, -0.15 * z_norm),
            0.0,
        ).b;
        let shadow_inward = exp(-4.1 * occ);
        alpha_dust_rgb = 4.9 * rho_dust * vec3<f32>(0.70, 1.0, 1.48);
        j_dust = (0.020 * shadow_inward * pow(clamp(g, 0.15, 2.0), 1.5))
            * vec3<f32>(0.90, 0.28, 0.04)
            * rho_dust;
    }

    // Exact slab radiative transfer over spatial segment ds
    let alpha_rgb = vec3<f32>(alpha_plasma) + alpha_dust_rgb;
    let tau_rgb = alpha_rgb * ds;
    let atten_rgb = exp(-tau_rgb);
    let scatter_rgb = (j_plasma + j_dust) * (vec3<f32>(1.0) - atten_rgb) / max(alpha_rgb, vec3<f32>(1e-5));

    (*state).rgb += (*state).transmittance_rgb * scatter_rgb;
    (*state).transmittance_rgb *= atten_rgb;
    (*state).transmittance = dot((*state).transmittance_rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    (*state).weighted_g += g * rho_plasma * ds;
    (*state).weight_sum += rho_plasma * ds;
}

fn eval_disk_step(
    x: vec4<f32>,
    p: vec4<f32>,
    r: f32,
    ds: f32,
    uniforms: FrameUniforms,
    state: ptr<function, RadianceState>,
) {
    let _unused_r = r;
    let mass = uniforms.bh_mass;
    let spin_clamped = clamp(uniforms.bh_spin, -0.999, 0.999);
    let a = spin_clamped * mass;
    let r_plus = ks_event_horizon(mass, spin_clamped);
    let r_isco = ks_isco_radius(mass, spin_clamped);
    let sqrt_m = sqrt(max(mass, 1e-4));
    let omega_isco = sqrt_m / max(pow(r_isco, 1.5) + a * sqrt_m, 1e-4);
    eval_disk_step_fast(x, p, x.w, x.w, ds, 0.0, mass, a, r_plus, r_isco, omega_isco, uniforms, state);
}
