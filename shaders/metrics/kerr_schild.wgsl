// Horizon-penetrating Ingoing Cartesian Kerr-Schild (CKS) spacetime metric.
// Spin is aligned along +z. Coordinates x = (t, x, y, z), signature (-, +, +, +).

#include "common/constants.wgsl"

struct KerrSample {
    r: f32,
    f: f32,
    l_cov: vec4<f32>, // (1, lx, ly, lz)
};

struct GeodesicRhs {
    dx: vec4<f32>,
    dp: vec4<f32>,
    r: f32,
};

fn ks_bl_radius(pos: vec3<f32>, a: f32) -> f32 {
    let a2 = a * a;
    let r_euclid2 = dot(pos, pos);
    let b = r_euclid2 - a2;
    let disc = sqrt(max(b * b + 4.0 * a2 * pos.z * pos.z, 0.0));
    var r2: f32;
    if b >= 0.0 {
        r2 = 0.5 * (b + disc);
    } else {
        r2 = (2.0 * a2 * pos.z * pos.z) / max(disc - b, 1e-12);
    }
    return sqrt(max(r2, 1e-10));
}

fn ks_event_horizon(mass: f32, a_star: f32) -> f32 {
    let a_s = clamp(a_star, -0.999, 0.999);
    return mass * (1.0 + sqrt(max(1.0 - a_s * a_s, 0.0)));
}

fn ks_inner_horizon(mass: f32, a_star: f32) -> f32 {
    let a_s = clamp(a_star, -0.999, 0.999);
    return mass * (1.0 - sqrt(max(1.0 - a_s * a_s, 0.0)));
}

fn ks_photon_ring_radii(mass: f32, a_star: f32) -> vec2<f32> {
    let a_abs = clamp(abs(a_star), 0.0, 0.999);
    let r_pro = 2.0 * mass * (1.0 + cos((2.0 / 3.0) * acos(-a_abs)));
    let r_ret = 2.0 * mass * (1.0 + cos((2.0 / 3.0) * acos(a_abs)));
    return vec2<f32>(r_pro, r_ret);
}

fn ks_sample(x: vec4<f32>, mass: f32, a: f32) -> KerrSample {
    let pos = x.yzw;
    let r = ks_bl_radius(pos, a);
    let r2 = r * r;
    let a2 = a * a;
    let sigma_ks = max(r2 * r2 + a2 * pos.z * pos.z, 1e-12);
    let denom_a = r2 + a2;
    let f = (2.0 * mass * r2 * r) / sigma_ks;
    let lx = (r * pos.x + a * pos.y) / denom_a;
    let ly = (r * pos.y - a * pos.x) / denom_a;
    let lz = pos.z / r;
    return KerrSample(r, f, vec4<f32>(1.0, lx, ly, lz));
}

fn ks_metric_cov(x: vec4<f32>, mass: f32, a: f32) -> mat4x4<f32> {
    let s = ks_sample(x, mass, a);
    let l = s.l_cov;
    let c0 = vec4<f32>(-1.0, 0.0, 0.0, 0.0) + s.f * l.x * l;
    let c1 = vec4<f32>(0.0, 1.0, 0.0, 0.0) + s.f * l.y * l;
    let c2 = vec4<f32>(0.0, 0.0, 1.0, 0.0) + s.f * l.z * l;
    let c3 = vec4<f32>(0.0, 0.0, 0.0, 1.0) + s.f * l.w * l;
    return mat4x4<f32>(c0, c1, c2, c3);
}

fn ks_metric_inv(x: vec4<f32>, mass: f32, a: f32) -> mat4x4<f32> {
    let s = ks_sample(x, mass, a);
    let lc = vec4<f32>(-s.l_cov.x, s.l_cov.y, s.l_cov.z, s.l_cov.w);
    let c0 = vec4<f32>(-1.0, 0.0, 0.0, 0.0) - s.f * lc.x * lc;
    let c1 = vec4<f32>(0.0, 1.0, 0.0, 0.0) - s.f * lc.y * lc;
    let c2 = vec4<f32>(0.0, 0.0, 1.0, 0.0) - s.f * lc.z * lc;
    let c3 = vec4<f32>(0.0, 0.0, 0.0, 1.0) - s.f * lc.w * lc;
    return mat4x4<f32>(c0, c1, c2, c3);
}

fn ks_hamiltonian(x: vec4<f32>, p: vec4<f32>, mass: f32, a: f32) -> f32 {
    let sample = ks_sample(x, mass, a);
    let s = dot(vec4<f32>(-1.0, sample.l_cov.yzw), p);
    return 0.5 * (-p.x * p.x + dot(p.yzw, p.yzw) - sample.f * s * s);
}

fn ks_rhs(x: vec4<f32>, p: vec4<f32>, mass: f32, a: f32) -> GeodesicRhs {
    let pos = x.yzw;
    let r = ks_bl_radius(pos, a);
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r2 * r2;
    let a2 = a * a;
    let sigma_ks = max(r4 + a2 * pos.z * pos.z, 1e-12);
    let denom_a = r2 + a2;
    let inv_r = 1.0 / r;
    let inv_denom_a = 1.0 / denom_a;
    let inv_sigma = 1.0 / sigma_ks;
    let f = (2.0 * mass * r3) * inv_sigma;

    let lx = (r * pos.x + a * pos.y) * inv_denom_a;
    let ly = (r * pos.y - a * pos.x) * inv_denom_a;
    let lz = pos.z * inv_r;
    let l_con = vec4<f32>(-1.0, lx, ly, lz);

    let s = dot(l_con, p);
    let p_eta = vec4<f32>(-p.x, p.y, p.z, p.w);
    let dx = p_eta - (f * s) * l_con;

    let dr = vec3<f32>(
        (r3 * pos.x) * inv_sigma,
        (r3 * pos.y) * inv_sigma,
        (r * pos.z * denom_a) * inv_sigma,
    );

    let df_coeff = (2.0 * mass * r2) * (inv_sigma * inv_sigma);
    let poly = 3.0 * a2 * pos.z * pos.z - r4;
    let df = vec3<f32>(
        df_coeff * poly * dr.x,
        df_coeff * poly * dr.y,
        df_coeff * (poly * dr.z - 2.0 * a2 * r * pos.z),
    );

    let dl_dx = vec3<f32>(
        (r + (pos.x - 2.0 * r * lx) * dr.x) * inv_denom_a,
        (-a + (pos.y - 2.0 * r * ly) * dr.x) * inv_denom_a,
        (-lz * dr.x) * inv_r,
    );
    let dl_dy = vec3<f32>(
        (a + (pos.x - 2.0 * r * lx) * dr.y) * inv_denom_a,
        (r + (pos.y - 2.0 * r * ly) * dr.y) * inv_denom_a,
        (-lz * dr.y) * inv_r,
    );
    let dl_dz = vec3<f32>(
        ((pos.x - 2.0 * r * lx) * dr.z) * inv_denom_a,
        ((pos.y - 2.0 * r * ly) * dr.z) * inv_denom_a,
        (1.0 - lz * dr.z) * inv_r,
    );

    let p_sp = p.yzw;
    let fs = f * s;
    let half_s2 = 0.5 * s * s;
    let dp = vec4<f32>(
        0.0,
        half_s2 * df.x + fs * dot(p_sp, dl_dx),
        half_s2 * df.y + fs * dot(p_sp, dl_dy),
        half_s2 * df.z + fs * dot(p_sp, dl_dz),
    );

    return GeodesicRhs(dx, dp, r);
}

// Smooth C1 reference 4-velocity: ZAMO outside 1.008 r_+, horizon-regular Eulerian normal inside 1.002 r_+.
fn ks_local_rest_u(x: vec4<f32>, mass: f32, a: f32, r_plus: f32) -> vec4<f32> {
    let s = ks_sample(x, mass, a);
    let w = sqrt(1.0 + s.f);
    let inv_w = 1.0 / w;
    let n_eul = vec4<f32>(
        w,
        -s.f * inv_w * s.l_cov.y,
        -s.f * inv_w * s.l_cov.z,
        -s.f * inv_w * s.l_cov.w,
    );

    if s.r <= 1.002 * r_plus {
        return n_eul;
    }

    let g = ks_metric_cov(x, mass, a);
    let a2 = a * a;
    let r2_plus_a2 = s.r * s.r + a2;
    let rho2 = x.y * x.y + x.z * x.z;
    let denom = max(r2_plus_a2 + (s.f * a2 * rho2) / max(r2_plus_a2, 1e-12), 1e-12);
    let omega_zamo = (s.f * a) / denom;
    let u_tilde = vec4<f32>(1.0, -omega_zamo * x.z, omega_zamo * x.y, 0.0);
    let norm2 = -dot(u_tilde, g * u_tilde);
    if norm2 <= 1e-10 {
        return n_eul;
    }
    let u_zamo = u_tilde * inverseSqrt(norm2);
    if s.r >= 1.008 * r_plus {
        return u_zamo;
    }

    let t = smoothstep(1.002 * r_plus, 1.008 * r_plus, s.r);
    let u_mix = mix(n_eul, u_zamo, t);
    let mix_norm2 = max(-dot(u_mix, g * u_mix), 1e-12);
    return u_mix * inverseSqrt(mix_norm2);
}

fn ks_build_observer_tetrad(
    x: vec4<f32>,
    right: vec3<f32>,
    up: vec3<f32>,
    fwd: vec3<f32>,
    mass: f32,
    a: f32,
    r_plus: f32,
) -> array<vec4<f32>, 4> {
    let s = ks_sample(x, mass, a);
    let g = ks_metric_cov(x, mass, a);
    let inv_w = 1.0 / sqrt(1.0 + s.f);
    let l_sp = s.l_cov.yzw;

    let v1 = right + (inv_w - 1.0) * dot(l_sp, right) * l_sp;
    let v2 = up + (inv_w - 1.0) * dot(l_sp, up) * l_sp;
    let v3 = fwd + (inv_w - 1.0) * dot(l_sp, fwd) * l_sp;

    var tet: array<vec4<f32>, 4>;
    tet[0] = ks_local_rest_u(x, mass, a, r_plus);
    tet[1] = vec4<f32>(0.0, v1);
    tet[2] = vec4<f32>(0.0, v2);
    tet[3] = vec4<f32>(0.0, v3);

    // Normalize timelike e0 (<e0, e0> = -1)
    let norm0 = sqrt(max(-dot(tet[0], g * tet[0]), 1e-12));
    tet[0] = tet[0] / norm0;

    // Two-pass Lorentzian Modified Gram-Schmidt for e1, e2, e3
    for (var i = 1u; i < 4u; i = i + 1u) {
        var v = tet[i];
        v += dot(tet[0], g * v) * tet[0];
        for (var j = 1u; j < i; j = j + 1u) {
            v -= dot(tet[j], g * v) * tet[j];
        }
        v += dot(tet[0], g * v) * tet[0];
        for (var j = 1u; j < i; j = j + 1u) {
            v -= dot(tet[j], g * v) * tet[j];
        }
        let norm_i = sqrt(max(dot(v, g * v), 1e-12));
        tet[i] = v / norm_i;
    }

    return tet;
}

fn ks_init_null_momentum(
    x: vec4<f32>,
    tetrad: array<vec4<f32>, 4>,
    n_dir: vec3<f32>,
    mass: f32,
    a: f32,
) -> vec4<f32> {
    let k_init = -tetrad[0] + n_dir.x * tetrad[1] + n_dir.y * tetrad[2] + n_dir.z * tetrad[3];
    return ks_metric_cov(x, mass, a) * k_init;
}

// Past-directed ray initialization satisfies p_mu e0^mu = +1, so E_emit = +dot4_cov_con(p, u_emit) > 0.
fn ks_doppler_factor(
    x: vec4<f32>,
    p: vec4<f32>,
    u_emit: vec4<f32>,
    mass: f32,
    a: f32,
    r_plus: f32,
    enable_doppler: bool,
) -> f32 {
    if enable_doppler {
        let denom_emit = dot4_cov_con(p, u_emit);
        if denom_emit <= 1e-6 {
            return 0.0;
        }
        return clamp(1.0 / denom_emit, 0.0, 40.0);
    }
    let u_rest = ks_local_rest_u(x, mass, a, r_plus);
    let denom_rest = dot4_cov_con(p, u_rest);
    if denom_rest <= 1e-6 {
        return 0.01;
    }
    return clamp(1.0 / denom_rest, 0.01, 40.0);
}
