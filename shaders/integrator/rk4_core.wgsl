// Core curvature-adaptive RK4 Kerr-Schild geodesic integrator for r < 75 M
// paired with 2-point Gauss-Legendre rational weak-field asymptotic escape tail at r >= 75 M.

#include "common/constants.wgsl"
#include "metrics/kerr_schild.wgsl"

struct Rk4Step {
    x_next: vec4<f32>,
    p_next: vec4<f32>,
    dx_next: vec4<f32>,
    dp_next: vec4<f32>,
    r_next: f32,
};

fn ks_adaptive_step_size(
    x: vec4<f32>,
    rhs: GeodesicRhs,
    u: FrameUniforms,
    r_plus: f32,
    step_idx: u32,
    jitter_01: f32,
) -> f32 {
    let mass = u.bh_mass;
    let t_ph = smoothstep(1.2 * r_plus, 5.0 * mass, rhs.r);
    let is_escaping = dot(x.yzw, rhs.dx.yzw) > 0.0;
    let t_out = select(0.0, smoothstep(5.0 * mass, 11.0 * mass, rhs.r), is_escaping);
    let t_in_far = select(smoothstep(8.5 * mass, 18.0 * mass, rhs.r), 0.0, is_escaping);
    let cr_base = mix(0.042, 0.115, t_ph);
    let cr = mix(mix(cr_base, 0.24, t_in_far), 0.45, t_out);
    let dp_mag = length(rhs.dp.yzw);
    var h_curv = min(cr * rhs.r, 0.26 / max(dp_mag, 0.02));
    let h_max_base = mix(0.115, 0.24, t_in_far);
    let h_max = max(mix(h_max_base, 0.45, t_out) * rhs.r, 0.038 * mass);

    if has_flag(u.flags, FLAG_ENABLE_DISK)
        && rhs.r >= u.disk_r_in * 0.85
        && rhs.r <= u.disk_r_out * 1.12
    {
        let t_z = smoothstep(1.2 * u.disk_half_height, 3.6 * u.disk_half_height, abs(x.w));
        let h_disk = clamp(0.052 * rhs.r, max(0.55 * u.disk_half_height, 0.19 * mass), 0.38 * mass);
        var h_slab = mix(h_disk, h_max, t_z);
        if x.w * rhs.dx.w < 0.0 && abs(x.w) > 1.2 * u.disk_half_height {
            let h_vert = (abs(x.w) - 0.8 * u.disk_half_height) / max(abs(rhs.dx.w), 0.15);
            h_slab = min(h_slab, max(h_vert, h_disk));
        }
        h_curv = min(h_curv, h_slab);
    }

    let _unused_j = jitter_01;
    let _unused_s = step_idx;

    return clamp(h_curv, 2e-5 * mass, h_max);
}

fn rk4_step(
    x: vec4<f32>,
    p: vec4<f32>,
    k1: GeodesicRhs,
    h: f32,
    mass: f32,
    a: f32,
) -> Rk4Step {
    let half_h = 0.5 * h;
    let x2 = x + half_h * k1.dx;
    let p2 = p + half_h * k1.dp;
    let k2 = ks_rhs(x2, p2, mass, a);

    let x3 = x + half_h * k2.dx;
    let p3 = p + half_h * k2.dp;
    let k3 = ks_rhs(x3, p3, mass, a);

    let x4 = x + h * k3.dx;
    let p4 = p + h * k3.dp;
    let k4 = ks_rhs(x4, p4, mass, a);

    let h6 = h * (1.0 / 6.0);
    let x_next = x + h6 * (k1.dx + 2.0 * (k2.dx + k3.dx) + k4.dx);
    let p_next = p + h6 * (k1.dp + 2.0 * (k2.dp + k3.dp) + k4.dp);
    let next_rhs = ks_rhs(x_next, p_next, mass, a);

    return Rk4Step(x_next, p_next, next_rhs.dx, next_rhs.dp, next_rhs.r);
}

// Gauge-clean weak-field escape direction from p.yzw + 2-point Gauss-Legendre rational tail
// quadrature over [r, +inf) with t = r * u / (1 - u), normalized by |p.yzw| so dlambda = ds / |p.yzw|.
fn ks_asymptotic_escape_dir(x: vec4<f32>, p: vec4<f32>, mass: f32, a: f32) -> vec3<f32> {
    let r = max(length(x.yzw), 1e-4 * max(mass, 1e-4));
    let p_mag = max(length(p.yzw), 1e-6);
    let p_unit = p / p_mag;
    let u_p = p_unit.yzw;
    var p_inf = u_p;

    let u1: f32 = 0.2113248654;
    let u2: f32 = 0.7886751350;

    let d1 = 1.0 - u1;
    let t1 = r * (u1 / d1);
    let x1 = vec4<f32>(x.x - t1 * p_unit.x, x.yzw + t1 * u_p);
    p_inf += (0.5 * r / (d1 * d1)) * ks_rhs(x1, p_unit, mass, a).dp.yzw;

    let d2 = 1.0 - u2;
    let t2 = r * (u2 / d2);
    let x2 = vec4<f32>(x.x - t2 * p_unit.x, x.yzw + t2 * u_p);
    p_inf += (0.5 * r / (d2 * d2)) * ks_rhs(x2, p_unit, mass, a).dp.yzw;

    return normalize(p_inf);
}
