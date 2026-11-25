// Full geodesic tracer integrating Kerr-Schild rays against the Gargantua volumetric disk & planet.

#include "integrator/rk4_core.wgsl"
#include "astrophysics/spectral_grrt.wgsl"
#include "astrophysics/disk_gargantua.wgsl"
#include "astrophysics/planet.wgsl"

struct GeodesicTraceResult {
    escaped: bool,
    dir_out: vec3<f32>,
    x_final: vec4<f32>,
    p_final: vec4<f32>,
    r_min: f32,
    r_final: f32,
    n_half_orbits: f32,
    steps: u32,
    radiance: vec3<f32>,
    transmittance: f32,
    transmittance_rgb: vec3<f32>,
    n_crossings: f32,
};

fn trace_geodesic(
    x0: vec4<f32>,
    p0: vec4<f32>,
    u: FrameUniforms,
    jitter_01: f32,
) -> GeodesicTraceResult {
    let mass = u.bh_mass;
    let spin_clamped = clamp(u.bh_spin, -0.999, 0.999);
    let a = spin_clamped * mass;
    let r_plus = ks_event_horizon(mass, spin_clamped);
    let r_minus = ks_inner_horizon(mass, spin_clamped);
    let r_inner_sink = max(1.03 * r_minus, 0.16 * mass);
    let r_escape = R_ESCAPE_TAIL * mass;

    var x = x0;
    var p = p0;
    var k1 = ks_rhs(x, p, mass, a);
    var r_curr = k1.r;
    var r_min = r_curr;
    var u_curr = normalize(x0.yzw);
    var total_angle = 0.0;
    var steps = 0u;
    var escaped = false;
    var dir_out = normalize(p.yzw);
    var state = init_radiance_state();
    state.prev_z = x0.w;

    let z_bound = 2.8 * u.disk_half_height;
    let r_disk_min = u.disk_r_in * 0.88;
    let r_disk_max = u.disk_r_out * 1.04;
    let planet_center = u.planet_pos_radius.xyz;
    let planet_rad = u.planet_pos_radius.w;
    let planet_r_orb = length(planet_center);
    let planet_r_min = max(planet_r_orb - planet_rad - 0.5, 0.0);
    let planet_r_max = planet_r_orb + planet_rad + 0.5;
    let check_planet = has_flag(u.flags, FLAG_ENABLE_PLANET) && planet_rad > 1e-5;
    let check_disk = has_flag(u.flags, FLAG_ENABLE_DISK);

    var r_isco = 0.0;
    var omega_isco = 0.0;
    if check_disk {
        r_isco = ks_isco_radius(mass, spin_clamped);
        let sqrt_m = sqrt(max(mass, 1e-4));
        omega_isco = sqrt_m / max(pow(r_isco, 1.5) + a * sqrt_m, 1e-4);
    }

    for (var step = 0u; step < 640u; step = step + 1u) {
        let h = ks_adaptive_step_size(x, k1, u, r_plus, step, jitter_01);
        let res = rk4_step(x, p, k1, h, mass, a);
        steps += 1u;

        r_min = min(r_min, res.r_next);
        let u_next = normalize(res.x_next.yzw);
        let chord = length(u_next - u_curr);
        total_angle += chord + (1.0 / 24.0) * chord * chord * chord;
        u_curr = u_next;

        let r_min_seg = min(r_curr, res.r_next);
        let r_max_seg = max(r_curr, res.r_next);

        if check_planet && r_max_seg >= planet_r_min && r_min_seg <= planet_r_max {
            let hit = intersect_planet_segment(x.yzw, res.x_next.yzw, u);
            if hit.hit {
                let ray_dir = normalize(res.x_next.yzw - x.yzw);
                let p_col = shade_planet(hit, ray_dir, u);
                state.rgb += state.transmittance_rgb * p_col * hit.edge_alpha;
                state.transmittance_rgb *= (1.0 - hit.edge_alpha);
                state.transmittance *= (1.0 - hit.edge_alpha);
                if state.transmittance < 1e-3 {
                    x = res.x_next;
                    p = res.p_next;
                    r_curr = res.r_next;
                    break;
                }
            }
        }

        if check_disk {
            let z0 = x.w;
            let z1 = res.x_next.w;
            let z_lo = min(z0, z1);
            let z_hi = max(z0, z1);
            if z_lo <= z_bound
                && z_hi >= -z_bound
                && r_max_seg >= r_disk_min
                && r_min_seg <= r_disk_max
            {
                var t0 = 0.0;
                var t1 = 1.0;
                let dz = z1 - z0;
                if abs(dz) > 1e-6 {
                    let ta = (-z_bound - z0) / dz;
                    let tb = (z_bound - z0) / dz;
                    t0 = clamp(min(ta, tb), 0.0, 1.0);
                    t1 = clamp(max(ta, tb), 0.0, 1.0);
                }
                if t1 > t0 {
                    let t_mid = 0.5 * (t0 + t1);
                    let x_mid = mix(x, res.x_next, t_mid);
                    let p_mid = mix(p, res.p_next, t_mid);
                    let z_a = mix(z0, z1, t0);
                    let z_b = mix(z0, z1, t1);
                    let ds_spatial = (t1 - t0) * length(res.x_next.yzw - x.yzw);
                    eval_disk_step_fast(
                        x_mid,
                        p_mid,
                        z_a,
                        z_b,
                        ds_spatial,
                        total_angle * INV_PI,
                        mass,
                        a,
                        r_plus,
                        r_isco,
                        omega_isco,
                        u,
                        &state,
                    );
                }
            }
            state.prev_z = z1;
        }

        x = res.x_next;
        p = res.p_next;
        k1 = GeodesicRhs(res.dx_next, res.dp_next, res.r_next);
        r_curr = res.r_next;

        if state.transmittance < 1e-3 {
            break;
        }
        if !(r_curr >= 0.0 && r_curr < 1e20) {
            break;
        }
        if r_curr <= r_inner_sink {
            break;
        }
        if abs(r_curr - r_plus) < 0.10 * mass && abs(k1.dx.x) > 80.0 * max(abs(p.x), 0.02) {
            break;
        }
        if r_curr >= r_escape && dot(x.yzw, k1.dx.yzw) > 0.0 {
            escaped = true;
            dir_out = ks_asymptotic_escape_dir(x, p, mass, a);
            break;
        }
    }

    return GeodesicTraceResult(
        escaped,
        dir_out,
        x,
        p,
        r_min,
        r_curr,
        total_angle * INV_PI,
        steps,
        state.rgb,
        state.transmittance,
        state.transmittance_rgb,
        state.n_crossings,
    );
}
