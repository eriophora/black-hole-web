// Main per-pixel Kerr-Schild geodesic raytracer with 2x2 workgroup anisotropic
// Jacobian ray differentials for artifact-free lensed starfield & nebula.
// Outputs linear HDR radiance to rgba16float for separable HDR bloom & AgX tonemapping.

#include "common/constants.wgsl"
#include "metrics/kerr_schild.wgsl"
#include "astrophysics/spectral_grrt.wgsl"
#include "astrophysics/disk_gargantua.wgsl"
#include "astrophysics/planet.wgsl"
#include "integrator/rk4_adaptive.wgsl"
#include "astrophysics/sky_procedural.wgsl"

@group(0) @binding(0) var<uniform> u: FrameUniforms;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<storage, read> star_cells: array<StarCell>;
@group(0) @binding(3) var<storage, read> star_entries: array<StarEntry>;

var<workgroup> wg_escape_dir: array<vec4<f32>, 64>;
var<workgroup> wg_tetrad: array<vec4<f32>, 4>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let w_u32 = u.resolution.x;
    let h_u32 = u.resolution.y;
    let in_bounds = (gid.x < w_u32) && (gid.y < h_u32);

    let x0 = vec4<f32>(0.0, u.cam_pos.xyz);
    let mass = u.bh_mass;
    let spin_clamped = clamp(u.bh_spin, -0.999, 0.999);
    let a = spin_clamped * mass;
    if lidx == 0u {
        let r_plus = ks_event_horizon(mass, spin_clamped);
        let tet = ks_build_observer_tetrad(
            x0,
            u.cam_right.xyz,
            u.cam_up.xyz,
            u.cam_fwd.xyz,
            mass,
            a,
            r_plus,
        );
        wg_tetrad[0] = tet[0];
        wg_tetrad[1] = tet[1];
        wg_tetrad[2] = tet[2];
        wg_tetrad[3] = tet[3];
    }
    workgroupBarrier();

    var escaped = false;
    var esc_dir = vec3<f32>(0.0, 1.0, 0.0);
    var hdr = vec3<f32>(0.0);
    var trace_res = GeodesicTraceResult(
        false,
        vec3<f32>(0.0, 1.0, 0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        1e6,
        1e6,
        0.0,
        0u,
        vec3<f32>(0.0),
        1.0,
        vec3<f32>(1.0),
        0.0,
    );

    if in_bounds {
        let w = f32(w_u32);
        let h = f32(h_u32);
        let aspect = w / h;
        let tan_half_fov = tan(0.5 * u.fov_y_rad);

        let ndc_x = ((f32(gid.x) + 0.5) / w * 2.0 - 1.0) * aspect * tan_half_fov;
        let ndc_y = (1.0 - (f32(gid.y) + 0.5) / h * 2.0) * tan_half_fov;
        let n_dir = normalize(vec3<f32>(ndc_x, ndc_y, 1.0));

        let tetrad = array<vec4<f32>, 4>(wg_tetrad[0], wg_tetrad[1], wg_tetrad[2], wg_tetrad[3]);
        let p0 = ks_init_null_momentum(x0, tetrad, n_dir, mass, a);
        let jitter = ign_noise(gid.xy, u.frame_index);

        trace_res = trace_geodesic(x0, p0, u, jitter);
        escaped = trace_res.escaped;
        esc_dir = trace_res.dir_out;
        hdr = trace_res.radiance;
    }

    // Uniform barrier across all 64 workgroup threads (never return before workgroupBarrier!)
    wg_escape_dir[lidx] = vec4<f32>(esc_dir, select(0.0, 1.0, in_bounds && escaped));
    workgroupBarrier();

    if in_bounds {
        var sky_contrib = vec3<f32>(0.0);
        if escaped && has_flag(u.flags, FLAG_ENABLE_STARS) && trace_res.transmittance >= 0.005 {
            // Central-difference screen-to-sky Jacobian across the 8x8 workgroup (no 2x2 quad secant bias)
            let lx_i = i32(lid.x);
            let ly_i = i32(lid.y);
            let has_l = lx_i > 0;
            let has_r = lx_i < 7;
            let has_u = ly_i > 0;
            let has_d = ly_i < 7;

            let nb_l = wg_escape_dir[lid.y * 8u + u32(max(lx_i - 1, 0))];
            let nb_r = wg_escape_dir[lid.y * 8u + u32(min(lx_i + 1, 7))];
            let nb_u = wg_escape_dir[u32(max(ly_i - 1, 0)) * 8u + lid.x];
            let nb_d = wg_escape_dir[u32(min(ly_i + 1, 7)) * 8u + lid.x];

            let pixel_ang = (2.0 * tan(0.5 * u.fov_y_rad)) / f32(max(h_u32, 1u));
            let ref_up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(esc_dir.z) < 0.9);
            let t1 = normalize(cross(ref_up, esc_dir));
            let t2 = cross(esc_dir, t1);

            var jx = vec3<f32>(0.0);
            var jy = vec3<f32>(0.0);
            var has_x = false;
            var has_y = false;
            var raw_div = pixel_ang;

            let ok_l = has_l && nb_l.w > 0.5;
            let ok_r = has_r && nb_r.w > 0.5;
            if ok_l || ok_r {
                var d = vec3<f32>(0.0);
                if ok_l && ok_r && dot(nb_l.xyz, esc_dir) > 0.85 && dot(nb_r.xyz, esc_dir) > 0.85 {
                    d = 0.5 * (nb_r.xyz - nb_l.xyz);
                } else if ok_r && (!ok_l || dot(nb_r.xyz, esc_dir) >= dot(nb_l.xyz, esc_dir)) {
                    d = nb_r.xyz - esc_dir;
                } else {
                    d = esc_dir - nb_l.xyz;
                }
                raw_div = max(raw_div, length(d));
                let d_tan = d - esc_dir * dot(d, esc_dir);
                let len_x = length(d_tan);
                if len_x < 0.35 {
                    jx = d_tan * min(len_x, 0.22) / max(len_x, 1e-8);
                } else {
                    let dir_x = select(t1, d_tan / max(len_x, 1e-8), len_x > 1e-6);
                    jx = dir_x * 0.22;
                }
                has_x = true;
            }

            let ok_u = has_u && nb_u.w > 0.5;
            let ok_d = has_d && nb_d.w > 0.5;
            if ok_u || ok_d {
                var d = vec3<f32>(0.0);
                if ok_u && ok_d && dot(nb_u.xyz, esc_dir) > 0.85 && dot(nb_d.xyz, esc_dir) > 0.85 {
                    d = 0.5 * (nb_d.xyz - nb_u.xyz);
                } else if ok_d && (!ok_u || dot(nb_d.xyz, esc_dir) >= dot(nb_u.xyz, esc_dir)) {
                    d = nb_d.xyz - esc_dir;
                } else {
                    d = esc_dir - nb_u.xyz;
                }
                raw_div = max(raw_div, length(d));
                let d_tan = d - esc_dir * dot(d, esc_dir);
                let len_y = length(d_tan);
                if len_y < 0.35 {
                    jy = d_tan * min(len_y, 0.22) / max(len_y, 1e-8);
                } else {
                    let dir_y = select(t2, d_tan / max(len_y, 1e-8), len_y > 1e-6);
                    jy = dir_y * 0.22;
                }
                has_y = true;
            }

            let near_limb = trace_res.n_half_orbits > 0.85;
            if !has_x && has_y {
                let len_y = length(jy);
                let ortho = select(t1, normalize(cross(esc_dir, jy)), len_y > 1e-7);
                let miss_len = select(pixel_ang, max(pixel_ang, max(len_y, 0.04)), near_limb);
                jx = ortho * miss_len;
                if near_limb {
                    raw_div = max(raw_div, 0.8);
                }
            } else if has_x && !has_y {
                let len_x = length(jx);
                let ortho = select(t2, normalize(cross(esc_dir, jx)), len_x > 1e-7);
                let miss_len = select(pixel_ang, max(pixel_ang, max(len_x, 0.04)), near_limb);
                jy = ortho * miss_len;
                if near_limb {
                    raw_div = max(raw_div, 0.8);
                }
            } else if !has_x && !has_y {
                let miss_len = select(pixel_ang, 0.04, near_limb);
                jx = t1 * miss_len;
                jy = t2 * miss_len;
                if near_limb {
                    raw_div = max(raw_div, 0.8);
                }
            }

            let g_inf = doppler_factor(
                vec4<f32>(0.0, esc_dir * 800.0),
                trace_res.p_final,
                vec4<f32>(1.0, 0.0, 0.0, 0.0),
                u,
            );
            let alias_blend = smoothstep(0.32, 0.82, raw_div);
            var sky_rgb = vec3<f32>(0.0);
            if alias_blend < 0.9999 {
                let eval_rgb = eval_sky_anisotropic_gr(esc_dir, jx, jy, g_inf, u);
                sky_rgb = mix(eval_rgb, vec3<f32>(0.0), alias_blend);
            }
            sky_contrib = trace_res.transmittance_rgb * sky_rgb;
        }

        hdr += sky_contrib;
        // Pack signed half-orbit count into alpha (+ for escaped, - for captured) for Pass 2 photon-ring SSAA
        let orbit_tag = select(-max(trace_res.n_half_orbits, 0.01), max(trace_res.n_half_orbits, 0.01), escaped);
        textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(max(hdr, vec3<f32>(0.0)), orbit_tag));
    }
}
