// Pure-GR critical-curve sub-pixel geodesic supersampling pass.
// Runs 4 rotated-grid Kerr-Schild geodesics ONLY on the ~1% of pixels straddling the
// shadow critical curve / n>=2 photon sub-rings, keeping Pass 1 register pressure minimal
// while resolving real n=2 and n=3 relativistic sub-rings cleanly with zero synthetic glow.

#include "common/constants.wgsl"
#include "metrics/kerr_schild.wgsl"
#include "astrophysics/spectral_grrt.wgsl"
#include "astrophysics/disk_gargantua.wgsl"
#include "astrophysics/planet.wgsl"
#include "integrator/rk4_adaptive.wgsl"

@group(0) @binding(0) var<uniform> u: FrameUniforms;
@group(0) @binding(1) var hdr_in: texture_2d<f32>;
@group(0) @binding(2) var hdr_out: texture_storage_2d<rgba16float, write>;

var<workgroup> wg_tetrad: array<vec4<f32>, 4>;
var<workgroup> wg_meta: array<f32, 64>;

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

    var center_px = vec4<f32>(0.0);
    if in_bounds {
        center_px = textureLoad(hdr_in, vec2<i32>(gid.xy), 0);
        wg_meta[lidx] = center_px.a;
    } else {
        wg_meta[lidx] = 0.0;
    }
    // Single uniform barrier across all 64 workgroup threads (100% WebGPU/Tint uniformity compliant)
    workgroupBarrier();

    if !in_bounds {
        return;
    }

    var need_ssaa = false;
    if has_flag(u.flags, FLAG_ENABLE_DISK) && has_flag(u.flags, FLAG_ENABLE_PHOTON_SSAA) {
        let lx = i32(lid.x);
        let ly = i32(lid.y);
        let x_l = max(i32(gid.x) - 1, 0);
        let x_r = min(i32(gid.x) + 1, i32(w_u32) - 1);
        let y_u = max(i32(gid.y) - 1, 0);
        let y_d = min(i32(gid.y) + 1, i32(h_u32) - 1);

        let m_self = center_px.a;
        let m_l = select(textureLoad(hdr_in, vec2<i32>(x_l, i32(gid.y)), 0).a, wg_meta[lid.y * 8u + u32(lx - 1)], lx > 0);
        let m_r = select(textureLoad(hdr_in, vec2<i32>(x_r, i32(gid.y)), 0).a, wg_meta[lid.y * 8u + u32(lx + 1)], lx < 7);
        let m_u = select(textureLoad(hdr_in, vec2<i32>(i32(gid.x), y_u), 0).a, wg_meta[u32(ly - 1) * 8u + lid.x], ly > 0);
        let m_d = select(textureLoad(hdr_in, vec2<i32>(i32(gid.x), y_d), 0).a, wg_meta[u32(ly + 1) * 8u + lid.x], ly < 7);

        let shadow_edge = ((m_self >= 0.0) != (m_l >= 0.0))
            || ((m_self >= 0.0) != (m_r >= 0.0))
            || ((m_self >= 0.0) != (m_u >= 0.0))
            || ((m_self >= 0.0) != (m_d >= 0.0));
        let max_orbits = max(
            abs(m_self),
            max(max(abs(m_l), abs(m_r)), max(abs(m_u), abs(m_d))),
        );
        need_ssaa = shadow_edge || (max_orbits >= 1.82);
    }

    if !need_ssaa {
        textureStore(hdr_out, vec2<i32>(gid.xy), vec4<f32>(center_px.rgb, 1.0));
        return;
    }

    let w = f32(w_u32);
    let h = f32(h_u32);
    let aspect = w / h;
    let tan_half_fov = tan(0.5 * u.fov_y_rad);
    let tetrad = array<vec4<f32>, 4>(wg_tetrad[0], wg_tetrad[1], wg_tetrad[2], wg_tetrad[3]);
    let jitter = ign_noise(gid.xy, u.frame_index);

    // 4-sample rotated-grid (RGSS) + center = 5-sample pure-GR sub-pixel integration on the critical curve
    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(-0.375, -0.125),
        vec2<f32>( 0.125, -0.375),
        vec2<f32>(-0.125,  0.375),
        vec2<f32>( 0.375,  0.125),
    );

    var accum = center_px.rgb;
    for (var s = 0u; s < 4u; s = s + 1u) {
        let off = offsets[s];
        let sx = ((f32(gid.x) + 0.5 + off.x) / w * 2.0 - 1.0) * aspect * tan_half_fov;
        let sy = (1.0 - (f32(gid.y) + 0.5 + off.y) / h * 2.0) * tan_half_fov;
        let sub_dir = normalize(vec3<f32>(sx, sy, 1.0));
        let p_sub = ks_init_null_momentum(x0, tetrad, sub_dir, mass, a);
        let sub_res = trace_geodesic(x0, p_sub, u, fract(jitter + 0.25 * f32(s + 1u)));
        accum += sub_res.radiance;
    }

    let final_rgb = max(accum * 0.20, vec3<f32>(0.0));
    textureStore(hdr_out, vec2<i32>(gid.xy), vec4<f32>(final_rgb, 1.0));
}
