// Curved geodesic-segment Keplerian orbiting planet intersection & crescent shading.

#include "common/constants.wgsl"

struct PlanetHit {
    hit: bool,
    t_frac: f32,
    pos: vec3<f32>,
    normal: vec3<f32>,
    edge_alpha: f32,
};

fn intersect_planet_segment(
    x0: vec3<f32>,
    x1: vec3<f32>,
    uniforms: FrameUniforms,
) -> PlanetHit {
    let miss = PlanetHit(false, 1.0, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), 0.0);
    let center = uniforms.planet_pos_radius.xyz;
    let rad = uniforms.planet_pos_radius.w;
    if rad <= 1e-5 {
        return miss;
    }

    let seg = x1 - x0;
    let seg_len = length(seg);
    if seg_len <= 1e-8 {
        return miss;
    }

    let oc = x0 - center;
    let pad_r = rad + 0.15;
    if dot(oc, oc) > (seg_len + pad_r) * (seg_len + pad_r) {
        return miss;
    }

    let d = seg / seg_len;
    let b = dot(oc, d);
    let c = dot(oc, oc) - rad * rad;
    let d_line2 = max(dot(oc, oc) - b * b, 0.0);
    let d_line = sqrt(d_line2);

    let res_y = f32(max(uniforms.resolution.y, 1u));
    let dist_cam = max(length(center - uniforms.cam_pos.xyz), 0.5);
    let px_world = max(dist_cam * (2.0 * tan(0.5 * uniforms.fov_y_rad) / res_y), 1e-4);
    let aa_half = 0.75 * px_world;

    if d_line >= rad + aa_half {
        return miss;
    }

    // Case A: Interior chord — single-trigger strictly on entry along segment
    if d_line < rad {
        let disc = max(b * b - c, 0.0);
        let s_enter = -b - sqrt(disc);
        if s_enter >= 0.0 && s_enter <= seg_len {
            let p_hit = x0 + d * s_enter;
            let n = normalize(p_hit - center);
            let alpha = smoothstep(rad + aa_half, rad - aa_half, d_line);
            return PlanetHit(true, s_enter / seg_len, p_hit, n, alpha);
        }
        return miss;
    }

    // Case B: Grazing exterior AA rim — single-trigger strictly on periapsis crossing
    if b < 0.0 && (-b) <= seg_len {
        let p_hit = x0 + d * (-b);
        let n = normalize(p_hit - center);
        let alpha = smoothstep(rad + aa_half, rad - aa_half, d_line);
        return PlanetHit(true, (-b) / seg_len, p_hit, n, alpha);
    }

    return miss;
}

fn shade_planet(
    hit: PlanetHit,
    ray_dir: vec3<f32>,
    uniforms: FrameUniforms,
) -> vec3<f32> {
    let l_dir = normalize(-hit.pos);
    let v_dir = normalize(-ray_dir);
    let n = hit.normal;

    let ndotl = dot(n, l_dir);
    let ndotv = max(dot(n, v_dir), 0.0);

    // Soft-penumbra crescent from finite inner-disk angular width
    let sharpness = max(uniforms.planet_params.y, 0.5);
    let lit_term = pow(smoothstep(-0.14, 0.42, ndotl), sharpness);

    // Atmospheric forward-scattering rim
    let rim = pow(1.0 - ndotv, 4.2)
        * smoothstep(-0.35, 0.50, ndotl)
        * uniforms.planet_params.z;

    let albedo = uniforms.planet_params.x;
    let day_col = vec3<f32>(1.0, 0.87, 0.64) * (1.85 * albedo * lit_term);
    let rim_col = vec3<f32>(1.0, 0.92, 0.76) * (1.25 * rim);
    let night_col = vec3<f32>(0.0006, 0.00055, 0.0005);

    return night_col + day_col + rim_col;
}
