// Resolution-independent celestial skyfield:
// - Cube-map accelerated 12,000-star catalog with full 2x2 tangent-plane EWA Jacobian filtering
// - Seamless 3x3x3 neighbor-checked procedural micro-stars with integer hash
// - Band-limited 3D analytic spherical Milky Way & nebula fBm

#include "common/constants.wgsl"
#include "astrophysics/spectral_grrt.wgsl"

struct StarEntry {
    dir_temp: vec4<f32>,  // (dx, dy, dz, T_eff K)
    flux_meta: vec4<f32>, // (bolometric_flux, tint_r, tint_g, tint_b)
};

struct StarCell {
    offset: u32,
    count: u32,
};

// Matches src/gpu/star_grid.rs identically
struct CubeFaceUV {
    face: u32,
    u_raw: f32,
    v_raw: f32,
};

struct CubeCell {
    valid: bool,
    face: u32,
    ix: i32,
    iy: i32,
};

fn dir_to_cube_face_uv(d: vec3<f32>) -> CubeFaceUV {
    let ax = abs(d.x);
    let ay = abs(d.y);
    let az = abs(d.z);
    if ax >= ay && ax >= az {
        if d.x >= 0.0 {
            return CubeFaceUV(0u, -d.z / ax, -d.y / ax);
        }
        return CubeFaceUV(1u, d.z / ax, -d.y / ax);
    } else if ay >= ax && ay >= az {
        if d.y >= 0.0 {
            return CubeFaceUV(2u, d.x / ay, d.z / ay);
        }
        return CubeFaceUV(3u, d.x / ay, -d.z / ay);
    } else if d.z >= 0.0 {
        return CubeFaceUV(4u, d.x / az, -d.y / az);
    }
    return CubeFaceUV(5u, -d.x / az, -d.y / az);
}

fn cube_uv_to_cell_64(c: CubeFaceUV) -> u32 {
    let u = clamp(0.5 * (c.u_raw + 1.0), 0.0, 0.999999);
    let v = clamp(0.5 * (c.v_raw + 1.0), 0.0, 0.999999);
    let ix = min(u32(u * 64.0), 63u);
    let iy = min(u32(v * 64.0), 63u);
    return c.face * 4096u + iy * 64u + ix;
}

fn dir_to_cube_cell(d: vec3<f32>) -> u32 {
    return cube_uv_to_cell_64(dir_to_cube_face_uv(d));
}

// Columns (N, U, V) matching src/gpu/star_grid.rs:90-98 identically
fn cube_face_basis(face: u32) -> mat3x3<f32> {
    switch face {
        case 0u: { return mat3x3<f32>(vec3<f32>( 1.0, 0.0, 0.0), vec3<f32>( 0.0, 0.0,-1.0), vec3<f32>( 0.0,-1.0, 0.0)); }
        case 1u: { return mat3x3<f32>(vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>( 0.0, 0.0, 1.0), vec3<f32>( 0.0,-1.0, 0.0)); }
        case 2u: { return mat3x3<f32>(vec3<f32>( 0.0, 1.0, 0.0), vec3<f32>( 1.0, 0.0, 0.0), vec3<f32>( 0.0, 0.0, 1.0)); }
        case 3u: { return mat3x3<f32>(vec3<f32>( 0.0,-1.0, 0.0), vec3<f32>( 1.0, 0.0, 0.0), vec3<f32>( 0.0, 0.0,-1.0)); }
        case 4u: { return mat3x3<f32>(vec3<f32>( 0.0, 0.0, 1.0), vec3<f32>( 1.0, 0.0, 0.0), vec3<f32>( 0.0,-1.0, 0.0)); }
        default: { return mat3x3<f32>(vec3<f32>( 0.0, 0.0,-1.0), vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>( 0.0,-1.0, 0.0)); }
    }
}

fn wrap_cube_cell(f: u32, ix: i32, iy: i32, n: i32) -> CubeCell {
    if ix >= 0 && ix < n && iy >= 0 && iy < n {
        return CubeCell(true, f, ix, iy);
    }
    let out_x = (ix < 0) || (ix >= n);
    let out_y = (iy < 0) || (iy >= n);
    if out_x && out_y {
        return CubeCell(false, 0u, 0, 0);
    }
    let last = n - 1;
    if ix < 0 {
        switch f {
            case 0u: { return CubeCell(true, 4u, last,      iy); }
            case 1u: { return CubeCell(true, 5u, last,      iy); }
            case 2u: { return CubeCell(true, 1u, iy,        0); }
            case 3u: { return CubeCell(true, 1u, last - iy, last); }
            case 4u: { return CubeCell(true, 1u, last,      iy); }
            default: { return CubeCell(true, 0u, last,      iy); }
        }
    } else if ix >= n {
        switch f {
            case 0u: { return CubeCell(true, 5u, 0,         iy); }
            case 1u: { return CubeCell(true, 4u, 0,         iy); }
            case 2u: { return CubeCell(true, 0u, last - iy, 0); }
            case 3u: { return CubeCell(true, 0u, iy,        last); }
            case 4u: { return CubeCell(true, 0u, 0,         iy); }
            default: { return CubeCell(true, 1u, 0,         iy); }
        }
    } else if iy < 0 {
        switch f {
            case 0u: { return CubeCell(true, 2u, last,      last - ix); }
            case 1u: { return CubeCell(true, 2u, 0,         ix); }
            case 2u: { return CubeCell(true, 5u, last - ix, 0); }
            case 3u: { return CubeCell(true, 4u, ix,        last); }
            case 4u: { return CubeCell(true, 2u, ix,        last); }
            default: { return CubeCell(true, 2u, last - ix, 0); }
        }
    } else {
        switch f {
            case 0u: { return CubeCell(true, 3u, last,      ix); }
            case 1u: { return CubeCell(true, 3u, 0,         last - ix); }
            case 2u: { return CubeCell(true, 4u, ix,        0); }
            case 3u: { return CubeCell(true, 5u, last - ix, last); }
            case 4u: { return CubeCell(true, 3u, ix,        0); }
            default: { return CubeCell(true, 3u, last - ix, last); }
        }
    }
}

// Galactic basis matching src/gpu/star_grid.rs
const GAL_Z: vec3<f32> = vec3<f32>(-0.882959, -0.083988, 0.461874);
const GAL_X: vec3<f32> = vec3<f32>(-0.350667, -0.533806, -0.769470);
const GAL_Y: vec3<f32> = vec3<f32>(0.311177, -0.841376, 0.441878);

fn hash33_i32(c: vec3<i32>) -> vec3<f32> {
    var p = vec3<u32>(bitcast<u32>(c.x), bitcast<u32>(c.y), bitcast<u32>(c.z));
    p = p * vec3<u32>(1664525u, 1013904223u, 2246822519u) + vec3<u32>(374761393u, 668265263u, 1274126177u);
    p.x += p.y * p.z;
    p.y += p.z * p.x;
    p.z += p.x * p.y;
    p ^= p >> vec3<u32>(16u);
    p.x += p.y * p.z;
    p.y += p.z * p.x;
    p.z += p.x * p.y;
    return vec3<f32>(p >> vec3<u32>(8u)) * (1.0 / 16777216.0);
}

fn noise_3d(p: vec3<f32>) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let n000 = hash33_i32(i + vec3<i32>(0, 0, 0)).x;
    let n100 = hash33_i32(i + vec3<i32>(1, 0, 0)).x;
    let n010 = hash33_i32(i + vec3<i32>(0, 1, 0)).x;
    let n110 = hash33_i32(i + vec3<i32>(1, 1, 0)).x;
    let n001 = hash33_i32(i + vec3<i32>(0, 0, 1)).x;
    let n101 = hash33_i32(i + vec3<i32>(1, 0, 1)).x;
    let n011 = hash33_i32(i + vec3<i32>(0, 1, 1)).x;
    let n111 = hash33_i32(i + vec3<i32>(1, 1, 1)).x;

    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);

    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

// Band-limited zero-mean-centered analytic 3D spherical nebula fBm
fn sky_fbm_aniso(p_in: vec3<f32>, sigma_max: f32, max_octaves: u32) -> f32 {
    var p = p_in;
    var val = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var i = 0u; i < max_octaves; i = i + 1u) {
        let nyquist = 1.0 - smoothstep(0.30, 1.6, freq * sigma_max * 1.4);
        if nyquist <= 1e-3 {
            break;
        }
        let n_oct = (noise_3d(p) - 0.5) * nyquist + 0.5;
        val += amp * n_oct;
        p = p * 2.07 + vec3<f32>(11.7, -7.3, 5.1);
        amp *= 0.50;
        freq *= 2.07;
    }
    return val;
}

// Liouville-exact Elliptical Weighted Average (EWA) star kernel against 2x2 pixel covariance.
// Normalized relative to the unlensed reference pixel (sigma_px0^2 + sigma_star^2) so total
// integrated screen flux == |det(J)|^-1 * Phi_0 == |mu| * Phi_0 (exact gravitational magnification),
// keeping every pixel along an Einstein ring at least as bright as the unlensed star.
fn eval_ewa_star(
    diff_vec: vec3<f32>,
    sigma_star: f32,
    sigma_px0: f32,
    t1: vec3<f32>,
    t2: vec3<f32>,
    px_a: f32,
    px_b: f32,
    px_c: f32,
) -> f32 {
    let s2 = sigma_star * sigma_star;
    let a = px_a + s2;
    let b = px_b;
    let c = px_c + s2;
    let det = max(a * c - b * b, 1e-20);
    let inv_det = 1.0 / det;
    let p1 = dot(diff_vec, t1);
    let p2 = dot(diff_vec, t2);
    let dm2 = (c * p1 * p1 - 2.0 * b * p1 * p2 + a * p2 * p2) * inv_det;
    if dm2 > 20.0 {
        return 0.0;
    }
    let ref_det_sqrt = sigma_px0 * sigma_px0 + s2;
    let mag_norm = clamp(ref_det_sqrt * sqrt(inv_det), 0.0, 6.0);
    return mag_norm * exp(-0.5 * dm2);
}

// Zero-miss Tangent-Warped Cube-Face 3x3 procedural micro-star layer on S^2
fn procedural_star_layer_cube(
    dir: vec3<f32>,
    face: u32,
    s01_u: f32,
    s01_v: f32,
    grid_n: i32,
    layer_seed: i32,
    sigma_px0: f32,
    t1: vec3<f32>,
    t2: vec3<f32>,
    px_a: f32,
    px_b: f32,
    px_c: f32,
    lambda_max: f32,
    g_inf: f32,
    density_thresh: f32,
    flux_scale: f32,
) -> vec3<f32> {
    let nf = f32(grid_n);
    let delta = 2.0 / nf;
    let ix0 = clamp(i32(floor(s01_u * nf)), 0, grid_n - 1);
    let iy0 = clamp(i32(floor(s01_v * nf)), 0, grid_n - 1);

    let core_rad: f32 = 0.000085;
    let max_r2 = 20.05 * (lambda_max + core_rad * core_rad);
    let base_basis = cube_face_basis(face);
    let pi_over_4: f32 = 0.78539816339;
    let g_boost = pow(clamp(g_inf, 0.25, 3.0), 1.4);
    var accum = vec3<f32>(0.0);

    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let cell = wrap_cube_cell(face, ix0 + dx, iy0 + dy, grid_n);
            if !cell.valid {
                continue;
            }
            let h = hash33_i32(vec3<i32>(cell.ix, cell.iy, i32(cell.face) + layer_seed));
            if h.x < density_thresh {
                continue;
            }
            let xu = -1.0 + (f32(cell.ix) + 0.5 + (h.y - 0.5) * 0.68) * delta;
            let xv = -1.0 + (f32(cell.iy) + 0.5 + (h.z - 0.5) * 0.68) * delta;
            let su = xu * (0.785398 + 0.214602 * xu * xu);
            let sv = xv * (0.785398 + 0.214602 * xv * xv);
            var b = base_basis;
            if cell.face != face {
                b = cube_face_basis(cell.face);
            }
            let star_dir = normalize(b[0] + su * b[1] + sv * b[2]);
            let diff = star_dir - dir;
            if dot(diff, diff) > max_r2 {
                continue;
            }
            let w = eval_ewa_star(diff, core_rad, sigma_px0, t1, t2, px_a, px_b, px_c);
            if w > 0.0 {
                let frac_x = clamp((h.x - density_thresh) / (1.0 - density_thresh), 0.0, 1.0);
                let flux = flux_scale * pow(frac_x, 2.4) * g_boost;
                let t_emit = mix(3400.0, 16000.0, h.y * h.y);
                let t_obs = t_emit * g_inf;
                accum += (flux * w) * normalize_luma(blackbody_rgb(t_obs));
            }
        }
    }
    return accum;
}

fn eval_sky_anisotropic_gr(
    dir: vec3<f32>,
    jx: vec3<f32>,
    jy: vec3<f32>,
    g_inf: f32,
    uniforms: FrameUniforms,
) -> vec3<f32> {
    if g_inf <= 1e-5 {
        return vec3<f32>(0.0);
    }

    let ref_up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(dir.z) < 0.9);
    let t1 = normalize(cross(ref_up, dir));
    let t2 = cross(dir, t1);
    let jx2 = vec2<f32>(dot(jx, t1), dot(jx, t2));
    let jy2 = vec2<f32>(dot(jy, t1), dot(jy, t2));

    let pixel_ang = (2.0 * tan(0.5 * uniforms.fov_y_rad)) / f32(max(uniforms.resolution.y, 1u));
    let sigma_px0 = 0.40 * pixel_ang;
    let star_gain = max(uniforms.render_params.w, 0.01);
    let g_boost = pow(clamp(g_inf, 0.25, 3.0), 1.4);

    let floor_var: f32 = 1e-11;
    let px_a = 0.16 * (jx2.x * jx2.x + jy2.x * jy2.x) + floor_var;
    let px_b = 0.16 * (jx2.x * jx2.y + jy2.x * jy2.y);
    let px_c = 0.16 * (jx2.y * jx2.y + jy2.y * jy2.y) + floor_var;
    let sigma_max = max(max(length(jx), length(jy)), 1e-5);

    // True obsidian vacuum background (0.0) so high exposure never lifts empty space to slate-grey.
    var sky_rgb = vec3<f32>(0.0);

    // 1. Subtle band-limited 3D spherical Milky Way & emission nebula (strictly localized to galactic plane)
    let gd = vec3<f32>(dot(dir, GAL_X), dot(dir, GAL_Y), dot(dir, GAL_Z));
    let lat = gd.z;
    let bulge = exp(-dot(gd - vec3<f32>(1.0, 0.0, 0.0), gd - vec3<f32>(1.0, 0.0, 0.0)) * 8.5);
    let plane = exp(-lat * lat * 28.0);
    let band_gate = smoothstep(0.02, 0.14, max(plane, bulge));

    if band_gate > 1e-4 {
        let octs = select(3u, 5u, plane >= 0.06 || bulge >= 0.03);
        let n_large = sky_fbm_aniso(gd * 3.8 + vec3<f32>(1.2, 3.4, -2.1), sigma_max, octs);
        var n_detail = 0.5;
        var dust_field = 0.3;
        if plane >= 0.04 || bulge >= 0.02 {
            n_detail = sky_fbm_aniso(gd * 12.5 + vec3<f32>(5.3, -1.9, 4.7), sigma_max, octs);
            dust_field = sky_fbm_aniso(gd * 9.0 + vec3<f32>(-4.1, 2.8, 7.2), sigma_max, octs);
        }

        let dust_mask = exp(-lat * lat * 110.0) * smoothstep(0.42, 0.74, dust_field);
        let ext = exp(-2.2 * dust_mask * vec3<f32>(0.78, 0.96, 1.24));

        let starcloud = band_gate * (
            0.00028 * plane * pow(0.35 + 1.10 * n_large, 1.35) * (0.55 + 0.85 * n_detail)
            + 0.00068 * bulge * (0.60 + 0.60 * n_detail)
        );

        let t_gal = mix(5400.0, 4100.0, bulge) * g_inf;
        sky_rgb += normalize_luma(blackbody_rgb(t_gal)) * starcloud * ext;

        let neb_mask = band_gate * exp(-lat * lat * 48.0) * smoothstep(0.58, 0.84, n_large * n_detail * 2.0);
        let neb_col = mix(vec3<f32>(0.00022, 0.00006, 0.00011), vec3<f32>(0.00006, 0.00014, 0.00025), n_detail);
        sky_rgb += neb_col * neb_mask * ext;
    }

    let diff_ac = px_a - px_c;
    let lambda_max = 0.5 * ((px_a + px_c) + sqrt(diff_ac * diff_ac + 4.0 * px_b * px_b));

    // 2. Cube-grid 22,000-star catalog with Liouville-exact 2x2 EWA filtering
    let cube_uv = dir_to_cube_face_uv(dir);
    let cell_idx = cube_uv_to_cell_64(cube_uv);
    let cell = star_cells[cell_idx];
    let max_count = min(cell.count, 128u);
    for (var i = 0u; i < max_count; i = i + 1u) {
        let st = star_entries[cell.offset + i];
        let st_dir = st.dir_temp.xyz;
        if dot(dir, st_dir) <= 0.992 {
            continue;
        }
        let diff = st_dir - dir;
        let flux = st.flux_meta.x;
        let core_rad = clamp(0.000115 * pow(clamp(flux, 0.05, 25.0), 0.25), 0.000095, 0.00022);
        let max_r2 = 20.05 * (lambda_max + core_rad * core_rad);
        if dot(diff, diff) >= max_r2 {
            continue;
        }
        let w = eval_ewa_star(diff, core_rad, sigma_px0, t1, t2, px_a, px_b, px_c);
        if w > 0.0 {
            let t_emit = st.dir_temp.w;
            let t_obs = t_emit * g_inf;
            let amp = 1.35 * star_gain * g_boost * pow(flux, 0.82) * w;
            sky_rgb += normalize_luma(blackbody_rgb(t_obs)) * st.flux_meta.yzw * amp;
        }
    }

    // 3. Zero-miss Tangent-Warped Cube-Face 3x3 procedural micro-stars (single crisp layer for WebGPU perf)
    let inv_pi_2: f32 = 2.0 / PI;
    let s01_u = clamp(0.5 + inv_pi_2 * atan(cube_uv.u_raw), 0.0, 0.999999);
    let s01_v = clamp(0.5 + inv_pi_2 * atan(cube_uv.v_raw), 0.0, 0.999999);
    let len_jx = length(jx);
    let len_jy = length(jy);
    let max_j = max(len_jx, len_jy);

    let w1 = 1.0 - smoothstep(0.010, 0.028, max_j);
    if w1 > 1e-4 {
        sky_rgb += w1 * procedural_star_layer_cube(
            dir, cube_uv.face, s01_u, s01_v, 192, 100,
            sigma_px0, t1, t2, px_a, px_b, px_c, lambda_max, g_inf, 0.85, 0.25 * star_gain,
        );
    }

    let _unused_u = uniforms.bh_mass;
    return sky_rgb;
}

fn eval_sky_anisotropic(
    dir: vec3<f32>,
    jx: vec3<f32>,
    jy: vec3<f32>,
    uniforms: FrameUniforms,
) -> vec3<f32> {
    return eval_sky_anisotropic_gr(dir, jx, jy, 1.0, uniforms);
}
