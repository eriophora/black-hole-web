use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use wgpu::util::DeviceExt;

use super::context::GpuContext;

pub const STAR_GRID_RES: usize = 64;
pub const STAR_GRID_CELLS: usize = 6 * STAR_GRID_RES * STAR_GRID_RES;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct StarEntry {
    pub dir_temp: [f32; 4],  // (dx, dy, dz, T_eff_K)
    pub flux_meta: [f32; 4], // (bolometric_flux, tint_r, tint_g, tint_b)
}
const _: () = assert!(std::mem::size_of::<StarEntry>() == 32);

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct StarCell {
    pub offset: u32,
    pub count: u32,
}
const _: () = assert!(std::mem::size_of::<StarCell>() == 8);

pub struct StarCatalogGpuData {
    pub cells: Vec<StarCell>,
    pub stars: Vec<StarEntry>,
    pub bright_landmark_dirs: Vec<(String, Vec3, f32, f32)>,
}

pub struct StarCatalogGpuBuffers {
    pub star_cells_buf: wgpu::Buffer,
    pub star_entries_buf: wgpu::Buffer,
}

impl StarCatalogGpuBuffers {
    pub fn new(ctx: &GpuContext, data: &StarCatalogGpuData) -> Self {
        let star_cells_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("star_cells_buf"),
                contents: bytemuck::cast_slice(&data.cells),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let star_entries_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("star_entries_buf"),
                contents: bytemuck::cast_slice(&data.stars),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Self {
            star_cells_buf,
            star_entries_buf,
        }
    }
}

/// Maps unit direction vector to `(face, u_raw, v_raw)` where `face in 0..6` and `u_raw, v_raw in [-1, 1]`.
/// Matches `shaders/astrophysics/sky_procedural.wgsl` identically.
#[inline]
pub fn dir_to_cube_face_uv(d: Vec3) -> (usize, f32, f32) {
    let ax = d.x.abs();
    let ay = d.y.abs();
    let az = d.z.abs();

    if ax >= ay && ax >= az {
        if d.x >= 0.0 {
            (0usize, -d.z / ax, -d.y / ax)
        } else {
            (1usize, d.z / ax, -d.y / ax)
        }
    } else if ay >= ax && ay >= az {
        if d.y >= 0.0 {
            (2usize, d.x / ay, d.z / ay)
        } else {
            (3usize, d.x / ay, -d.z / ay)
        }
    } else if d.z >= 0.0 {
        (4usize, d.x / az, -d.y / az)
    } else {
        (5usize, -d.x / az, -d.y / az)
    }
}

/// Orthonormal cube-face `(N, U, V)` basis matching `dir_to_cube_face_uv` signs such that
/// `(N + u_raw * U + v_raw * V).normalize() == d`.
#[inline]
pub fn face_basis(face: usize) -> (Vec3, Vec3, Vec3) {
    match face {
        0 => (Vec3::X, -Vec3::Z, -Vec3::Y),
        1 => (-Vec3::X, Vec3::Z, -Vec3::Y),
        2 => (Vec3::Y, Vec3::X, Vec3::Z),
        3 => (-Vec3::Y, Vec3::X, -Vec3::Z),
        4 => (Vec3::Z, Vec3::X, -Vec3::Y),
        _ => (-Vec3::Z, -Vec3::X, -Vec3::Y),
    }
}

/// Map unit direction vector to cube-map cell index in `[0..STAR_GRID_CELLS)`.
#[inline]
pub fn dir_to_cube_cell(d: Vec3) -> usize {
    let (face, u_raw, v_raw) = dir_to_cube_face_uv(d);
    let u = (0.5 * (u_raw + 1.0)).clamp(0.0, 0.999999);
    let v = (0.5 * (v_raw + 1.0)).clamp(0.0, 0.999999);
    let ix = ((u * STAR_GRID_RES as f32) as usize).min(STAR_GRID_RES - 1);
    let iy = ((v * STAR_GRID_RES as f32) as usize).min(STAR_GRID_RES - 1);
    face * (STAR_GRID_RES * STAR_GRID_RES) + iy * STAR_GRID_RES + ix
}

fn radec_to_dir(ra_hours: f64, dec_deg: f64) -> Vec3 {
    let ra = (ra_hours / 24.0) * std::f64::consts::TAU;
    let dec = dec_deg.to_radians();
    let eq = Vec3::new(
        (dec.cos() * ra.cos()) as f32,
        (dec.cos() * ra.sin()) as f32,
        dec.sin() as f32,
    );
    let rot_z = Quat::from_rotation_z(-0.35);
    let rot_x = Quat::from_rotation_x(0.42);
    (rot_x * (rot_z * eq)).normalize()
}

fn mag_to_flux(mag_v: f32) -> f32 {
    3.2 * 10.0_f32.powf(-0.4 * mag_v)
}

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        let v = (self.next_u64() >> 40) as u32;
        v as f32 / (1u32 << 24) as f32
    }
}

/// Builds a deterministic 12,000-star astronomical catalog sorted into a 64x64x6 cube-map
/// spatial grid with exact 3x3 cube-UV neighbor replication (`1.05 * cell_uv`) across edges/corners.
pub fn build_star_catalog() -> StarCatalogGpuData {
    let mut raw_stars: Vec<StarEntry> = Vec::with_capacity(12_500);
    let mut landmarks = Vec::new();

    let named: &[(&str, f64, f64, f32, f32)] = &[
        ("Sirius", 6.7525, -16.7161, -1.46, 9940.0),
        ("Canopus", 6.3992, -52.6957, -0.74, 7400.0),
        ("Alpha Centauri", 14.6601, -60.8351, -0.01, 5790.0),
        ("Arcturus", 14.2610, 19.1824, -0.05, 4286.0),
        ("Vega", 18.6156, 38.7837, 0.03, 9600.0),
        ("Capella", 5.2782, 45.9980, 0.08, 4970.0),
        ("Rigel", 5.2423, -8.2016, 0.13, 12100.0),
        ("Procyon", 7.6550, 5.2250, 0.34, 6530.0),
        ("Betelgeuse", 5.9195, 7.4071, 0.42, 3600.0),
        ("Achernar", 1.6286, -57.2368, 0.46, 15000.0),
        ("Hadar", 14.0637, -60.3730, 0.61, 25000.0),
        ("Altair", 19.8464, 8.8683, 0.76, 7550.0),
        ("Acrux", 12.4433, -63.0991, 0.76, 28000.0),
        ("Aldebaran", 4.5987, 16.5093, 0.86, 3910.0),
        ("Antares", 16.4901, -26.4320, 0.96, 3660.0),
        ("Spica", 13.4199, -11.1613, 0.97, 25300.0),
        ("Pollux", 7.7553, 28.0262, 1.14, 4666.0),
        ("Fomalhaut", 22.9608, -29.6222, 1.16, 8590.0),
        ("Deneb", 20.6905, 45.2803, 1.25, 8525.0),
        ("Mimosa", 12.7953, -59.6888, 1.25, 27000.0),
        ("Regulus", 10.1395, 11.9672, 1.35, 12460.0),
        ("Alnilam", 5.6036, -1.2019, 1.69, 27500.0),
        ("Alnitak", 5.6793, -1.9426, 1.77, 29500.0),
        ("Mintaka", 5.5334, -0.2991, 2.23, 29500.0),
        ("Bellatrix", 5.4188, 6.3497, 1.64, 22000.0),
        ("Saiph", 5.7959, -9.6696, 2.09, 26500.0),
        ("Polaris", 2.5303, 89.2641, 1.98, 6015.0),
        ("Alcyone", 3.7914, 24.1051, 2.87, 12258.0),
    ];

    for &(name, ra, dec, mag, temp) in named {
        let dir = radec_to_dir(ra, dec);
        let flux = mag_to_flux(mag);
        landmarks.push((name.to_string(), dir, temp, flux));
        raw_stars.push(StarEntry {
            dir_temp: [dir.x, dir.y, dir.z, temp],
            flux_meta: [flux, 1.0, 1.0, 1.0],
        });
    }

    let gal_z = radec_to_dir(12.857, 27.128).normalize();
    let gal_x = radec_to_dir(17.761, -29.008).normalize();
    let gal_y = gal_z.cross(gal_x).normalize();

    let mut rng = SplitMix64(0x5265_6c61_7469_7669);
    let target_stars = 22_000usize;
    while raw_stars.len() < target_stars {
        let in_disk = rng.next_f32() < 0.50;
        let phi = rng.next_f32() * std::f32::consts::TAU;
        let dir = if in_disk {
            let u1 = rng.next_f32().max(1e-5);
            let u2 = rng.next_f32();
            let g = (-2.0 * u1.ln()).sqrt() * (u2 * std::f32::consts::TAU).cos();
            let lat = (g * 0.24).clamp(-1.45, 1.45);
            let clat = lat.cos();
            let dir_gal = Vec3::new(clat * phi.cos(), clat * phi.sin(), lat.sin());
            (gal_x * dir_gal.x + gal_y * dir_gal.y + gal_z * dir_gal.z).normalize()
        } else {
            let z = 2.0 * rng.next_f32() - 1.0;
            let r_xy = (1.0 - z * z).max(0.0).sqrt();
            Vec3::new(r_xy * phi.cos(), r_xy * phi.sin(), z).normalize()
        };

        let u_mag = rng.next_f32();
        let mag = 0.6 + 5.6 * u_mag.powf(0.42);
        let flux = mag_to_flux(mag);

        let u_spec = rng.next_f32();
        let temp = if u_spec < 0.08 {
            12_000.0 + rng.next_f32() * 18_000.0
        } else if u_spec < 0.28 {
            7_200.0 + rng.next_f32() * 4_500.0
        } else if u_spec < 0.65 {
            4_800.0 + rng.next_f32() * 2_400.0
        } else {
            2_900.0 + rng.next_f32() * 1_900.0
        };

        raw_stars.push(StarEntry {
            dir_temp: [dir.x, dir.y, dir.z, temp],
            flux_meta: [flux, 1.0, 1.0, 1.0],
        });
    }

    // Exact 3x3 cube-UV neighbor replication with `1.05 * cell_uv` wrapping across edges/corners.
    let mut bucket_lists: Vec<Vec<StarEntry>> = vec![Vec::new(); STAR_GRID_CELLS];
    let cell_uv = 2.0 / STAR_GRID_RES as f32;
    let step = 1.05 * cell_uv;

    for star in &raw_stars {
        let d = Vec3::new(star.dir_temp[0], star.dir_temp[1], star.dir_temp[2]).normalize();
        let (face, u_raw, v_raw) = dir_to_cube_face_uv(d);
        let (n_vec, u_vec, v_vec) = face_basis(face);

        let mut seen = [usize::MAX; 9];
        let mut seen_count = 0usize;
        for dv in [-1_i32, 0, 1] {
            for du in [-1_i32, 0, 1] {
                let u_p = u_raw + du as f32 * step;
                let v_p = v_raw + dv as f32 * step;
                let probe = (n_vec + u_p * u_vec + v_p * v_vec).normalize();
                let cell = dir_to_cube_cell(probe);
                if !seen[..seen_count].contains(&cell) {
                    seen[seen_count] = cell;
                    seen_count += 1;
                    bucket_lists[cell].push(*star);
                }
            }
        }
    }

    let mut cells = Vec::with_capacity(STAR_GRID_CELLS);
    let mut sorted_stars = Vec::with_capacity(raw_stars.len() * 4);
    for list in bucket_lists {
        let offset = sorted_stars.len() as u32;
        let count = list.len() as u32;
        sorted_stars.extend(list);
        cells.push(StarCell { offset, count });
    }

    StarCatalogGpuData {
        cells,
        stars: sorted_stars,
        bright_landmark_dirs: landmarks,
    }
}
