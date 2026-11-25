use wgpu::util::DeviceExt;

use super::context::GpuContext;

pub const NOISE_TEX_SIZE: u32 = 128;

pub struct DiskNoiseTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

#[inline]
fn quintic_fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lattice_hash(ix: i32, iy: i32, iz: i32, period: i32, seed: u64) -> f32 {
    let wx = ix.rem_euclid(period) as u64;
    let wy = iy.rem_euclid(period) as u64;
    let wz = iz.rem_euclid(period) as u64;
    let mut z = seed
        ^ wx.wrapping_mul(0x9E3779B97F4A7C15)
        ^ wy.wrapping_mul(0xBF58476D1CE4E5B9)
        ^ wz.wrapping_mul(0x94D049BB133111EB);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    ((z >> 40) as f32) * (1.0 / 16_777_216.0)
}

fn periodic_value_noise_3d(x_norm: f32, y_norm: f32, z_norm: f32, period: i32, seed: u64) -> f32 {
    let p = period as f32;
    let px = x_norm * p;
    let py = y_norm * p;
    let pz = z_norm * p;

    let ix = px.floor() as i32;
    let iy = py.floor() as i32;
    let iz = pz.floor() as i32;

    let fx = quintic_fade(px - ix as f32);
    let fy = quintic_fade(py - iy as f32);
    let fz = quintic_fade(pz - iz as f32);

    let n000 = lattice_hash(ix, iy, iz, period, seed);
    let n100 = lattice_hash(ix + 1, iy, iz, period, seed);
    let n010 = lattice_hash(ix, iy + 1, iz, period, seed);
    let n110 = lattice_hash(ix + 1, iy + 1, iz, period, seed);
    let n001 = lattice_hash(ix, iy, iz + 1, period, seed);
    let n101 = lattice_hash(ix + 1, iy, iz + 1, period, seed);
    let n011 = lattice_hash(ix, iy + 1, iz + 1, period, seed);
    let n111 = lattice_hash(ix + 1, iy + 1, iz + 1, period, seed);

    let nx00 = n000 + (n100 - n000) * fx;
    let nx10 = n010 + (n110 - n010) * fx;
    let nx01 = n001 + (n101 - n001) * fx;
    let nx11 = n011 + (n111 - n011) * fx;

    let nxy0 = nx00 + (nx10 - nx00) * fy;
    let nxy1 = nx01 + (nx11 - nx01) * fy;

    nxy0 + (nxy1 - nxy0) * fz
}

fn periodic_fbm(
    x_norm: f32,
    y_norm: f32,
    z_norm: f32,
    periods: &[i32; 3],
    weights: &[f32; 3],
    seed_base: u64,
    ridged: bool,
) -> f32 {
    let mut sum = 0.0_f32;
    let mut w_sum = 0.0_f32;
    for i in 0..3 {
        let mut n = periodic_value_noise_3d(
            x_norm,
            y_norm,
            z_norm,
            periods[i],
            seed_base.wrapping_add((i as u64).wrapping_mul(0xD6E8FEB86659FD93)),
        );
        if ridged {
            n = 1.0 - (2.0 * n - 1.0).abs();
        }
        sum += n * weights[i];
        w_sum += weights[i];
    }
    sum / w_sum.max(1e-6)
}

pub fn build_disk_noise_bytes() -> Vec<u8> {
    let size = NOISE_TEX_SIZE as usize;
    let total_voxels = size * size * size;
    let mut raw_f32 = vec![[0.0_f32; 4]; total_voxels];

    let mut rng = SplitMix64(0x4741_5247_414E_5455);
    let seed_r = rng.next_u64();
    let seed_g = rng.next_u64();
    let seed_b = rng.next_u64();
    let seed_a = rng.next_u64();

    let mut ch_min = [f32::INFINITY; 4];
    let mut ch_max = [f32::NEG_INFINITY; 4];
    let inv_size = 1.0_f32 / size as f32;

    for iz in 0..size {
        let zn = iz as f32 * inv_size;
        for iy in 0..size {
            let yn = iy as f32 * inv_size;
            let row_base = (iz * size + iy) * size;
            for ix in 0..size {
                let xn = ix as f32 * inv_size;

                // R: Macro turbulence (periods 4, 8, 16)
                let r = periodic_fbm(xn, yn, zn, &[4, 8, 16], &[0.55, 0.30, 0.15], seed_r, false);
                // G: Ionized ridged plasma filaments (periods 8, 16, 32)
                let g_raw =
                    periodic_fbm(xn, yn, zn, &[8, 16, 32], &[0.50, 0.32, 0.18], seed_g, true);
                let g = g_raw.clamp(0.0, 1.0).powf(1.3);
                // B: Decorrelated dust density (independent seed, periods 6, 12, 24)
                let b = periodic_fbm(xn, yn, zn, &[6, 12, 24], &[0.50, 0.32, 0.18], seed_b, false);
                // A: High-frequency micro-granulation (periods 16, 32, 64)
                let a = periodic_fbm(
                    xn,
                    yn,
                    zn,
                    &[16, 32, 64],
                    &[0.30, 0.40, 0.30],
                    seed_a,
                    false,
                );

                let vals = [r, g, b, a];
                for c in 0..4 {
                    ch_min[c] = ch_min[c].min(vals[c]);
                    ch_max[c] = ch_max[c].max(vals[c]);
                }
                raw_f32[row_base + ix] = vals;
            }
        }
    }

    let mut bytes = Vec::with_capacity(total_voxels * 4);
    for vals in &raw_f32 {
        for c in 0..4 {
            let span = (ch_max[c] - ch_min[c]).max(1e-5);
            let norm = ((vals[c] - ch_min[c]) / span).clamp(0.0, 1.0);
            bytes.push((norm * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
        }
    }
    bytes
}

impl DiskNoiseTexture {
    pub fn new(ctx: &GpuContext) -> Self {
        let bytes = build_disk_noise_bytes();
        let extent = wgpu::Extent3d {
            width: NOISE_TEX_SIZE,
            height: NOISE_TEX_SIZE,
            depth_or_array_layers: NOISE_TEX_SIZE,
        };
        let texture = ctx.device.create_texture_with_data(
            &ctx.queue,
            &wgpu::TextureDescriptor {
                label: Some("disk_noise_3d_tex"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &bytes,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("disk_noise_3d_view"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("disk_noise_3d_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }
}
