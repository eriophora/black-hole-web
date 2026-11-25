use std::array;
use std::io::Cursor;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::context::GpuContext;
use super::noise_tex::DiskNoiseTexture;
use super::shader_loader::ShaderLoader;
use super::star_grid::{build_star_catalog, StarCatalogGpuBuffers, StarCatalogGpuData};
use super::uniforms::{FrameUniforms, FLAG_ENABLE_BLOOM};

const BLOOM_MIPS: usize = 6;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BlitParams {
    swapchain_is_srgb: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BloomParams {
    src_texel: [f32; 2],
    dst_size: [u32; 2],
    is_first_pass: u32,
    threshold: f32,
    knee: f32,
    scatter: f32,
}

struct RenderTargets {
    width: u32,
    height: u32,
    _hdr_raw_tex: wgpu::Texture,
    hdr_raw_view: wgpu::TextureView,
    _hdr_tex: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    _bloom_down_tex: [wgpu::Texture; BLOOM_MIPS],
    bloom_down_views: [wgpu::TextureView; BLOOM_MIPS],
    _bloom_up_tex: [wgpu::Texture; BLOOM_MIPS - 1],
    bloom_up_views: [wgpu::TextureView; BLOOM_MIPS - 1],
    bloom_sizes: [(u32, u32); BLOOM_MIPS],
    display_tex: wgpu::Texture,
    display_view: wgpu::TextureView,
}

struct BloomBindGroups {
    down_bgs: [wgpu::BindGroup; BLOOM_MIPS],
    up_bgs: [wgpu::BindGroup; BLOOM_MIPS - 1],
    post_bg: wgpu::BindGroup,
}

pub struct ComputeRenderPipeline {
    uniform_buf: wgpu::Buffer,
    star_catalog: StarCatalogGpuData,
    star_buffers: StarCatalogGpuBuffers,
    noise_tex: DiskNoiseTexture,
    compute_bgl: wgpu::BindGroupLayout,
    photon_ssaa_bgl: wgpu::BindGroupLayout,
    bloom_down_bgl: wgpu::BindGroupLayout,
    bloom_up_bgl: wgpu::BindGroupLayout,
    postprocess_bgl: wgpu::BindGroupLayout,
    blit_bgl: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
    photon_ssaa_pipeline: wgpu::ComputePipeline,
    bloom_down_pipeline: wgpu::ComputePipeline,
    bloom_up_pipeline: wgpu::ComputePipeline,
    postprocess_pipeline: wgpu::ComputePipeline,
    blit_pipeline: Option<wgpu::RenderPipeline>,
    blit_params_buf: wgpu::Buffer,
    blit_sampler: wgpu::Sampler,
    linear_clamp_sampler: wgpu::Sampler,
    bloom_down_params_bufs: [wgpu::Buffer; BLOOM_MIPS],
    bloom_up_params_bufs: [wgpu::Buffer; BLOOM_MIPS - 1],
    swapchain_format: Option<wgpu::TextureFormat>,
    targets: RenderTargets,
    compute_bg: wgpu::BindGroup,
    photon_ssaa_bg: wgpu::BindGroup,
    bloom_bgs: BloomBindGroups,
    blit_bg: wgpu::BindGroup,
}

impl ComputeRenderPipeline {
    pub async fn new(
        ctx: &GpuContext,
        width: u32,
        height: u32,
        swapchain_format: Option<wgpu::TextureFormat>,
    ) -> Result<Self, String> {
        let device = &ctx.device;
        let w = width.max(16);
        let h = height.max(16);

        let initial_uniforms = FrameUniforms::zeroed();
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("frame_uniforms_buf"),
            contents: bytemuck::bytes_of(&initial_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let star_catalog = build_star_catalog();
        let star_buffers = StarCatalogGpuBuffers::new(ctx, &star_catalog);
        let noise_tex = DiskNoiseTexture::new(ctx);

        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                storage_2d_write_entry(1, wgpu::TextureFormat::Rgba16Float),
                storage_ro_entry(2),
                storage_ro_entry(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let photon_ssaa_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("photon_ssaa_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                tex_2d_entry(1),
                storage_2d_write_entry(2, wgpu::TextureFormat::Rgba16Float),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bloom_down_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom_down_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                tex_2d_entry(1),
                sampler_entry(2),
                storage_2d_write_entry(3, wgpu::TextureFormat::Rgba16Float),
            ],
        });

        let bloom_up_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom_up_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                tex_2d_entry(1),
                tex_2d_entry(2),
                sampler_entry(3),
                storage_2d_write_entry(4, wgpu::TextureFormat::Rgba16Float),
            ],
        });

        let postprocess_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("postprocess_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                tex_2d_entry(1),
                tex_2d_entry(2),
                sampler_entry(3),
                storage_2d_write_entry(4, wgpu::TextureFormat::Rgba8Unorm),
            ],
        });

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let is_srgb = swapchain_format.map(|f| f.is_srgb()).unwrap_or(false) as u32;
        let blit_params = BlitParams {
            swapchain_is_srgb: is_srgb,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let blit_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blit_params_buf"),
            contents: bytemuck::bytes_of(&blit_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let linear_clamp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear_clamp_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bloom_down_params_bufs = array::from_fn(|i| {
            let zero = BloomParams::zeroed();
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("bloom_down_params_{}", i)),
                contents: bytemuck::bytes_of(&zero),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });
        let bloom_up_params_bufs = array::from_fn(|i| {
            let zero = BloomParams::zeroed();
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("bloom_up_params_{}", i)),
                contents: bytemuck::bytes_of(&zero),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });

        let compute_pipeline = compile_compute_shader(
            device,
            "raytrace.comp.wgsl",
            &compute_bgl,
            "raytrace_compute",
        )
        .await?;
        let photon_ssaa_pipeline = compile_compute_shader(
            device,
            "photon_ssaa.comp.wgsl",
            &photon_ssaa_bgl,
            "photon_ssaa",
        )
        .await?;
        let bloom_down_pipeline = compile_compute_shader(
            device,
            "bloom_down.comp.wgsl",
            &bloom_down_bgl,
            "bloom_down",
        )
        .await?;
        let bloom_up_pipeline = compile_compute_shader(
            device,
            "bloom_up.comp.wgsl",
            &bloom_up_bgl,
            "bloom_up",
        )
        .await?;
        let postprocess_pipeline = compile_compute_shader(
            device,
            "postprocess.comp.wgsl",
            &postprocess_bgl,
            "postprocess",
        )
        .await?;
        let blit_pipeline = if let Some(fmt) = swapchain_format {
            Some(compile_blit_pipeline(device, &blit_bgl, fmt).await?)
        } else {
            None
        };

        let targets = create_render_targets(device, w, h);
        write_bloom_params_buffers(
            &ctx.queue,
            &targets,
            &bloom_down_params_bufs,
            &bloom_up_params_bufs,
        );

        let (compute_bg, photon_ssaa_bg, bloom_bgs, blit_bg) = create_bind_groups(
            device,
            &targets,
            &uniform_buf,
            &star_buffers,
            &noise_tex,
            &linear_clamp_sampler,
            &bloom_down_params_bufs,
            &bloom_up_params_bufs,
            &blit_params_buf,
            &blit_sampler,
            &compute_bgl,
            &photon_ssaa_bgl,
            &bloom_down_bgl,
            &bloom_up_bgl,
            &postprocess_bgl,
            &blit_bgl,
        );

        Ok(Self {
            uniform_buf,
            star_catalog,
            star_buffers,
            noise_tex,
            compute_bgl,
            photon_ssaa_bgl,
            bloom_down_bgl,
            bloom_up_bgl,
            postprocess_bgl,
            blit_bgl,
            compute_pipeline,
            photon_ssaa_pipeline,
            bloom_down_pipeline,
            bloom_up_pipeline,
            postprocess_pipeline,
            blit_pipeline,
            blit_params_buf,
            blit_sampler,
            linear_clamp_sampler,
            bloom_down_params_bufs,
            bloom_up_params_bufs,
            swapchain_format,
            targets,
            compute_bg,
            photon_ssaa_bg,
            bloom_bgs,
            blit_bg,
        })
    }

    pub fn star_catalog(&self) -> &StarCatalogGpuData {
        &self.star_catalog
    }

    pub fn size(&self) -> (u32, u32) {
        (self.targets.width, self.targets.height)
    }

    pub fn swapchain_format(&self) -> Option<wgpu::TextureFormat> {
        self.swapchain_format
    }

    pub fn resize(&mut self, ctx: &GpuContext, width: u32, height: u32) {
        let w = width.max(16);
        let h = height.max(16);
        if w == self.targets.width && h == self.targets.height {
            return;
        }
        self.targets = create_render_targets(&ctx.device, w, h);
        write_bloom_params_buffers(
            &ctx.queue,
            &self.targets,
            &self.bloom_down_params_bufs,
            &self.bloom_up_params_bufs,
        );
        let (cbg, pbg, bloom_bgs, bbg) = create_bind_groups(
            &ctx.device,
            &self.targets,
            &self.uniform_buf,
            &self.star_buffers,
            &self.noise_tex,
            &self.linear_clamp_sampler,
            &self.bloom_down_params_bufs,
            &self.bloom_up_params_bufs,
            &self.blit_params_buf,
            &self.blit_sampler,
            &self.compute_bgl,
            &self.photon_ssaa_bgl,
            &self.bloom_down_bgl,
            &self.bloom_up_bgl,
            &self.postprocess_bgl,
            &self.blit_bgl,
        );
        self.compute_bg = cbg;
        self.photon_ssaa_bg = pbg;
        self.bloom_bgs = bloom_bgs;
        self.blit_bg = bbg;
    }

    pub fn render_compute_frame(&mut self, ctx: &GpuContext, uniforms: &FrameUniforms) {
        ctx.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniforms));

        let wg_x = self.targets.width.div_ceil(8);
        let wg_y = self.targets.height.div_ceil(8);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("awe_compute_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("awe_raytrace_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &self.compute_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("awe_photon_ssaa_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.photon_ssaa_pipeline);
            pass.set_bind_group(0, &self.photon_ssaa_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        if (uniforms.flags & FLAG_ENABLE_BLOOM) != 0 {
            for i in 0..BLOOM_MIPS {
                let (dw, dh) = self.targets.bloom_sizes[i];
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("awe_bloom_down_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.bloom_down_pipeline);
                pass.set_bind_group(0, &self.bloom_bgs.down_bgs[i], &[]);
                pass.dispatch_workgroups(dw.div_ceil(8), dh.div_ceil(8), 1);
            }

            for i in (0..(BLOOM_MIPS - 1)).rev() {
                let (dw, dh) = self.targets.bloom_sizes[i];
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("awe_bloom_up_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.bloom_up_pipeline);
                pass.set_bind_group(0, &self.bloom_bgs.up_bgs[i], &[]);
                pass.dispatch_workgroups(dw.div_ceil(8), dh.div_ceil(8), 1);
            }
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("awe_postprocess_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.postprocess_pipeline);
            pass.set_bind_group(0, &self.bloom_bgs.post_bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn encode_blit<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if let Some(ref pipe) = self.blit_pipeline {
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &self.blit_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    pub fn capture_display_png_async(
        &self,
        ctx: &GpuContext,
        on_done: impl FnOnce(Result<Vec<u8>, String>) + 'static,
    ) {
        let width = self.targets.width;
        let height = self.targets.height;
        let bytes_per_pixel = 4u32;
        let unpadded_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = unpadded_row.div_ceil(align) * align;
        let total_size = padded_row as u64 * height as u64;

        let staging = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("web_snapshot_staging"),
            size: total_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("web_snapshot_encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.targets.display_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue.submit(std::iter::once(encoder.finish()));

        let staging_cb = Arc::clone(&staging);
        staging.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            let out = (|| -> Result<Vec<u8>, String> {
                res.map_err(|e| format!("GPU buffer map failed: {:?}", e))?;
                let mapped = staging_cb.slice(..).get_mapped_range();
                let mut rgba = vec![0u8; (unpadded_row * height) as usize];
                for row in 0..height as usize {
                    let src_start = row * padded_row as usize;
                    let dst_start = row * unpadded_row as usize;
                    rgba[dst_start..dst_start + unpadded_row as usize]
                        .copy_from_slice(&mapped[src_start..src_start + unpadded_row as usize]);
                }
                drop(mapped);
                staging_cb.unmap();

                let img = image::RgbaImage::from_raw(width, height, rgba)
                    .ok_or_else(|| "Failed to build RgbaImage from GPU readback".to_string())?;
                let mut png_bytes = Vec::new();
                img.write_to(&mut Cursor::new(&mut png_bytes), image::ImageFormat::Png)
                    .map_err(|e| format!("PNG encode failed: {}", e))?;
                Ok(png_bytes)
            })();
            on_done(out);
        });
    }
}

fn uniform_bgl_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_ro_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_2d_write_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn tex_2d_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

async fn compile_compute_shader(
    device: &wgpu::Device,
    entry_rel_path: &str,
    bgl: &wgpu::BindGroupLayout,
    label: &str,
) -> Result<wgpu::ComputePipeline, String> {
    let loaded = ShaderLoader::load_embedded(entry_rel_path)?;
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry_rel_path),
        source: wgpu::ShaderSource::Wgsl(loaded.source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{}_layout", label)),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{}_pipeline", label)),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(err) = device.pop_error_scope().await {
        return Err(format!(
            "WGSL compilation error in '{}': {}",
            entry_rel_path, err
        ));
    }
    Ok(pipe)
}

async fn compile_blit_pipeline(
    device: &wgpu::Device,
    blit_bgl: &wgpu::BindGroupLayout,
    target_format: wgpu::TextureFormat,
) -> Result<wgpu::RenderPipeline, String> {
    let loaded = ShaderLoader::load_embedded("blit.wgsl")?;
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("blit.wgsl"),
        source: wgpu::ShaderSource::Wgsl(loaded.source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("blit_layout"),
        bind_group_layouts: &[blit_bgl],
        push_constant_ranges: &[],
    });
    let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("blit_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });
    if let Some(err) = device.pop_error_scope().await {
        return Err(format!("WGSL compilation error in 'blit.wgsl': {}", err));
    }
    Ok(pipe)
}

fn create_rgba16f_tex(device: &wgpu::Device, label: &str, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn create_render_targets(device: &wgpu::Device, width: u32, height: u32) -> RenderTargets {
    let (hdr_raw_tex, hdr_raw_view) = create_rgba16f_tex(device, "awe_hdr_raw_tex", width, height);
    let (hdr_tex, hdr_view) = create_rgba16f_tex(device, "awe_hdr_tex", width, height);

    let bloom_sizes: [(u32, u32); BLOOM_MIPS] = array::from_fn(|i| {
        let mw = (width >> (i + 1)).max(1);
        let mh = (height >> (i + 1)).max(1);
        (mw, mh)
    });

    let down_pairs: [(wgpu::Texture, wgpu::TextureView); BLOOM_MIPS] = array::from_fn(|i| {
        let (mw, mh) = bloom_sizes[i];
        create_rgba16f_tex(device, &format!("awe_bloom_down_{}", i), mw, mh)
    });
    let up_pairs: [(wgpu::Texture, wgpu::TextureView); BLOOM_MIPS - 1] = array::from_fn(|i| {
        let (mw, mh) = bloom_sizes[i];
        create_rgba16f_tex(device, &format!("awe_bloom_up_{}", i), mw, mh)
    });

    let bloom_down_tex = down_pairs.each_ref().map(|(t, _)| t.clone());
    let bloom_down_views = down_pairs.map(|(_, v)| v);
    let bloom_up_tex = up_pairs.each_ref().map(|(t, _)| t.clone());
    let bloom_up_views = up_pairs.map(|(_, v)| v);

    let display_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("awe_display_tex"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let display_view = display_tex.create_view(&wgpu::TextureViewDescriptor::default());

    RenderTargets {
        width,
        height,
        _hdr_raw_tex: hdr_raw_tex,
        hdr_raw_view,
        _hdr_tex: hdr_tex,
        hdr_view,
        _bloom_down_tex: bloom_down_tex,
        bloom_down_views,
        _bloom_up_tex: bloom_up_tex,
        bloom_up_views,
        bloom_sizes,
        display_tex,
        display_view,
    }
}

fn write_bloom_params_buffers(
    queue: &wgpu::Queue,
    targets: &RenderTargets,
    down_bufs: &[wgpu::Buffer; BLOOM_MIPS],
    up_bufs: &[wgpu::Buffer; BLOOM_MIPS - 1],
) {
    for i in 0..BLOOM_MIPS {
        let (sw, sh) = if i == 0 {
            (targets.width, targets.height)
        } else {
            targets.bloom_sizes[i - 1]
        };
        let (dw, dh) = targets.bloom_sizes[i];
        let params = BloomParams {
            src_texel: [1.0 / sw.max(1) as f32, 1.0 / sh.max(1) as f32],
            dst_size: [dw, dh],
            is_first_pass: if i == 0 { 1 } else { 0 },
            threshold: 0.95,
            knee: 0.40,
            scatter: 0.88,
        };
        queue.write_buffer(&down_bufs[i], 0, bytemuck::bytes_of(&params));
    }

    for i in 0..(BLOOM_MIPS - 1) {
        let (sw, sh) = targets.bloom_sizes[i + 1];
        let (dw, dh) = targets.bloom_sizes[i];
        let params = BloomParams {
            src_texel: [1.0 / sw.max(1) as f32, 1.0 / sh.max(1) as f32],
            dst_size: [dw, dh],
            is_first_pass: 0,
            threshold: 1.0,
            knee: 0.5,
            scatter: 0.85,
        };
        queue.write_buffer(&up_bufs[i], 0, bytemuck::bytes_of(&params));
    }
}

#[allow(clippy::too_many_arguments)]
fn create_bind_groups(
    device: &wgpu::Device,
    targets: &RenderTargets,
    uniform_buf: &wgpu::Buffer,
    star_buffers: &StarCatalogGpuBuffers,
    noise_tex: &DiskNoiseTexture,
    linear_clamp_sampler: &wgpu::Sampler,
    bloom_down_params_bufs: &[wgpu::Buffer; BLOOM_MIPS],
    bloom_up_params_bufs: &[wgpu::Buffer; BLOOM_MIPS - 1],
    blit_params_buf: &wgpu::Buffer,
    blit_sampler: &wgpu::Sampler,
    compute_bgl: &wgpu::BindGroupLayout,
    photon_ssaa_bgl: &wgpu::BindGroupLayout,
    bloom_down_bgl: &wgpu::BindGroupLayout,
    bloom_up_bgl: &wgpu::BindGroupLayout,
    postprocess_bgl: &wgpu::BindGroupLayout,
    blit_bgl: &wgpu::BindGroupLayout,
) -> (wgpu::BindGroup, wgpu::BindGroup, BloomBindGroups, wgpu::BindGroup) {
    let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_bg"),
        layout: compute_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&targets.hdr_raw_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: star_buffers.star_cells_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: star_buffers.star_entries_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&noise_tex.view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&noise_tex.sampler),
            },
        ],
    });

    let photon_ssaa_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("photon_ssaa_bg"),
        layout: photon_ssaa_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&targets.hdr_raw_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&targets.hdr_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&noise_tex.view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&noise_tex.sampler),
            },
        ],
    });

    let down_bgs = array::from_fn(|i| {
        let src_view = if i == 0 {
            &targets.hdr_view
        } else {
            &targets.bloom_down_views[i - 1]
        };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("bloom_down_bg_{}", i)),
            layout: bloom_down_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bloom_down_params_bufs[i].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(linear_clamp_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&targets.bloom_down_views[i]),
                },
            ],
        })
    });

    let up_bgs = array::from_fn(|i| {
        let low_view = if i == BLOOM_MIPS - 2 {
            &targets.bloom_down_views[BLOOM_MIPS - 1]
        } else {
            &targets.bloom_up_views[i + 1]
        };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("bloom_up_bg_{}", i)),
            layout: bloom_up_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bloom_up_params_bufs[i].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(low_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&targets.bloom_down_views[i]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(linear_clamp_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&targets.bloom_up_views[i]),
                },
            ],
        })
    });

    let post_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("postprocess_bg"),
        layout: postprocess_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&targets.hdr_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&targets.bloom_up_views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(linear_clamp_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&targets.display_view),
            },
        ],
    });

    let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blit_bg"),
        layout: blit_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&targets.display_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(blit_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: blit_params_buf.as_entire_binding(),
            },
        ],
    });

    (
        compute_bg,
        photon_ssaa_bg,
        BloomBindGroups {
            down_bgs,
            up_bgs,
            post_bg,
        },
        blit_bg,
    )
}
