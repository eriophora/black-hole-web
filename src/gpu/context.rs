use std::sync::Arc;

use wgpu::{Adapter, AdapterInfo, Backends, Device, Instance, Queue, Surface};
use winit::window::Window;

pub struct GpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub adapter_info: AdapterInfo,
    pub device: Device,
    pub queue: Queue,
}

impl GpuContext {
    pub async fn new_with_window(window: Arc<Window>) -> Result<(Self, Surface<'static>), String> {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::BROWSER_WEBGPU,
            ..Default::default()
        });
        let surface = instance
            .create_surface(window)
            .map_err(|e| format!("Failed to create WebGPU surface: {}", e))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                "No compatible WebGPU adapter found. Please use a WebGPU-enabled browser (Chrome, Edge, Brave, or Safari 18+) with hardware acceleration enabled.".to_string()
            })?;

        let adapter_info = adapter.get_info();
        log::info!(
            "Selected WebGPU Adapter: '{}' | Backend: {:?} | Type: {:?}",
            adapter_info.name,
            adapter_info.backend,
            adapter_info.device_type
        );

        let limits = wgpu::Limits::default().using_resolution(adapter.limits());
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("black_hole_web_gpu_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create WebGPU device: {}", e))?;

        Ok((
            Self {
                instance,
                adapter,
                adapter_info,
                device,
                queue,
            },
            surface,
        ))
    }
}
