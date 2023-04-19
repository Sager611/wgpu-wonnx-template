use std::iter;
use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use cgmath::prelude::*;
use wgpu::util::DeviceExt;
use winit::{event::*, window::Window};

use crate::camera;
use crate::model;
use crate::nn::{load_wonnx_session, classify_image};
use crate::resources;
use crate::texture;

use camera::CameraUniform;
use model::{DrawLight, DrawModel, Vertex};

pub struct Instance {
  pub position: cgmath::Vector3<f32>,
  pub rotation: cgmath::Quaternion<f32>,
}

impl Instance {
  pub fn to_raw(&self) -> InstanceRaw {
    InstanceRaw {
      model: (cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation)).into(),
      normal: cgmath::Matrix3::from(self.rotation).into(),
    }
  }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
pub struct InstanceRaw {
  model: [[f32; 4]; 4],
  normal: [[f32; 3]; 3],
}

impl model::Vertex for InstanceRaw {
  fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
    use std::mem;
    wgpu::VertexBufferLayout {
      array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
      // We need to switch from using a step mode of Vertex to Instance
      // This means that our shaders will only change to use the next
      // instance when the shader starts processing a new instance
      step_mode: wgpu::VertexStepMode::Instance,
      attributes: &[
        wgpu::VertexAttribute {
          offset: 0,
          // Our vertex shader only uses locations 0, and 1
          shader_location: 5,
          format: wgpu::VertexFormat::Float32x4,
        },
        // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
        // for each vec4. We don't have to do this in code though.
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
          shader_location: 6,
          format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
          shader_location: 7,
          format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
          shader_location: 8,
          format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
          shader_location: 9,
          format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
          shader_location: 10,
          format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
          offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
          shader_location: 11,
          format: wgpu::VertexFormat::Float32x3,
        },
      ],
    }
  }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
  position: [f32; 3],
  // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
  _padding: u32,
  color: [f32; 3],
  _padding2: u32,
}

pub struct State {
  window: Arc<Window>,
  surface: wgpu::Surface,

  // we have to share GPU with wonnx, so we use an Rc
  device: wgpu::Device,
  queue: wgpu::Queue,

  pub config: wgpu::SurfaceConfiguration,
  pub render_pipeline: wgpu::RenderPipeline,
  pub obj_model: model::Model,
  pub camera: camera::Camera,
  pub projection: camera::Projection,
  pub camera_controller: camera::CameraController,
  pub camera_uniform: CameraUniform,
  pub camera_buffer: wgpu::Buffer,
  pub camera_bind_group: wgpu::BindGroup,
  pub instances: Vec<Instance>,
  #[allow(dead_code)]
  pub instance_buffer: wgpu::Buffer,
  pub depth_texture: texture::Texture,
  pub size: winit::dpi::PhysicalSize<u32>,
  pub light_uniform: LightUniform,
  pub light_buffer: wgpu::Buffer,
  pub light_bind_group: wgpu::BindGroup,
  pub light_render_pipeline: wgpu::RenderPipeline,
  #[allow(dead_code)]
  pub debug_material: model::Material,
  pub mouse_pressed: bool,
  time0: instant::Instant,
  nn_session: Arc<wonnx::Session>,
}

fn create_render_pipeline(
  device: &wgpu::Device,
  layout: &wgpu::PipelineLayout,
  color_format: wgpu::TextureFormat,
  depth_format: Option<wgpu::TextureFormat>,
  vertex_layouts: &[wgpu::VertexBufferLayout],
  shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
  let shader = device.create_shader_module(shader);

  device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    label: Some(&format!("{:?}", shader)),
    layout: Some(layout),
    vertex: wgpu::VertexState {
      module: &shader,
      entry_point: "vs_main",
      buffers: vertex_layouts,
    },
    fragment: Some(wgpu::FragmentState {
      module: &shader,
      entry_point: "fs_main",
      targets: &[Some(wgpu::ColorTargetState {
        format: color_format,
        blend: Some(wgpu::BlendState {
          alpha: wgpu::BlendComponent::REPLACE,
          color: wgpu::BlendComponent::REPLACE,
        }),
        write_mask: wgpu::ColorWrites::ALL,
      })],
    }),
    primitive: wgpu::PrimitiveState {
      topology: wgpu::PrimitiveTopology::TriangleList,
      strip_index_format: None,
      front_face: wgpu::FrontFace::Ccw,
      cull_mode: Some(wgpu::Face::Back),
      // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
      polygon_mode: wgpu::PolygonMode::Fill,
      // Requires Features::DEPTH_CLIP_CONTROL
      unclipped_depth: false,
      // Requires Features::CONSERVATIVE_RASTERIZATION
      conservative: false,
    },
    depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
      format,
      depth_write_enabled: true,
      depth_compare: wgpu::CompareFunction::Less,
      stencil: wgpu::StencilState::default(),
      bias: wgpu::DepthBiasState::default(),
    }),
    multisample: wgpu::MultisampleState {
      count: 1,
      mask: !0,
      alpha_to_coverage_enabled: false,
    },
    // If the pipeline will be used with a multiview render pass, this
    // indicates how many array layers the attachments will have.
    multiview: None,
  })
}

impl State {
  pub async fn new(window: Arc<Window>) -> Self {
    let window_size = window.inner_size();

    // The instance is a handle to our GPU
    // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
    //
    // Honoring WGPU_BACKEND environment variable on native
    let backends = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::all());
    let dx12_shader_compiler = wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default();
    let wgpu_instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
      backends,
      dx12_shader_compiler,
    });

    let backends = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::all());
    // Surface
    //
    // The surface needs to live as long as the window that created it.
    let surface = unsafe { wgpu_instance.create_surface(window.as_ref()) }.unwrap();

    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&wgpu_instance, backends, Some(&surface))
      .await
      .expect("No suitable GPU adapters found on the system!");

    let device_descriptor = wgpu::DeviceDescriptor {
          label: None,
          //features: wgpu::Features::all_webgpu_mask(),
          //features: adapter.features(),
          features: wgpu::Features::empty(),
          // WebGL doesn't support all of wgpu's features, so if
          // we're building for the web we'll have to disable some.
          limits: if cfg!(target_arch = "wasm32") {
            // we need loose enough limits so that compute shaders are included
            // so that `wonnx` works
            wgpu::Limits::downlevel_defaults()
            //wgpu::Limits::downlevel_webgl2_defaults()
          } else {
            wgpu::Limits::default()
          },
        };
    //let adapter_features = adapter.features();
    //log::info!("Adapter features: {:?}", adapter_features);
    let (device, queue) = adapter
      .request_device(
        &device_descriptor,
        None, // Trace path
      )
      .await
      .expect("Unable to request GPU device");

    //log::info!("Adapter limits: {:?}", adapter.limits());

    let surface_config = surface
      .get_default_config(&adapter, window_size.width, window_size.height)
      .expect("Surface isn't supported by the adapter.");

    // TODO: some wasm-bindgen/wgpu shenanigans prevent this from working, in js `surface` becomes
    // a GPUDevice in this line.
    surface.configure(&device, &surface_config);

    log::info!("Surface Capabilities: {:?}", surface.get_capabilities(&adapter));

    // neural network session
    let nn_session = load_wonnx_session().await.expect("Couldn't load wonnx session");

    // Texture
    let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      entries: &[
        wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::FRAGMENT,
          ty: wgpu::BindingType::Texture {
            multisampled: false,
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
          },
          count: None,
        },
        wgpu::BindGroupLayoutEntry {
          binding: 1,
          visibility: wgpu::ShaderStages::FRAGMENT,
          ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
          count: None,
        },
        // normal map
        wgpu::BindGroupLayoutEntry {
          binding: 2,
          visibility: wgpu::ShaderStages::FRAGMENT,
          ty: wgpu::BindingType::Texture {
            multisampled: false,
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
          },
          count: None,
        },
        wgpu::BindGroupLayoutEntry {
          binding: 3,
          visibility: wgpu::ShaderStages::FRAGMENT,
          ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
          count: None,
        },
      ],
      label: Some("texture_bind_group_layout"),
    });

    // Camera
    let camera = camera::Camera::new((0.0, 5.0, 10.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
    let projection = camera::Projection::new(surface_config.width, surface_config.height, cgmath::Deg(45.0), 0.1, 100.0);
    let camera_controller = camera::CameraController::new(4.0, 0.4);

    let mut camera_uniform = CameraUniform::new();
    camera_uniform.update_view_proj(&camera, &projection);

    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Camera Buffer"),
      contents: bytemuck::cast_slice(&[camera_uniform]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Initial rotation and pos of objects
    let instances = vec![Instance {
      position: cgmath::Vector3::new(0.0, 0.0, 0.0),
      rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0)),
    }];

    // Data buffer for shader
    let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Instance Buffer"),
      contents: bytemuck::cast_slice(&instance_data),
      usage: wgpu::BufferUsages::VERTEX,
    });

    // Camera BindGroupLayout and BindGroup for wgsl shader file
    let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
          ty: wgpu::BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      }],
      label: Some("camera_bind_group_layout"),
    });

    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &camera_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: camera_buffer.as_entire_binding(),
      }],
      label: Some("camera_bind_group"),
    });

    // Load OBJ model
    let obj_model = resources::load_model("suzanne.obj", &device, &queue, &texture_bind_group_layout)
      .await
      .unwrap();

    // Light
    let light_uniform = LightUniform {
      position: [2.0, 2.0, 2.0],
      _padding: 0,
      color: [1.0, 1.0, 1.0],
      _padding2: 0,
    };

    let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Light VB"),
      contents: bytemuck::cast_slice(&[light_uniform]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let light_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
          ty: wgpu::BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      }],
      label: None,
    });

    let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &light_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: light_buffer.as_entire_binding(),
      }],
      label: None,
    });

    // Depth texture
    let depth_texture = texture::Texture::create_depth_texture(&device, &surface_config, "depth_texture");

    // Render Pipeline
    let render_pipeline = {
      let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout, &light_bind_group_layout],
        push_constant_ranges: &[],
      });
      let shader = wgpu::ShaderModuleDescriptor {
        label: Some("Normal Shader"),
        // we inject the shader file in the code as a str using a macro
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
      };
      create_render_pipeline(
        &device,
        &layout,
        surface_config.format,
        Some(texture::Texture::DEPTH_FORMAT),
        &[model::ModelVertex::desc(), InstanceRaw::desc()],
        shader,
      )
    };

    // Light Render Pipeline
    let light_render_pipeline = {
      let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Light Pipeline Layout"),
        bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
        push_constant_ranges: &[],
      });
      let shader = wgpu::ShaderModuleDescriptor {
        label: Some("Light Shader"),
        // we inject the shader file in the code as a str using a macro
        source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
      };
      create_render_pipeline(
        &device,
        &layout,
        surface_config.format,
        Some(texture::Texture::DEPTH_FORMAT),
        &[model::ModelVertex::desc()],
        shader,
      )
    };

    let debug_material = {
      // inject bytes from files directly in-code
      let diffuse_bytes = include_bytes!("../res/cobble-diffuse.png");
      let normal_bytes = include_bytes!("../res/cobble-normal.png");

      let diffuse_texture = texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "res/alt-diffuse.png", false).unwrap();
      let normal_texture = texture::Texture::from_bytes(&device, &queue, normal_bytes, "res/alt-normal.png", true).unwrap();

      model::Material::new(&device, "alt-material", diffuse_texture, normal_texture, &texture_bind_group_layout)
    };

    // start time
    let time0 = instant::Instant::now();

    Self {
      window,
      surface,
      device,
      queue,
      config: surface_config,
      render_pipeline,
      obj_model,
      camera,
      projection,
      camera_controller,
      camera_buffer,
      camera_bind_group,
      camera_uniform,
      instances,
      instance_buffer,
      depth_texture,
      size: window_size,
      light_uniform,
      light_buffer,
      light_bind_group,
      light_render_pipeline,
      #[allow(dead_code)]
      debug_material,
      mouse_pressed: false,
      time0,
      nn_session,
    }
  }

  pub fn window(&self) -> &Window {
    &self.window
  }

  pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
    if new_size.width > 0 && new_size.height > 0 {
      self.projection.resize(new_size.width, new_size.height);
      self.size = new_size;
      self.config.width = new_size.width;
      self.config.height = new_size.height;
      self.surface.configure(&self.device, &self.config);
      self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
    }
  }

  pub fn input(&mut self, event: &WindowEvent) -> bool {
    match event {
      WindowEvent::KeyboardInput {
        input: KeyboardInput {
          virtual_keycode: Some(key),
          state,
          ..
        },
        ..
      } => match key {
        VirtualKeyCode::Return => {
          match state {
            ElementState::Released => self.classify_screenshot(),
            _ => true
          }
        }
        _ => self.camera_controller.process_keyboard(*key, *state),
      },
      WindowEvent::MouseWheel { delta, .. } => {
        self.camera_controller.process_scroll(delta);
        true
      }
      WindowEvent::MouseInput {
        button: MouseButton::Left,
        state,
        ..
      } => {
        self.mouse_pressed = *state == ElementState::Pressed;
        true
      }
      _ => false,
    }
  }

  pub fn update(&mut self, dt: std::time::Duration) {
    // elapsed time since start
    let t: f32 = self.time0.elapsed().as_secs_f32();

    // Update camera's perpective
    self.camera_controller.update_camera(&mut self.camera, dt);
    self.camera_uniform.update_view_proj(&self.camera, &self.projection);

    // write to buffer
    self
      .queue
      .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));

    // Update the light's pos
    let s = cgmath::Angle::sin(cgmath::Rad(t));
    let s_2 = cgmath::Angle::sin(cgmath::Rad(2. * t));
    let s_3 = cgmath::Angle::sin(cgmath::Rad(3. * t));
    self.light_uniform.position = [2. * s_2, 2. * s, 2. * s_3];

    // write to buffer
    self
      .queue
      .write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[self.light_uniform]));
  }

  pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
    let output = self.surface.get_current_texture()?;
    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
      label: Some("Render Encoder"),
    });

    {
      let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Render Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
          view: &view,
          resolve_target: None,
          ops: wgpu::Operations {
            // background color
            load: wgpu::LoadOp::Clear(wgpu::Color {
              r: 0.0,
              g: 0.0,
              b: 0.0,
              a: 1.0,
            }),
            store: true,
          },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
          view: &self.depth_texture.view,
          depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(1.0),
            store: true,
          }),
          stencil_ops: None,
        }),
      });

      render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
      render_pass.set_pipeline(&self.light_render_pipeline);
      render_pass.draw_light_model(&self.obj_model, &self.camera_bind_group, &self.light_bind_group);

      render_pass.set_pipeline(&self.render_pipeline);
      render_pass.draw_model_instanced(
        &self.obj_model,
        0..self.instances.len() as u32,
        &self.camera_bind_group,
        &self.light_bind_group,
      );
    }
    self.queue.submit(iter::once(encoder.finish()));
    output.present();

    Ok(())
  }

  pub fn screenshot(&mut self) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>>  {
    // See https://github.com/onnx/models/blob/master/vision/classification/imagenet_inference.ipynb
    // for pre-processing image.
    // Final size: (1, 3, 224, 224)
    // TODO: get image screenshot
    let image = ndarray::Array::from_shape_fn((1, 3, 224, 224), |(_, _, _, _)| {
      // range [-1, 1]
      0.0f32
    });

    image
  }

  pub fn classify_screenshot(&mut self) -> bool {
      let image = self.screenshot();
      let _future = classify_image(&self.nn_session, image.as_slice().unwrap().into());

      let _res = pollster::block_on(future);

      true
  }
}
