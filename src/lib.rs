use std::sync::Arc;

use winit::{
  event::*,
  event_loop::{ControlFlow, EventLoop},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use winit::dpi::PhysicalSize;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator, which reduces code size but it is less efficient.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

mod camera;
mod model;
mod nn;
mod resources;
mod state;
mod texture;

use state::State;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub async fn run() {
  //// SETUP ////

  // Set logger backend depending on architecture
  cfg_if::cfg_if! {
      if #[cfg(target_arch = "wasm32")] {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).expect("Could't initialize logger");
      } else {
        env_logger::init();
      }
  }

  log::info!("Starting ..");

  let event_loop = EventLoop::new();
  let title = env!("CARGO_PKG_NAME");
  // window is a shared pointer since we will access the window's set functions from other threads
  let window = Arc::from(winit::window::WindowBuilder::new().with_title(title).build(&event_loop).unwrap());

  // auto-size when web page window size changes (closure is called async)
  // Adapted from: https://github.com/rust-windowing/winit/pull/2074/files
  #[cfg(target_arch = "wasm32")]
  let resize_closure = {
    let window = Arc::clone(&window);
    Closure::wrap(Box::new(move |entries: js_sys::Array| {
      let resizes: Vec<_> = entries
        .iter()
        .map(|entry| {
          let entry: web_sys::ResizeObserverEntry = entry
            .dyn_into()
            .expect("`ResizeObserver` callback not called with array of `ResizeObserverEntry`");

          let rect = entry.content_rect();
          let physical_size = PhysicalSize::new(rect.width(), rect.height());
          log::info!("Observed resize. Rect: {}x{}", physical_size.width, physical_size.height);

          // window is a cloned Arc pointer since closure will be called async from another thread
          window.set_inner_size(physical_size);
        })
        .collect();
    }) as Box<dyn FnMut(_)>)
  };

  #[cfg(target_arch = "wasm32")]
  {
    // Winit does not support sizing with CSS, so we have to set
    // the size manually when on web.
    let global_window = web_sys::window().expect("No global `window` exists");
    window.set_inner_size(PhysicalSize::new(
      global_window.inner_width().unwrap().as_f64().expect("Couldn't convert js value to `f64`"),
      global_window
        .inner_height()
        .unwrap()
        .as_f64()
        .expect("Couldn't convert js value to `f64`"),
    ));

    // Create canvas (import required to include canvas() function)
    use winit::platform::web::WindowExtWebSys;
    let doc = global_window.document().expect("Couldn't retrieve document");
    let dst = doc
      .get_element_by_id("wasm-container")
      .expect("Couldn't get div with id: wasm-container");
    let canvas = web_sys::Element::from(window.canvas());
    dst.append_child(&canvas).ok().expect("Couldn't append canvas to div");

    // auto-size when web page window size changes
    // Adapted from: https://github.com/rust-windowing/winit/pull/2074/files
    let observer = web_sys::ResizeObserver::new(resize_closure.as_ref().unchecked_ref()).unwrap();
    observer.observe(&dst);
  };

  //// EVENT LOOP ////

  let mut state = State::new(Arc::clone(&window)).await;
  let mut last_render_time = instant::Instant::now();
  event_loop.run(move |event, _, control_flow| {
    *control_flow = ControlFlow::Poll;
    match event {
            Event::MainEventsCleared => state.window().request_redraw(),
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion{ delta, },
                .. // We're not using device_id currently
            } => if state.mouse_pressed {
                state.camera_controller.process_mouse(delta.0, delta.1)
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window().id() && !state.input(event) => {
                match event {
                    #[cfg(not(target_arch="wasm32"))]
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                let now = instant::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                state.update(dt);
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // We're ignoring timeouts
                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                }
            }
            _ => {}
        }
  });
}
