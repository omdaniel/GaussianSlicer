use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::Window;

struct Foo<'a> {
    window: Option<Window>,
    surface: Option<wgpu::Surface<'a>>,
}

fn main() {}
