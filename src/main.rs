use {{crate_name}}::run;

fn main() {
  // run is async to work for web,
  // so we have to block here on desktop
  pollster::block_on(run());
}
