<div align="center">

  <h1><code>wgpu-wonnx-template</code></h1>

  <strong>A template for kick starting a Rust, <a href="https://gpuweb.github.io/gpuweb/">WebGPU</a> and <a href="https://onnx.ai/">ONNX</a> project using <a href="https://wgpu.rs/"><code>wgpu</code></a> and <a href="https://github.com/webonnx/wonnx"><code>wonnx</code></a>.</strong>

</div>

# :warning: WIP PROJECT :warning:

## :memo: About

<div align="center" style="background: #9d0006; border-left: 10px solid red; padding: 10px;">
<b>ATTENTION</b>: You need a <a href="#enabling-webgpu-on-your-browser">WebGPU compatible browser</a> to run your project on web.
</div>
<br/>

This template comes with:

&emsp; :ballot_box_with_check: &ensp; Minimal [Node.js](https://nodejs.org/en) web project that binds to the WebAssembly under `www/`. <br/>
&emsp; :ballot_box_with_check: &ensp; Example model, texture, light and camera shader based on the [learn-wgpu tutorial](https://github.com/sotrh/learn-wgpu/tree/master/code/intermediate/tutorial12-camera). <br/>
&emsp; :black_square_button: &ensp; Example classification of rendered image using [SqueezeNet](https://arxiv.org/abs/1602.07360) ONNX model. <br/>
&emsp; :ballot_box_with_check: &ensp; Auto-resizing of web canvas using [ResizeObserver](https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver). <br/>

## :bulb: Usage

Quickstart a project named `my-project`:

```
cargo generate --git https://github.com/Sager611/wgpu-wonnx-template.git --name my-project
cd my-project
```

## :rocket: Build &amp; Run your project

We use [`just`](https://github.com/casey/just) as a multi-platform build toolchain.

* Install `just` with:

```
cargo install just
```

Additional dependencies are needed to build the web application:

* [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/): to compile to WebAssembly.
* [npm](https://www.npmjs.com/get-npm): to install and run the web application under `www/`.

Finally, [`wonnx`](https://github.com/webonnx/wonnx) uses compute shaders, which on web are only available by [enabling WebGPU on your browser](#enabling-webgpu-on-your-browser).

### :hammer_and_pick: Build for desktop

* Debug:

```
just desktop
```

* Release:

```
just desktop --release
```

### :hammer_and_pick: Build for web

* Debug:

```
just web
```

* Release:

```
just web --release
```

Resulting web project is saved under `www/`.

### :racing_car: Run on desktop

* Debug:

```
just run-desktop
```

* Release:

```
just run-desktop --release
```

### :racing_car: Run on web

* Debug:

```
just run-web
```

* Release:

```
just run-web --release
```

Then, open [localhost:8080](http://localhost:8080/) in your browser.

# :computer: Enabling WebGPU on your browser

The [WebGPU API](https://gpuweb.github.io/gpuweb/) implementation is still in development on modern browsers, so it has to be enabled manually.

For information on all officially supported browsers and how to activate WebGPU, check [here](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status).

To check if WebGPU is currently enabled in your browser, along with the adapter limits, [follow this link](https://browserleaks.com/webgpu).

Here are some specific examples on how to enable WebGPU on Firefox and Chrome

## Firefox

You need to install [Firefox Nightly](https://www.mozilla.org/en-US/firefox/113.0a1/releasenotes/).

Then, follow these steps:

* Go to [`about:config`](about:config).
* Set `dom.webgpu.enabled` and `gfx.webgpu.force-enabled` to `true`.

## Chrome

You need to install [Chrome Canary](https://www.google.com/chrome/canary/).

Then, follow these steps:

* Go to [`chrome://flags`](chrome://flags).
* Set `Unsafe WebGPU` and `WebGPU Developer Features` to `Enabled`.

# :balance_scale: Licensing

This template comes with the two most common licenses for Rust crates:

* [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0) ([LICENSE-APACHE](LICENSE-APACHE)).
* [MIT License](http://opensource.org/licenses/MIT) ([LICENSE-MIT](LICENSE-MIT)).

This template is also licensed under either of [Apache-2.0](http://www.apache.org/licenses/LICENSE-2.0) OR [MIT](http://opensource.org/licenses/MIT), except for:

* [`res/models/squeezenet-labels.txt`](res/models/squeezenet-labels.txt): under Apache-2.0 only.
