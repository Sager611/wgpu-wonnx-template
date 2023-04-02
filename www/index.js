import * as wasm from "wgpu-wonnx-template";

// check if WebGPU is not available
if (!("gpu" in navigator)) {
    let no_webgpu = document.getElementById("no-webgpu");
    no_webgpu.style.visibility = "visible";
    document.styleSheets.wasm_container.visibility = "hidden";
} else {
    // run() entrypoint found in src/lib.rs (async)
    wasm.run();
}

