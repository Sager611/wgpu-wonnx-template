use std::sync::Arc;
use std::collections::HashMap;

use wonnx::{Session, WonnxError};
use wonnx::utils::OutputTensor;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use crate::resources::load_binary;

#[inline]
pub async fn load_wonnx_session() -> Result<Arc<Session>, WonnxError> {
  let model_path = "models/opt-squeeze.onnx";
  let model_bytes = load_binary(model_path).await.unwrap();
  let session = Arc::new(Session::from_bytes(model_bytes.as_slice()).await?);

  Ok(session)
}

// Hardware management
pub async fn classify_image(session: &Session, image: &[f32])
    -> Result<HashMap<String, OutputTensor>, WonnxError> {
  let mut input_data = HashMap::new();
  input_data.insert("data".to_string(), image.into());

  //let time_pre_compute = intant::Instant::now();
  //log::info!("Start Compute");

  let result = session.run(&input_data).await?;

  //let time_post_compute = Instant::now();
  //log::info!("time: first_prediction: {:#?}", time_post_compute - time_pre_compute);

  Ok(result)
}
