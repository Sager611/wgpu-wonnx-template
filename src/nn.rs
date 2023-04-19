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

//pub fn get_label(output: OutputTensor) -> String {

//}

// Hardware management
pub async fn classify_image(session: &Session, image: &[f32]) -> Result<HashMap<String, OutputTensor>, WonnxError> {
  log::info!("Start Compute");

  let mut input_data = HashMap::new();
  input_data.insert("data".to_string(), image.into());

  let result = session.run(&input_data).await?;

  log::info!("End Compute");

  // PRINT CLASSIFICATION //
  let probabilities = result.clone().into_iter().next().unwrap().1;
  let probabilities: Vec<f32> = probabilities.try_into().unwrap();
  let mut probabilities = probabilities.iter().enumerate().collect::<Vec<_>>();
  probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());

  //let class_labels = get_imagenet_labels();

  log::info!("-- Predicted classes:");
  for i in 0..10 {
    log::info!(
      "Class index: {} Logit: {}",
      probabilities[i].0,
      probabilities[i].1 //"Infered result: {} of class: {}",
                         //class_labels[probabilities[i].0], probabilities[i].0
    );
  }

  Ok(result)
}
