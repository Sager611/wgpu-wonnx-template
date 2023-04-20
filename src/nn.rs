// File adapted from: https://github.com/webonnx/wonnx/blob/bb5f57fb1a8838294ca506904f9879ccdf178815/wonnx/examples/squeeze.rs

use std::collections::HashMap;
use std::sync::Arc;

use wonnx::utils::OutputTensor;
use wonnx::{Session, WonnxError};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use crate::resources::{load_string, load_binary};

#[inline]
pub async fn load_wonnx_session() -> Result<Arc<Session>, WonnxError> {
  let model_path = "models/opt-squeeze.onnx";
  let model_bytes = load_binary(model_path).await.unwrap();
  let session = Arc::new(Session::from_bytes(model_bytes.as_slice()).await?);

  Ok(session)
}

async fn get_imagenet_labels() -> String {
  // Download the ImageNet class labels, matching SqueezeNet's classes.
  let labels_path = "models/squeeze-labels.txt";
  let file_contents = load_string(labels_path).await.unwrap();

  file_contents
}

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

  let file_contents = get_imagenet_labels().await;
  let class_labels = file_contents.split("\n").collect::<Vec<&str>>();

  log::info!("-- Predicted classes:");
  for i in 0..10 {
    log::info!(
      "Class index: {} Class name: {} Logit: {}",
      probabilities[i].0,
      class_labels[probabilities[i].0],
      probabilities[i].1
    );
  }

  Ok(result)
}
