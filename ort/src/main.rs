use std::time::Instant;
use ort_test::{get_inference_session, predict, read_csv_file, write_csv};
type Error = Box<dyn std::error::Error>;

fn main() -> Result<(), Error> {
    let start = Instant::now();
    let model_path = "onnx_models/pipeline_svc.onnx";
    let session = get_inference_session(model_path)?;

    let dataset_path = "data/processed_data.csv";
    let features = read_csv_file(dataset_path)?;
    let mut output_df = predict(features, session)?;

    let output_csv = "output.csv";
    write_csv(&mut output_df, output_csv)?;
    let end = start.elapsed();

    println!("Inference time {:?} seconds", end.as_secs());
    Ok(())
}
