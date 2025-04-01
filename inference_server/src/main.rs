use std::time::Instant;

use onnxruntime::{
    GraphOptimizationLevel, LoggingLevel,
    environment::Environment, tensor::OrtOwnedTensor,
};
use polars::{
    io::SerReader,
    prelude::{CsvReadOptions, Float32Type, IndexOrder},
};

type Error = Box<dyn std::error::Error>;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Error> {

    let start = Instant::now();
    let dataset_path = "data/processed_data.csv";

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(dataset_path.into()))
        .unwrap()
        .finish()?;
    let _target_column = df.column("diabetes").unwrap().clone();

    let df = df.drop("diabetes").unwrap();

    let builder = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Warning);

    let environment = builder.build().unwrap();

    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(4)?
        .with_model_from_file("onnx_models/pipeline_svc.onnx")?;


    println!("DataFrame shape: {:?}", df.shape());

    let features = df.to_ndarray::<Float32Type>(IndexOrder::Fortran).unwrap();
    let input_tensor_values = vec![features.into_dyn()];
    let outputs: Vec<OrtOwnedTensor<f32, _>>  = session.run(input_tensor_values)?; 

    let duration = start.elapsed();

    // println!("Outputs: {:?}", outputs[0].to_slice());
    // println!("Proba: {:?}", outputs[1].to_slice());

    println!("Duration in Secs {}", duration.as_secs());

    Ok(())
}


