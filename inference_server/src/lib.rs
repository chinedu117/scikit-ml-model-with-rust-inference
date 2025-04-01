// use std::time::Instant;

// use onnxruntime::{
//     GraphOptimizationLevel, LoggingLevel,
//     environment::Environment, tensor::OrtOwnedTensor,
// };
// use polars::{
//     io::SerReader,
//     prelude::{CsvReadOptions, Float32Type, IndexOrder},
// };


// fn load_model(){

//    let builder = Environment::builder()
//         .with_name("test")
//         .with_log_level(LoggingLevel::Warning);

//     let environment = builder.build().unwrap();

//     let mut session = environment
//         .new_session_builder()?
//         .with_optimization_level(GraphOptimizationLevel::Basic)?
//         .with_number_threads(4)?
//         .with_model_from_file("onnx_models/pipeline_svc.onnx")?;
// }
