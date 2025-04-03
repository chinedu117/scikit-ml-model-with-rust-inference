use std::{fmt, fs::File, num::NonZero, time::Instant};

use ndarray::{ArrayBase, Axis};
use ort::{
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
};
use polars::prelude::*;
use polars::{
    frame::DataFrame,
    io::{SerReader, SerWriter},
    series::Series,
};

#[derive(Debug, Clone)]
pub enum AppError {
    Message(String),
    Session(String),
}

impl std::error::Error for AppError {}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.clone() {
            AppError::Message(e) => write!(f, "{}", e),
            AppError::Session(e) => write!(f, "{}", e),
        }
    }
}

pub fn get_inference_session(model_path: &str) -> Result<Session, AppError> {
    let num_cores = std::thread::available_parallelism()
        .unwrap_or(NonZero::new(1).unwrap())
        .get();

    let session = Session::builder()
        .map_err(|_| AppError::Message("Unable to create ONNX runtime".into()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|_| AppError::Message("Unable to create ONNX runtime".into()))?
        .with_intra_threads(num_cores)
        .map_err(|_| AppError::Message("Unable to create ONNX runtime".into()))?
        .commit_from_file(model_path)
        .map_err(|_| {
            AppError::Message("Unable to read model. Ensure its a valid ONNX model".into())
        })?;

    Ok(session)
}

type FeatureTensor = ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>;

pub fn read_csv_file(dataset_path: &str) -> Result<FeatureTensor, AppError> {

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(dataset_path.into()))
        .unwrap()
        .finish()
        .map_err(|_| {
            AppError::Message("Unable to read CSV to file.".into())
        })?;
    
    let _target_column = df.column("diabetes").unwrap().clone();
    let df = df.drop("diabetes").unwrap();

    df.to_ndarray::<Float32Type>(IndexOrder::C)
        .map_err(|_| {
            AppError::Message("Generate features from supplied CSV file.".into())
        })
}

pub fn predict(features: FeatureTensor, session: Session) -> Result<DataFrame, ort::Error> {

    let output: ort::session::SessionOutputs<'_, '_> = session.run(inputs![features]?)?;

    let predictions = output["label"].try_extract_tensor::<i64>()?;

    let prediction_vec: Vec<i64> = predictions.iter().cloned().collect();

    let prediction_series = Series::new("label".into(), prediction_vec);
    let mut prob_0_series = Series::new_empty("0".into(), &DataType::Float32);
    let mut prob_1_series = Series::new_empty("1".into(), &DataType::Float32);

    let probabilities = output["probabilities"].try_extract_tensor::<f32>()?;

    probabilities.axis_iter(Axis(0)).for_each(|row| {
        let row_vec: Vec<f32> = row.iter().cloned().collect();
        let _ = prob_0_series.append(&Series::from_vec("0".into(), vec![row_vec[0]]));
        let _ = prob_1_series.append(&Series::from_vec("1".into(), vec![row_vec[1]]));
    });

    let df = DataFrame::new(vec![
        prediction_series.into(),
        prob_0_series.into(),
        prob_1_series.into(),
    ]);

    Ok(df.unwrap())
}

pub fn write_csv(df: &mut DataFrame, destination_path: &str) -> Result<(), AppError> {
    if !destination_path.ends_with(".csv") {
        return Err(AppError::Message(
            "Invalid output file. Must end with .csv".into(),
        ));
    }

    let mut file = File::create(destination_path)
        .map_err(|_| AppError::Message("Unable to create output file.".into()))?;

    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(df)
        .map_err(|_| AppError::Message("Unable to write CSV to file.".into()))?;

    println!("Result Saved at {}", destination_path);

    Ok(())
}


pub fn run(model_path: &str, dataset_path: &str, output_path: &str) -> Result<(), AppError> {

    let start = Instant::now();
    let session = get_inference_session(model_path)?;

    let features = read_csv_file(dataset_path)?;
    let mut output_df = predict(features, session)
        .map_err(|_| AppError::Message("An error occured. Unable to make inference".into()))?;

    write_csv(&mut output_df, output_path)?;
    let end = start.elapsed();

    println!("Inference time {:?} seconds", end.as_secs());

    Ok(())
}