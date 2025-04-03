use clap::{Parser, Subcommand};
use ort_test::run;
type Error = Box<dyn std::error::Error>;

#[derive(Parser)]
#[command(author = "Chinedu", version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    /// Run inference on the model
    Predict {
        /// ONNX model path
        model: String,
        /// Input dataset as .csv
        input: String,
        /// Output file name
        output: String,
    },
}

fn main() -> Result<(), Error> {
    let args = Args::parse();

    match args.cmd {
        Command::Predict {
            model,
            input,
            output,
        } => run(&model, &input, &output),
    };

    Ok(())
}
