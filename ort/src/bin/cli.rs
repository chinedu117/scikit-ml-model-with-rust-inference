use clap::{Parser, Subcommand};
use infer::{error_to_response, run};
type Error = Box<dyn std::error::Error>;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    /// Run inference on the model
    Predict {
        /// ONNX model path
        #[arg(short, long)]
        model: String,
        /// Input dataset as .csv
        #[arg(short, long)]
        input: String,
        /// Output file name
        #[arg(short, long, default_value = "output.csv")]
        output: String,

        /// Overwrite output file if it exists
        #[arg(short, long, default_value = "false")]
        overwrite: bool,
    },
}

fn main() -> Result<(), Error> {
    let args = Args::parse();

    match args.cmd {
        Command::Predict {
            model,
            input,
            output,
            overwrite,
        } => {
           let res =  run(
            &model, 
            &input, 
            &output, 
            overwrite,
        );
           if res.is_err() {
              error_to_response!(res);
           }
    }
    };

    Ok(())
}
