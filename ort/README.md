## Infer

A CLI tool for making inference on scikit models in onnx format in ONNX runtime. 

### Build

```
  cargo build --release
```

Then copy the binary named cli from target/releases folder.


### How to use

```
./cli predict --model <ONNX model path> --input <features in CSV format> 

```

The predictions and their corresponding probabilities will be saved in output.csv