
# Install ONNXRuntime Build dependency

## Download the latest CPU version (Replace version if needed)
curl -LO https://github.com/microsoft/onnxruntime/releases/latest/download/onnxruntime-linux-x64-1.8.1.tgz

## Extract the package
tar -xzf onnxruntime-linux-x64-1.8.1.tgz

## Move to a system-wide directory (optional)
sudo mv onnxruntime-linux-x64-1.8.1 /usr/local/onnxruntime

## Set the dynamic library required by onnxruntime 
```
echo "# set env for onnxruntime" >> ~/.bashrc
echo "export ONNXRUNTIME_DIR=/usr/local/onnxruntime" >> ~/.bashrc
echo "export PATH=$ONNXRUNTIME_DIR/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

## Build for release

cargo build --release

## Run

Ensure LD_LIBRARY_PATH is set and points to the folder of the onnxruntime.



