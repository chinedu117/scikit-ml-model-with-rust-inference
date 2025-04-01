import subprocess

# Step 1: Run data download and preprocessing
subprocess.call(["python", "data_download.py"])

# Step 2: Run data visualization script
subprocess.call(["python", "data_visualization.py"])

# Step 3: Run model building script
subprocess.call(["python", "model_building.py"])

# Step 4: Run the Streamlit app
subprocess.call(["streamlit", "run", "app.py"])
