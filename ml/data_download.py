import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# # Function to download dataset from Kaggle
# def download_kaggle_dataset():
#     kaggle_dataset = 'blastchar/telco-customer-churn'  # Dataset identifier
#     data_dir = 'data'
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
#     os.environ['KAGGLE_USERNAME'] = 'YOUR_KAGGLE_USERNAME'  # Replace with your Kaggle username
#     os.environ['KAGGLE_KEY'] = 'YOUR_KAGGLE_KEY'  # Replace with your Kaggle API key
#     subprocess.call([
#         'kaggle', 'datasets', 'download', kaggle_dataset, '--unzip',
#         '-p', data_dir
#     ])

# # Download the dataset if not already downloaded
# data_file = os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
# if not os.path.exists(data_file):
#     download_kaggle_dataset()

# # Load the dataset
# df_raw = pd.read_csv(data_file)  # Original DataFrame

# # Convert 'TotalCharges' to numeric
# df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')

# # Handle missing values in 'TotalCharges'
# df_raw['TotalCharges'].fillna(df_raw['TotalCharges'].median(), inplace=True)

RAW_DATASET_FILE = "assets/diabetes_dataset_with_notes.csv"

df_raw = pd.read_csv(RAW_DATASET_FILE)


# Data preprocessing
def preprocess_data(df):

    model_df = df.drop(
        columns=[
            "clinical_notes",
            "year",
        ]
    )

    # Convert categorical variables using one-hot encoding
    model_df = pd.get_dummies(model_df, drop_first=True)

    scaler = StandardScaler()
    columns = model_df.columns.to_list()
    model_df[columns] = scaler.fit_transform(model_df)

    return model_df


# Preprocess the data
df_processed = preprocess_data(df_raw)

# Save the preprocessed data
df_processed.to_csv(os.path.join("assets", "processed_data.csv"), index=False)
