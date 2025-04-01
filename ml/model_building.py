import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the preprocessed data
df = pd.read_csv("assets/diabetes_dataset_with_notes.csv")


# Convert categorical columns to type 'category'
categorical_cols = ["gender", "location", "smoking_history"]
for col in categorical_cols:
    df[col] = df[col].astype("category")

# Prepare the dataset for modeling. Drop columns that are non-numeric and not needed for modelling
model_df = df.drop(
    columns=[
        "clinical_notes",
        "year",
    ]
)

# Convert categorical variables using one-hot encoding
model_df = pd.get_dummies(model_df, drop_first=True)

# Remove location colums
model_df = model_df.drop(
    columns=[
        "location_Alaska",
        "location_Arizona",
        "location_Arkansas",
        "location_California",
        "location_Colorado",
        "location_Connecticut",
        "location_Delaware",
        "location_District of Columbia",
        "location_Florida",
        "location_Georgia",
        "location_Guam",
        "location_Hawaii",
        "location_Idaho",
        "location_Illinois",
        "location_Indiana",
        "location_Iowa",
        "location_Kansas",
        "location_Kentucky",
        "location_Louisiana",
        "location_Maine",
        "location_Maryland",
        "location_Massachusetts",
        "location_Michigan",
        "location_Minnesota",
        "location_Mississippi",
        "location_Missouri",
        "location_Montana",
        "location_Nebraska",
        "location_Nevada",
        "location_New Hampshire",
        "location_New Jersey",
        "location_New Mexico",
        "location_New York",
        "location_North Carolina",
        "location_North Dakota",
        "location_Ohio",
        "location_Oklahoma",
        "location_Oregon",
        "location_Pennsylvania",
        "location_Puerto Rico",
        "location_Rhode Island",
        "location_South Carolina",
        "location_South Dakota",
        "location_Tennessee",
        "location_Texas",
        "location_United States",
        "location_Utah",
        "location_Vermont",
        "location_Virgin Islands",
        "location_Virginia",
        "location_Washington",
        "location_West Virginia",
        "location_Wisconsin",
        "location_Wyoming",
    ]
)

model_df.rename(
    columns={
        "smoking_history_not current": "smoking_history_not_current",
        "race:AfricanAmerican": "race_african_american",
        "race:Asian": "race_asian",
        "race:Caucasian": "race_caucasian",
        "race:Hispanic": "race_hispanic",
        "race:Other": "race_other",
    },
    inplace=True,
)

# Save processed data to CSV

model_df.to_csv(os.path.join("assets", "processed_data.csv"), index=False)

# Define features and target
X = model_df.drop("diabetes", axis=1)

y = model_df["diabetes"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Feature scaling for better performance of the logistic regression
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# Function to train modelsy_test
def train_models(X_train, y_train, X_test, y_test):
    models = {}

    # support vector machine

    # Create an SVM classifier with a linear kernel

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # Step 1: Standard scaling
            (
                "svc",
                SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
            ),  # Step 2: Regression model
        ]
    )

    pipeline.fit(X_train, y_train)
    models["SVC"] = pipeline

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # Step 1: Standard scaling
            (
                "xgboost",
                xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            ),  # Step 2: Regression model
        ]
    )
    # Build and train XGBoost model
    # model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    pipeline.fit(X_train, y_train)
    models["xgboost"] = pipeline

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # Step 1: Standard scaling
            (
                "naive_bayes",
                GaussianNB(),
            ),  # Step 2: Regression model
        ]
    )
    # Initialize and train Naive Bayes model
    # model = GaussianNB()
    pipeline.fit(X_train, y_train)
    models["naive_bayes"] = pipeline

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # Step 1: Standard scaling
            (
                "catboost",
                CatBoostClassifier(verbose=0),
            ),  # Step 2: Regression model
        ]
    )
    # model = CatBoostClassifier(verbose=0)
    pipeline.fit(X_train, y_train)
    models["catboost"] = pipeline

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # Step 1: Standard scaling
            (
                "lightgbm",
                lgb.LGBMClassifier(
                    learning_rate=0.09,
                    max_depth=-5,
                    random_state=42,
                ),
            ),  # Step 2: Regression model
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
        lightgbm__eval_set=[(X_test, y_test), (X_train, y_train)],
        lightgbm__eval_metric="logloss",
    )

    models["lightgbm"] = pipeline

    return models


# Train the models
models = train_models(X_train, y_train, X_test, y_test)


# Function to evaluate models
def evaluate_models(models, X_test, y_test):
    if not os.path.exists("model_reports"):
        os.makedirs("model_reports")
    if not os.path.exists("models"):
        os.makedirs("models")
    reports = {}
    for name, model in models.items():

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        reports[name] = {"report": report, "confusion_matrix": cm}

        # Save the classification report
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(f"model_reports/{name}_classification_report.csv", index=True)

        # Save the confusion matrix
        df_cm = pd.DataFrame(
            cm,
            index=["Actual_No", "Actual_Yes"],
            columns=["Predicted_No", "Predicted_Yes"],
        )
        df_cm.to_csv(f"model_reports/{name}_confusion_matrix.csv", index=True)

        # Save the model
        joblib.dump(model, f"models/{name}_model.pkl")

    return reports


# Evaluate the models
reports = evaluate_models(models, X_test, y_test)
