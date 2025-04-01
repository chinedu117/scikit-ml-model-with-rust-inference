import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Load the preprocessed data
df = pd.read_csv("assets/diabetes_dataset_with_notes.csv")


def create_visualization(df):

    numeric_df = df.select_dtypes(include=[np.number])

    # Correlation Heatmap (only if four or more numeric columns are present)
    if numeric_df.shape[1] >= 4:
        plt.figure(figsize=(10, 8))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap of Numeric Features")
        plt.tight_layout()
        plt.savefig("visualizations/heat_map.png")
    else:
        print("Not enough numeric columns for a correlation heatmap.")

    # Histogram for numeric distributions - age, bmi, hbA1c_level
    plt.figure(figsize=(12, 4))
    for i, col in enumerate(["age", "bmi", "hbA1c_level"]):
        plt.subplot(1, 3, i + 1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")

    plt.tight_layout()
    plt.savefig("visualizations/distr_age_bmi_hbaic.png")

    # Count plot for categorical variables such as gender
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="gender", palette="pastel")
    plt.title("Count Plot of Gender")
    plt.savefig("visualizations/gender_count.png")

    # Grouped Bar Plot for race distribution across diabetes statuses
    race_cols = [
        "race:AfricanAmerican",
        "race:Asian",
        "race:Caucasian",
        "race:Hispanic",
        "race:Other",
    ]
    race_df = df[race_cols + ["diabetes"]].copy()
    race_df = race_df.melt(id_vars="diabetes", var_name="Race", value_name="Count")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=race_df, x="Race", y="Count", hue="diabetes", palette="muted")
    plt.title("Race Distribution Grouped by Diabetes Status")
    plt.savefig("visualizations/race_vs_dm.png")

    # Box Plot for BMI grouped by diabetes status
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="diabetes", y="bmi", palette="Set2")
    plt.title("BMI Distribution by Diabetes Status")
    plt.savefig("visualizations/bmi_vs_dm.png")

    # Creating a pie chart to count diabetics
    df_smoking_history = df.groupby(["smoking_history"]).count()[["diabetes"]]
    df_smoking_history.plot.pie(y="diabetes", figsize=(5, 5))
    plt.legend(loc=0)
    plt.savefig("visualizations/smoking_vs_dm.png")


# Create visualizations directory if it doesn't exist
if not os.path.exists("visualizations"):
    os.makedirs("visualizations")

# Generate visualizations
create_visualization(df)
