import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv("assets/processed_data.csv")
df_raw = pd.read_csv("assets/diabetes_dataset_with_notes.csv")


# Function to display data preview
def display_data():
    st.header("Data Preview")
    st.subheader("Raw Data")
    st.write(df_raw.head(10))
    st.subheader("Processed Data")
    st.write(df.head(10))


# Function to display visualizations
def display_visualizations():
    st.header("Visualizations")
    visualizations = os.listdir("visualizations")
    for viz in visualizations:
        st.subheader(viz.replace(".png", "").replace("_", " ").title())
        img = plt.imread(os.path.join("visualizations", viz))
        st.image(img)


# Function to display model metrics
def display_model_metrics():
    st.header("Model Metrics")
    model_reports = os.listdir("model_reports")
    for report in model_reports:
        if "classification_report" in report:
            st.subheader(
                report.replace("_classification_report.csv", "").replace("_", " ")
            )
            df_report = pd.read_csv(os.path.join("model_reports", report), index_col=0)
            st.table(df_report)


# Function to display confusion matrices
def display_confusion_matrices():
    st.header("Confusion Matrices")
    model_reports = os.listdir("model_reports")
    for cm_file in model_reports:
        if "confusion_matrix" in cm_file:
            st.subheader(cm_file.replace("_confusion_matrix.csv", "").replace("_", " "))
            df_cm = pd.read_csv(os.path.join("model_reports", cm_file), index_col=0)
            fig, ax = plt.subplots()
            sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)


# Function to display model comparisons
def display_model_comparisons():
    st.header("Model Comparisons")
    # Compare models based on precision, recall, f1-score
    comparison_metrics = ["precision", "recall", "f1-score"]
    comparison_df = pd.DataFrame(columns=["Model"] + comparison_metrics)

    model_reports = [
        f for f in os.listdir("model_reports") if "classification_report" in f
    ]
    for report in model_reports:
        model_name = report.replace("_classification_report.csv", "").replace("_", " ")
        df_report = pd.read_csv(os.path.join("model_reports", report), index_col=0)

        # Extract weighted average metrics
        avg_metrics = df_report.loc["weighted avg", comparison_metrics]
        avg_metrics["Model"] = model_name
        avg_metrics = avg_metrics[["Model"] + comparison_metrics]  # Reorder columns

        # Convert Series to DataFrame and reset index
        avg_metrics_df = avg_metrics.to_frame().T.reset_index(drop=True)

        # Concatenate using pd.concat
        comparison_df = pd.concat([comparison_df, avg_metrics_df], ignore_index=True)

    # Convert metric columns to numeric
    for metric in comparison_metrics:
        comparison_df[metric] = pd.to_numeric(comparison_df[metric])

    # Display comparison table
    st.subheader("Model Comparison Table")
    st.table(comparison_df.set_index("Model"))

    # Plot comparison
    st.subheader("Model Comparison Plot")
    comparison_df.set_index("Model", inplace=True)
    comparison_df.plot(kind="bar", figsize=(10, 6))
    st.pyplot(plt)


# Main function
def main():
    st.title("Diabetes (DM) Cases Analysis")

    menu = [
        "Data Preview",
        "Visualizations",
        "Model Metrics",
        "Confusion Matrices",
        "Model Comparisons",
    ]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Data Preview":
        display_data()
    elif choice == "Visualizations":
        display_visualizations()
    elif choice == "Model Metrics":
        display_model_metrics()
    elif choice == "Confusion Matrices":
        display_confusion_matrices()
    elif choice == "Model Comparisons":
        display_model_comparisons()


if __name__ == "__main__":
    main()
