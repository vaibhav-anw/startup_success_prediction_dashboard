import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------------
# Load models
# -------------------------

models = {
    "Logistic Regression": joblib.load("logistic_regression.pkl"),
    "Decision Tree": joblib.load("decision_tree.pkl"),
    "Random Forest": joblib.load("random_forest.pkl"),
    "SVM": joblib.load("svm.pkl"),
    "Gradient Boosting": joblib.load("gradient_boosting.pkl")
}

scaler = joblib.load("scaler.pkl")

le = joblib.load("label_encoder.pkl")

accuracy_df = pd.read_csv("model_accuracy.csv")

# -------------------------
# Load dataset
# -------------------------

df = pd.read_csv("startup_data.csv")

df["category_code"] = le.transform(df["category_code"].astype(str))

# -------------------------
# Dashboard Title
# -------------------------

st.title("Startup Success Prediction Dashboard")

st.write("Predict startup success using Machine Learning models")

# -------------------------
# Dataset Explorer
# -------------------------

st.header("Dataset Explorer")

if st.checkbox("Show Dataset"):
    st.dataframe(df)

st.write("Dataset Shape:", df.shape)

if st.checkbox("Show Dataset Statistics"):
    st.write(df.describe())

# -------------------------
# Model Accuracy Leaderboard
# -------------------------

st.header("Model Accuracy Leaderboard")

st.bar_chart(
    accuracy_df.set_index("Model")
)

# -------------------------
# Startup selection
# -------------------------

selected_startup = st.sidebar.selectbox(
    "Choose Startup",
    df["name"]
)

row = df[df["name"] == selected_startup].iloc[0]

st.subheader("Selected Startup Data")

st.dataframe(row)

# -------------------------
# Actual Status
# -------------------------

actual_status = "Success" if row["status"] == "acquired" else "Failure"

st.subheader("Actual Startup Status")

st.write(actual_status)

# -------------------------
# Prepare input
# -------------------------

input_data = np.array([[
    row["funding_total_usd"],
    row["funding_rounds"],
    row["milestones"],
    row["relationships"],
    row["age_first_funding_year"],
    row["age_last_funding_year"],
    row["category_code"]
]])

input_scaled = scaler.transform(input_data)

# -------------------------
# Model Predictions
# -------------------------

st.header("Model Predictions")

results = []

for name, model in models.items():

    pred = model.predict(input_scaled)[0]

    prob = model.predict_proba(input_scaled)[0][1]

    results.append({
        "Model": name,
        "Prediction": "Success" if pred == 1 else "Failure",
        "Probability": round(prob, 3)
    })

results_df = pd.DataFrame(results)

st.dataframe(results_df)

# -------------------------
# Top 10 promising startups
# -------------------------

st.header("Top 10 Most Promising Startups")

features = [
    "funding_total_usd",
    "funding_rounds",
    "milestones",
    "relationships",
    "age_first_funding_year",
    "age_last_funding_year",
    "category_code"
]

X = df[features].fillna(0)

X_scaled = scaler.transform(X)

df["success_probability"] = models["Random Forest"].predict_proba(X_scaled)[:, 1]

top10 = df.sort_values(
    "success_probability",
    ascending=False
).head(10)

st.dataframe(
    top10[["name", "success_probability"]]
)