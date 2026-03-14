import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

print("Starting model training...")

# -------------------------
# Load dataset
# -------------------------

df = pd.read_csv("startup_data.csv")

print("Dataset loaded:", df.shape)

# -------------------------
# Target variable
# -------------------------

df = df[df["status"].isin(["acquired", "closed"])]

df["status"] = df["status"].map({
    "acquired": 1,
    "closed": 0
})

# -------------------------
# Encode category
# -------------------------

le = LabelEncoder()

df["category_code"] = le.fit_transform(df["category_code"].astype(str))

# -------------------------
# Feature selection
# -------------------------

features = [
    "funding_total_usd",
    "funding_rounds",
    "milestones",
    "relationships",
    "age_first_funding_year",
    "age_last_funding_year",
    "category_code"
]

X = df[features]
y = df["status"]

X = X.fillna(0)

# -------------------------
# Train test split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Scaling
# -------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# Models
# -------------------------

models = {

    "logistic_regression": LogisticRegression(max_iter=1000),

    "decision_tree": DecisionTreeClassifier(),

    "random_forest": RandomForestClassifier(n_estimators=200),

    "svm": SVC(probability=True),

    "gradient_boosting": GradientBoostingClassifier()
}

results = {}

# -------------------------
# Train models
# -------------------------

for name, model in models.items():

    print("Training:", name)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    results[name] = acc

    joblib.dump(model, f"{name}.pkl")

# -------------------------
# Save scaler and encoder
# -------------------------

joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

# -------------------------
# Save model accuracy
# -------------------------

accuracy_df = pd.DataFrame(
    results.items(),
    columns=["Model", "Accuracy"]
)

accuracy_df.to_csv("model_accuracy.csv", index=False)

print("\nModel Accuracy")

print(accuracy_df)

print("\nTraining complete")