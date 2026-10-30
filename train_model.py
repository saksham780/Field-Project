"""Train and evaluate the single Random Forest model used by DiaPredict."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

PATH = "data/diabetes_data_upload.csv"
FEATURES = [
    "Age", "Gender", "Polyuria", "Polydipsia", "sudden weight loss",
    "weakness", "Polyphagia", "Genital thrush", "visual blurring",
    "Itching", "Irritability", "delayed healing", "partial paresis",
    "muscle stiffness", "Alopecia", "Obesity"
]

if not os.path.exists(PATH):
    print("Dataset missing. Run: python download_dataset.py")
    raise SystemExit(1)

df = pd.read_csv(PATH)
df.columns = [str(column).strip() for column in df.columns]
target = next((column for column in df.columns if column.lower() == "class"), None)
if target is None:
    raise ValueError("Could not find the Class target column.")
df = df.rename(columns={target: "Class"})
df = df[FEATURES + ["Class"]].drop_duplicates().copy()
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
for column in FEATURES:
    if column != "Age":
        df[column] = df[column].astype(str).str.strip().str.title()
df["Class"] = df["Class"].astype(str).str.strip().str.title()
df = df.dropna(subset=["Age", "Class"])

X, y = df[FEATURES], df["Class"]
categorical = [column for column in FEATURES if column != "Age"]
preprocessing = ColumnTransformer([
    ("age", Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), ["Age"]),
    ("categories", Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = Pipeline([
    ("preprocessing", preprocessing),
    ("classifier", RandomForestClassifier(
        n_estimators=300, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    ))
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
positive = "Positive" if "Positive" in set(y_test) else sorted(set(y_test))[0]

print("\\nDiaPredict — Random Forest evaluation")
print("-" * 42)
print("Model: Random Forest (300 trees)")
print("Dataset records:", len(df))
print("Training records:", len(X_train))
print("Test records:", len(X_test))
print("Accuracy : {:.2f}%".format(accuracy_score(y_test, predictions) * 100))
print("Precision: {:.2f}%".format(precision_score(y_test, predictions, pos_label=positive, zero_division=0) * 100))
print("Recall   : {:.2f}%".format(recall_score(y_test, predictions, pos_label=positive, zero_division=0) * 100))
print("F1-score : {:.2f}%".format(f1_score(y_test, predictions, pos_label=positive, zero_division=0) * 100))
print("\\nNote: These results are from one held-out test split and do not prove clinical accuracy.")
