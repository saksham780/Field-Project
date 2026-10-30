import os
import urllib.request
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="DiaPredict | Random Forest", page_icon="🩺", layout="wide")

DATA = "data/diabetes_data_upload.csv"
URL = "https://raw.githubusercontent.com/W-ayang/Early-stage-diabetes-risk-prediction/main/diabetes_data_upload.csv"

FEATURES = [
    "Age", "Gender", "Polyuria", "Polydipsia", "sudden weight loss",
    "weakness", "Polyphagia", "Genital thrush", "visual blurring",
    "Itching", "Irritability", "delayed healing", "partial paresis",
    "muscle stiffness", "Alopecia", "Obesity"
]
NAMES = {
    "Polyuria": "Frequent urination",
    "Polydipsia": "Excessive thirst",
    "sudden weight loss": "Sudden weight loss",
    "weakness": "General weakness",
    "Polyphagia": "Excessive hunger",
    "Genital thrush": "Genital thrush",
    "visual blurring": "Blurred vision",
    "Itching": "Itching",
    "Irritability": "Irritability",
    "delayed healing": "Delayed wound healing",
    "partial paresis": "Partial muscle weakness",
    "muscle stiffness": "Muscle stiffness",
    "Alopecia": "Hair loss",
    "Obesity": "Obesity"
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

/* Base page: dark readable text on a pale background */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #f1faf8 0%, #f7faff 65%, #f2f6ff 100%);
    color: #243746 !important;
}
[data-testid="stHeader"] { background: rgba(247,250,255,.96); }
[data-testid="stAppViewContainer"] .main { color: #243746 !important; }
/* Keep normal page text readable */
[data-testid="stAppViewContainer"] .main p,
[data-testid="stAppViewContainer"] .main li,
[data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"],
[data-testid="stAppViewContainer"] .main [data-testid="stWidgetLabel"],
[data-testid="stAppViewContainer"] .main [data-testid="stWidgetLabel"] *,
[data-testid="stAppViewContainer"] .main label,
[data-testid="stAppViewContainer"] .main label * {
    color: #243746 !important;
    -webkit-text-fill-color: #243746 !important;
    opacity: 1 !important;
}

/* Streamlit form labels: headings and field names must not fade into the page */
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] span,
.stApp .stSelectbox label,
.stApp .stNumberInput label {
    color: #243746 !important;
    -webkit-text-fill-color: #243746 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* Sidebar: explicitly override the page text styles */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #103943 0%, #185b60 100%);
}
section[data-testid="stSidebar"] * {
    color: #eaf7f5 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #d6eeea !important;
}
section[data-testid="stSidebar"] input { color: #243746 !important; }

/* Hero banner */
.hero {
    padding: 28px 30px;
    border-radius: 22px;
    color: #f7fffe !important;
    background: linear-gradient(115deg, #103943, #137e78 65%, #36ad9b);
    margin-bottom: 20px;
    box-shadow: 0 10px 28px #0c55551a;
}
.hero h1 { color: #ffffff !important; font-family: Manrope, sans-serif; font-size: 2.2rem; font-weight: 800; margin: 5px 0; }
.hero p { color: #def8f2 !important; margin: 5px 0 0; }
.eyebrow { color: #a9eee1 !important; font-size: .75rem; font-weight: 700; letter-spacing: .15em; }

/* Dashboard cards */
.metric {
    background: #ffffff;
    border: 1px solid #dfefeb;
    border-radius: 16px;
    padding: 17px 18px;
    min-height: 90px;
    box-shadow: 0 4px 16px #20484b08;
}
.metric-label { color: #62777c !important; font-size: .85rem; font-weight: 600; }
.metric-value { color: #103f46 !important; font-size: 1.45rem; font-weight: 800; margin-top: 5px; }
h1, h2, h3 { font-family: Manrope, sans-serif; color: #143d45 !important; }
.stApp [data-testid="stForm"] h1,
.stApp [data-testid="stForm"] h2,
.stApp [data-testid="stForm"] h3,
.stApp [data-testid="stForm"] p,
.stApp [data-testid="stForm"] label {
    color: #243746 !important;
    -webkit-text-fill-color: #243746 !important;
    opacity: 1 !important;
}

/* Fix Streamlit form controls: pale backgrounds + dark text */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stDateInput"] input {
    background-color: #ffffff !important;
    color: #243746 !important;
    -webkit-text-fill-color: #243746 !important;
    border: 1px solid #c7d9dc !important;
    border-radius: 9px !important;
}
[data-testid="stNumberInput"] button {
    background: #eaf4f3 !important;
    color: #143d45 !important;
    border-color: #c7d9dc !important;
}

/* High-contrast select boxes (Gender and Yes/No symptom fields) */
.stApp div[data-baseweb="select"] > div,
.stApp [data-testid="stSelectbox"] [role="combobox"] {
    background-color: #ffffff !important;
    border: 1px solid #9bb8bd !important;
    color: #243746 !important;
    border-radius: 9px !important;
    box-shadow: none !important;
    opacity: 1 !important;
}
.stApp div[data-baseweb="select"] span,
.stApp div[data-baseweb="select"] input,
.stApp div[data-baseweb="select"] div,
.stApp div[data-baseweb="select"] svg,
.stApp [data-testid="stSelectbox"] [role="combobox"] * {
    color: #243746 !important;
    fill: #526b73 !important;
    -webkit-text-fill-color: #243746 !important;
    opacity: 1 !important;
}

/* Make the selected option readable even when Streamlit applies theme styles */
.stApp [data-testid="stSelectbox"] [data-testid="stMarkdownContainer"],
.stApp [data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] * {
    color: #243746 !important;
    -webkit-text-fill-color: #243746 !important;
}
ul[data-testid="stSelectboxVirtualDropdown"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
div[role="listbox"],
div[role="option"] {
    background-color: #ffffff !important;
    color: #243746 !important;
}
div[role="option"]:hover { background-color: #e5f5f1 !important; }

/* Buttons and form submit */
div.stButton > button,
div.stFormSubmitButton > button {
    background: #11877d !important;
    color: #ffffff !important;
    border: 0 !important;
    border-radius: 11px !important;
    font-weight: 700 !important;
}
div.stButton > button p,
div.stFormSubmitButton > button p { color: #ffffff !important; }

/* Tables and notices */
[data-testid="stDataFrame"] { background: #ffffff; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stAlert"] p { color: inherit !important; }
</style>
""", unsafe_allow_html=True)


def download_data():
    os.makedirs("data", exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "DiaPredict Student Project"})
    with urllib.request.urlopen(request, timeout=25) as response:
        content = response.read()
    if len(content) < 1000:
        raise ValueError("Dataset download looks incomplete.")
    with open(DATA, "wb") as file:
        file.write(content)


@st.cache_data
def load_data():
    if not os.path.exists(DATA):
        download_data()
    df = pd.read_csv(DATA)
    df.columns = [str(column).strip() for column in df.columns]
    target = next((column for column in df.columns if column.lower() == "class"), None)
    if target is None:
        raise ValueError("Could not find the target column named Class.")
    df = df.rename(columns={target: "Class"})
    df = df[FEATURES + ["Class"]].drop_duplicates().copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    for column in FEATURES:
        if column != "Age":
            df[column] = df[column].astype(str).str.strip().str.title()
    df["Class"] = df["Class"].astype(str).str.strip().str.title()
    return df.dropna(subset=["Age", "Class"])


@st.cache_resource
def train_model():
    df = load_data()
    X, y = df[FEATURES], df["Class"]

    numeric_pipeline = Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    categorical_features = [column for column in FEATURES if column != "Age"]
    categorical_pipeline = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessing = ColumnTransformer([
        ("age", numeric_pipeline, ["Age"]),
        ("categories", categorical_pipeline, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # One model only: Random Forest. Its score is measured on a held-out test set.
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    positive_label = "Positive" if "Positive" in set(y_test) else sorted(set(y_test))[0]
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, pos_label=positive_label, zero_division=0),
        "Recall": recall_score(y_test, predictions, pos_label=positive_label, zero_division=0),
        "F1-score": f1_score(y_test, predictions, pos_label=positive_label, zero_division=0)
    }
    matrix = confusion_matrix(y_test, predictions, labels=["Negative", "Positive"])
    return df, model, metrics, X_test, y_test, predictions, matrix


with st.sidebar:
    st.markdown("# 🩺 DiaPredict")
    st.caption("DIABETES RISK • ONE ML MODEL")
    st.markdown("---")
    page = st.radio("NAVIGATION", ["Overview", "Risk Assessment", "Model Performance", "About Project"])
    st.markdown("---")
    st.caption("Student project • Educational use only")

try:
    df, model, metrics, X_test, y_test, test_predictions, matrix = train_model()
except Exception as error:
    st.markdown(
        '<div class="hero"><div class="eyebrow">HEALTHCARE • MACHINE LEARNING</div>'
        '<h1>DiaPredict</h1><p>Diabetes Risk Prediction with Random Forest</p></div>',
        unsafe_allow_html=True
    )
    st.error("The dataset could not be loaded. Check your internet connection.")
    st.markdown("[Download the dataset CSV](https://raw.githubusercontent.com/W-ayang/Early-stage-diabetes-risk-prediction/main/diabetes_data_upload.csv)")
    st.write("Save it as `data/diabetes_data_upload.csv` inside the project folder, then restart the app.")
    st.caption(str(error))
    st.stop()

if page == "Overview":
    st.markdown(
        '<div class="hero"><div class="eyebrow">HEALTHCARE • MACHINE LEARNING</div>'
        '<h1>DiaPredict</h1><p>A simple diabetes risk classification demonstration using one Random Forest model.</p></div>',
        unsafe_allow_html=True
    )
    columns = st.columns(4)
    cards = [
        (len(df), "Dataset records"),
        (len(FEATURES), "Input features"),
        ("Random Forest", "Selected model"),
        (f"{metrics['Accuracy']:.1%}", "Test accuracy")
    ]
    for column, (value, label) in zip(columns, cards):
        with column:
            size = "1rem" if len(str(value)) > 14 else "1.45rem"
            st.markdown(
                f'<div class="metric"><div class="metric-label">{label}</div>'
                f'<div class="metric-value" style="font-size:{size}">{value}</div></div>',
                unsafe_allow_html=True
            )

    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("What this project does")
        st.markdown(
            "- Uses a public early-stage diabetes symptom dataset.\n"
            "- Cleans and prepares the input data.\n"
            "- Trains one Random Forest classifier.\n"
            "- Evaluates it on data held aside during training.\n"
            "- Predicts a dataset class from the user's answers."
        )
    with right:
        st.subheader("Dataset class distribution")
        counts = df["Class"].value_counts().rename_axis("Class").reset_index(name="Records")
        fig = px.pie(counts, names="Class", values="Records", hole=0.58)
        fig.update_layout(height=300, margin=dict(t=5, b=5, l=5, r=5), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Project workflow")
    st.write("Dataset → Data preprocessing → Train/test split → Random Forest → Evaluation → Prediction")

elif page == "Risk Assessment":
    st.markdown(
        '<div class="hero"><div class="eyebrow">INTERACTIVE DEMO</div>'
        '<h1>Risk assessment</h1><p>Answer the questions to get a classification from the trained Random Forest model.</p></div>',
        unsafe_allow_html=True
    )
    st.info("The dataset contains age, gender and symptom questions. It does not contain blood-glucose or BMI readings.")
    with st.form("risk_form"):
        st.subheader("01 · Personal details")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=35)
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male"])

        st.subheader("02 · Health indicators")
        answers = {}
        col1, col2 = st.columns(2)
        symptom_features = [feature for feature in FEATURES if feature not in ("Age", "Gender")]
        for index, feature in enumerate(symptom_features):
            with (col1 if index % 2 == 0 else col2):
                answers[feature] = st.selectbox(NAMES.get(feature, feature), ["No", "Yes"], key=feature)

        submitted = st.form_submit_button("Generate assessment", use_container_width=True)

    if submitted:
        record = {"Age": age, "Gender": gender}
        record.update(answers)
        sample = pd.DataFrame([record])[FEATURES]
        result = str(model.predict(sample)[0])

        st.markdown("---")
        st.subheader("Model output")
        if result.lower() == "positive":
            st.error("The model classified the submitted answers as POSITIVE.")
            st.write("This output is a pattern-based prediction from the dataset, not confirmation of diabetes.")
        else:
            st.success("The model classified the submitted answers as NEGATIVE.")
            st.write("A negative output does not rule out diabetes or other health concerns.")
        with st.expander("View submitted features"):
            st.dataframe(sample, use_container_width=True, hide_index=True)

elif page == "Model Performance":
    st.markdown(
        '<div class="hero"><div class="eyebrow">MODEL EVALUATION</div>'
        '<h1>Random Forest performance</h1><p>Metrics below are calculated on a held-out test set.</p></div>',
        unsafe_allow_html=True
    )
    metric_columns = st.columns(4)
    for column, (name, value) in zip(metric_columns, metrics.items()):
        with column:
            st.markdown(
                f'<div class="metric"><div class="metric-label">{name}</div>'
                f'<div class="metric-value">{value:.1%}</div></div>',
                unsafe_allow_html=True
            )
    st.write("")
    st.subheader("Confusion matrix")
    cm_df = pd.DataFrame(
        matrix,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"]
    )
    st.dataframe(cm_df, use_container_width=True)
    st.caption("Accuracy is the fraction of all test examples classified correctly. Precision, recall and F1-score describe performance for the positive class.")
    st.subheader("Classification report")
    report = classification_report(y_test, test_predictions, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

else:
    st.markdown(
        '<div class="hero"><div class="eyebrow">PROJECT INFORMATION</div>'
        '<h1>About DiaPredict</h1><p>A beginner-friendly diabetes classification project using a single machine-learning algorithm.</p></div>',
        unsafe_allow_html=True
    )
    st.subheader("Objective")
    st.write("To demonstrate how a machine-learning model can learn patterns from an early-stage diabetes symptom dataset and classify example inputs.")
    st.subheader("Why Random Forest?")
    st.markdown(
        "- Combines predictions from many decision trees.\n"
        "- Can learn non-linear relationships between features.\n"
        "- Works well with mixed patterns in tabular datasets.\n"
        "- Supports a simple preprocessing and prediction workflow.\n"
        "- Uses only one model in this project, making the application easier to explain."
    )
    st.subheader("Dataset")
    st.write("The UCI Early Stage Diabetes Risk Prediction dataset contains questionnaire-based records and 16 predictor features. It is small and geographically specific.")
    st.markdown("[UCI Machine Learning Repository — dataset page](https://archive.ics.uci.edu/dataset/529/early)")
    st.subheader("Technology used")
    st.write("Python • Streamlit • Pandas • Scikit-learn • Plotly")
