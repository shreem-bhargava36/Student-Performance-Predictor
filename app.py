import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Predictor")
st.write("Predict student Pass/Fail status and expected final score.")

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("dataset/student-mat.csv", sep=";")

# Classification target
df["result"] = (df["G3"] >= 10).astype(int)

# Features
X = df.drop(["G3", "result"], axis=1)

y_classification = df["result"]
y_regression = df["G3"]

# -----------------------------
# Preprocessing
# -----------------------------

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)

# -----------------------------
# Train/Test Split
# -----------------------------

X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X_processed,
    y_classification,
    y_regression,
    test_size=0.2,
    random_state=42,
    stratify=y_classification
)

# -----------------------------
# Models
# -----------------------------

classification_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

regression_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

classification_model.fit(X_train, y_cls_train)
regression_model.fit(X_train, y_reg_train)

# -----------------------------
# User Interface
# -----------------------------

st.subheader("Enter Student Details")

G1 = st.slider(
    "Previous Grade G1",
    0, 20, 10
)

G2 = st.slider(
    "Previous Grade G2",
    0, 20, 10
)

studytime = st.selectbox(
    "Weekly Study Time",
    [1, 2, 3, 4],
    index=1
)

failures = st.selectbox(
    "Past Class Failures",
    [0, 1, 2, 3],
    index=0
)

absences = st.number_input(
    "Number of Absences",
    min_value=0,
    max_value=100,
    value=5
)

if st.button("🔮 Predict Performance"):

    # Create a default student using dataset mode values
    student = X.iloc[[0]].copy()

    # Replace important values
    student["G1"] = G1
    student["G2"] = G2
    student["studytime"] = studytime
    student["failures"] = failures
    student["absences"] = absences

    # Preprocess
    student_processed = preprocessor.transform(student)

    # Predictions
    class_prediction = classification_model.predict(student_processed)[0]
    score_prediction = regression_model.predict(student_processed)[0]

    score_prediction = max(0, min(20, score_prediction))

    st.subheader("Prediction Result")

    if class_prediction == 1:
        st.success("✅ PASS")
    else:
        st.error("❌ FAIL")

    st.metric(
        "Expected Final Score (G3)",
        f"{score_prediction:.2f} / 20"
    )

    st.info(
        "The prediction is based on previous grades, study time, "
        "past failures and absences."
    )