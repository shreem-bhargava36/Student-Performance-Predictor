import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Student Performance Predictor")

st.write(
    "Predict a student's Pass/Fail status and expected final "
    "grade using a machine learning model."
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = "dataset/student-mat.csv"

CLASSIFICATION_MODEL_PATH = (
    "models/classification_pipeline.pkl"
)

REGRESSION_MODEL_PATH = (
    "models/regression_pipeline.pkl"
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(
        DATA_PATH,
        sep=";"
    )


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    classification_model = joblib.load(
        CLASSIFICATION_MODEL_PATH
    )

    regression_model = joblib.load(
        REGRESSION_MODEL_PATH
    )

    return (
        classification_model,
        regression_model
    )


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

try:

    df = load_dataset()

    classification_model, regression_model = load_models()

except FileNotFoundError as e:

    st.error(
        "Required file was not found."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# CREATE DEFAULT STUDENT
# ============================================================

# Remove target columns from model input
X = df.drop(
    ["G3"],
    axis=1
)

# We don't want result as an input feature either.
if "result" in X.columns:
    X = X.drop(
        ["result"],
        axis=1
    )


# ============================================================
# USER INPUT
# ============================================================

st.subheader("Enter Student Details")


G1 = st.slider(
    "Previous Grade G1",
    min_value=0,
    max_value=20,
    value=10
)


G2 = st.slider(
    "Previous Grade G2",
    min_value=0,
    max_value=20,
    value=10
)


studytime = st.selectbox(
    "Weekly Study Time",
    options=[1, 2, 3, 4],
    index=1
)


failures = st.selectbox(
    "Past Class Failures",
    options=[0, 1, 2, 3],
    index=0
)


absences = st.number_input(
    "Number of Absences",
    min_value=0,
    max_value=100,
    value=5,
    step=1
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Performance",
    use_container_width=True
):

    # --------------------------------------------------------
    # Create a valid student row
    # --------------------------------------------------------

    student = X.iloc[[0]].copy()

    # --------------------------------------------------------
    # Replace user-controlled features
    # --------------------------------------------------------

    student["G1"] = G1
    student["G2"] = G2
    student["studytime"] = studytime
    student["failures"] = failures
    student["absences"] = absences


    # --------------------------------------------------------
    # Classification prediction
    # --------------------------------------------------------

    class_prediction = (
        classification_model.predict(student)[0]
    )


    # --------------------------------------------------------
    # Classification probability
    # --------------------------------------------------------

    probability = None

    if hasattr(
        classification_model,
        "predict_proba"
    ):

        probabilities = (
            classification_model.predict_proba(student)
        )

        probability = probabilities[0][1]


    # --------------------------------------------------------
    # Regression prediction
    # --------------------------------------------------------

    score_prediction = (
        regression_model.predict(student)[0]
    )


    # Keep prediction between 0 and 20
    score_prediction = max(
        0,
        min(
            20,
            score_prediction
        )
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader("Prediction Result")


    if class_prediction == 1:

        st.success(
            "✅ PASS"
        )

    else:

        st.error(
            "❌ FAIL"
        )


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    if probability is not None:

        st.metric(
            "Probability of Passing",
            f"{probability * 100:.2f}%"
        )


    # --------------------------------------------------------
    # Predicted G3
    # --------------------------------------------------------

    st.metric(
        "Expected Final Score (G3)",
        f"{score_prediction:.2f} / 20"
    )


    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    st.info(
        "The prediction uses the same preprocessing and "
        "trained machine-learning pipeline evaluated in the "
        "project notebook."
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

with st.expander(
    "ℹ️ About this project"
):

    st.write(
        """
        This project predicts student academic performance
        using machine learning.

        Classification:
        - Pass
        - Fail

        Regression:
        - Predicted final grade (G3)

        The classification target is created as:

        G3 >= 10 → Pass
        G3 < 10 → Fail

        G3 and the derived result label are excluded from
        the input features to prevent target leakage.
        """
    )