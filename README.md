# 🎓 Student Performance Predictor

An end-to-end Machine Learning application that predicts **student academic performance** using classification and regression techniques. The project compares multiple ML algorithms, evaluates them using standard metrics, and deploys the final prediction pipeline through an interactive **Streamlit web application**.

🔗 **Live Demo:** https://student-performance-predictor-39.streamlit.app/
🔗 **GitHub Repository:** https://github.com/shreem-bhargava36/Student-Performance-Predictor

---

## 📌 Project Overview

Educational institutions collect various student-related attributes that can be used to understand and predict academic performance.

This project develops a machine learning pipeline that:

* Preprocesses student performance data
* Performs exploratory analysis
* Predicts student pass/fail outcomes using classification
* Predicts academic scores using regression
* Compares multiple machine learning algorithms
* Evaluates models using multiple performance metrics
* Saves trained preprocessing and model pipelines
* Provides an interactive Streamlit interface
* Compares the final model with selected results reported in previous research

The project emphasizes **reproducible evaluation and leakage-free model development** rather than artificially maximizing accuracy.

---

## 🎯 Objectives

1. Collect and preprocess student performance data.
2. Handle categorical and numerical features appropriately.
3. Develop classification models for pass/fail prediction.
4. Develop regression models for academic score prediction.
5. Compare different machine learning algorithms.
6. Evaluate models using Accuracy, Precision, Recall and F1-Score.
7. Tune the best-performing model.
8. Save the trained ML pipelines for deployment.
9. Build an interactive Streamlit application.
10. Compare the obtained results with selected research benchmarks.

---

## 🧠 Machine Learning Workflow

```text
Student Dataset
      ↓
Data Cleaning
      ↓
Feature Selection
      ↓
Preprocessing
      ↓
Train / Test Split
      ↓
Model Training
      ↓
Model Comparison
      ↓
Hyperparameter Tuning
      ↓
Final Model Selection
      ↓
Save ML Pipeline
      ↓
Streamlit Application
      ↓
Student Performance Prediction
```

---

## 📊 Dataset

The project uses the **UCI Student Performance dataset**.

The dataset contains academic, demographic, social and behavioural attributes of students.

Important features include:

* Gender
* Age
* Study time
* Previous failures
* Absences
* Family-related attributes
* Parental education
* School-related attributes
* Previous grades

### Target Variables

**Classification**

Predicts whether a student is likely to:

```text
PASS / FAIL
```

**Regression**

Predicts the student's final academic score.

---

## 🔍 Data Preprocessing

The preprocessing stage includes:

* Duplicate checking
* Missing-value handling
* Categorical feature encoding
* Numerical feature processing
* Feature/target separation
* Train-test splitting
* Prevention of target leakage

A particularly important validation step was ensuring that the target-derived `result` variable was not included as an input feature.

The classification features are separated from the target before model training.

---

## 🤖 Models Used

The following classification algorithms were evaluated:

### 1. Logistic Regression

Used as a baseline linear classification model.

### 2. Decision Tree

A tree-based model capable of learning non-linear decision boundaries.

### 3. Random Forest

An ensemble of decision trees that generally provides stronger generalization than a single decision tree.

### 4. XGBoost

A gradient-boosting algorithm that combines multiple decision trees sequentially to improve predictive performance.

---

# 📈 Classification Results

The final leakage-free evaluation on the held-out test set produced the following results:

| Model               |   Accuracy |  Precision |     Recall |   F1-Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     87.34% |     95.74% |     84.91% |     90.00% |
| Decision Tree       |     86.08% |     97.73% |     81.13% |     88.66% |
| Random Forest       |     87.34% |     95.74% |     84.91% |     90.00% |
| **XGBoost**         | **89.87%** | **97.87%** | **86.79%** | **92.00%** |

### 🏆 Best Classification Model

**XGBoost** achieved the highest held-out test accuracy:

> **89.87% Accuracy**

It also achieved the highest F1-score:

> **92.00% F1-Score**

Therefore, XGBoost was selected as the final classification model.

---

## ⚠️ Target Leakage Check

During model validation, a target leakage issue was identified in an earlier experiment.

The target variable was derived as:

```python
result = (G3 >= 10).astype(int)
```

Therefore, including `result` among the input features would allow the model to directly access information derived from the target.

The corrected feature selection excludes both:

```python
G3
result
```

from the classification input features.

The final reported **89.87% test accuracy** is based on the corrected leakage-free workflow.

This correction was important to ensure that the reported performance represents actual predictive capability rather than information leakage.

---

# 📉 Regression

In addition to classification, the project also includes regression models for predicting the final academic score.

The regression workflow follows:

```text
Input Features
      ↓
Preprocessing
      ↓
Train/Test Split
      ↓
Regression Model
      ↓
Predicted Final Score
```

Regression models evaluated include:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost Regressor

Model performance is evaluated using regression metrics such as:

* R² Score
* Mean Squared Error
* Root Mean Squared Error

---

# 📚 Research Comparison

The final XGBoost result was compared with selected algorithm-specific results reported in previous student-performance research.

| Research Study   | Year | Selected Algorithm             | Reported Accuracy | Our XGBoost |
| ---------------- | ---: | ------------------------------ | ----------------: | ----------: |
| Agrawal & Mavani | 2015 | Multivariate Linear Regression |            70.48% |  **89.87%** |
| Oche et al.      | 2020 | BayesNet                       |            74.00% |  **89.87%** |
| Ghosh et al.     |    — | Random Forest                  |            83.00% |  **89.87%** |
| Ahmed et al.     | 2024 | Naïve Bayes                    |            83.32% |  **89.87%** |
| Wakelam et al.   | 2020 | Reported average               |            67.00% |  **89.87%** |

### Research Comparison Note

These studies use different datasets, features, institutions, prediction tasks and evaluation methodologies.

Therefore, the comparison should be interpreted as a **benchmark comparison**, not as a direct claim that the proposed model is universally superior to the models in those studies.

The selected reported results are all below the project's final held-out test accuracy of **89.87%**.

---

# 🌐 Streamlit Application

The trained models are integrated into an interactive Streamlit application.

### Application Features

* Student input form
* Pass/fail prediction
* Prediction probability
* Academic score prediction
* Interactive results
* Model-based prediction output

### Live Application

👉 https://student-performance-predictor-39.streamlit.app/

---

# 🏗️ Project Structure

```text
Student-Performance-Predictor/
│
├── dataset/
│   └── student-mat.csv
│
├── models/
│   ├── classification_pipeline.pkl
│   └── regression_pipeline.pkl
│
├── notebooks/
│   └── student_performance.ipynb
│
├── app.py
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Data Visualization

* Matplotlib

### Web Application

* Streamlit

### Model Persistence

* Joblib

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

# 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/shreem-bhargava36/Student-Performance-Predictor.git
```

### 2. Move into the project directory

```bash
cd Student-Performance-Predictor
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📌 Key Features

✅ End-to-end ML pipeline
✅ Classification and regression
✅ Multiple model comparison
✅ XGBoost implementation
✅ Hyperparameter tuning
✅ Leakage-aware evaluation
✅ Saved ML pipelines
✅ Interactive Streamlit application
✅ Live deployment
✅ Research benchmark comparison
✅ Reproducible project structure

---

# 🔬 Limitations

Although the project provides a working predictive system, several limitations remain:

* The dataset is relatively limited compared with large institutional datasets.
* Results may vary for students from different institutions.
* Student performance is influenced by many factors that may not be present in the dataset.
* The research-paper comparison uses different datasets and experimental settings.
* The model should be considered a decision-support tool rather than a definitive assessment of a student's future performance.

---

# 🔮 Future Scope

Possible future improvements include:

* Larger and more diverse institutional datasets
* Additional behavioural and attendance features
* Explainable AI using SHAP
* Student risk-level prediction
* Early-warning notifications
* Personalized academic recommendations
* Model monitoring
* Dashboard-based analytics
* Cross-institution validation
* More advanced ensemble and deep-learning models

---

# 📖 References

1. Agrawal, H., & Mavani, H. — *Student Performance Prediction using Machine Learning*.
2. Oche, O. E., Nasir, S. M., & Ibrahim, A. M. — *Analysis and Prediction of Student Performance Using Data Mining Classification Algorithms*, 2020.
3. Ghosh et al. — *Data Mining Approach to Predict Academic Performance of Students*.
4. Ahmed et al. — Student performance prediction using machine learning classification algorithms, 2024.
5. Wakelam, E., Jefferies, A., Davey, N., & Sun, Y. — *The Potential for Student Performance Prediction in Small Cohorts with Minimal Available Attributes*, 2020.
6. UCI Machine Learning Repository — Student Performance Dataset.

---

**Student Performance Predictor**

Developed as an academic Machine Learning project.

---

## ⭐ Project Summary

> **Student Performance Predictor is an end-to-end machine learning system that compares multiple classification and regression algorithms, performs leakage-aware evaluation, and deploys the selected XGBoost model through an interactive Streamlit application. The final XGBoost classifier achieved 89.87% accuracy and a 92.00% F1-score on the held-out test set.**
