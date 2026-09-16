# Student Performance Predictor

## Overview

Student Performance Predictor is a Machine Learning project designed to predict student academic performance using the UCI Student Performance Dataset.

The project performs two prediction tasks:

1. Classification - Predict whether a student will Pass or Fail.
2. Regression - Predict the student's final grade (G3).

An interactive Streamlit web application is included for demonstration.

## Objectives

- Collect and preprocess student performance data.
- Perform feature engineering.
- Create a Pass/Fail classification target.
- Train multiple Machine Learning models.
- Compare model performance.
- Perform hyperparameter tuning.
- Save trained models.
- Build an interactive Streamlit application.

## Dataset

The project uses the UCI Student Performance Dataset.

Dataset file:

`dataset/student-mat.csv`

The dataset contains 395 student records and 33 original features.

Important features include:

- G1 - First-period grade
- G2 - Second-period grade
- G3 - Final grade
- studytime - Weekly study time
- failures - Number of past class failures
- absences - Number of school absences

## Target Variables

### Classification

A Pass/Fail target named `result` is created using:

```python
df["result"] = (df["G3"] >= 10).astype(int)

Where:

1 = Pass
0 = Fail

The original G3 and result columns are excluded from the classification input features.

Regression

The regression model predicts the final grade:

G3

The G3 score ranges from 0 to 20.

Data Preprocessing

The preprocessing workflow includes:

Data loading
Data quality checking
Duplicate checking
Missing-value checking
Feature engineering
Train/test splitting
Numerical feature scaling
Categorical feature encoding

Numerical features are processed using StandardScaler.

Categorical features are processed using OneHotEncoder.

The preprocessing object is saved as:

models/preprocessor.pkl

Machine Learning Models
Classification

The following classification algorithms are implemented:

Logistic Regression
Decision Tree Classifier
Random Forest Classifier
XGBoost Classifier

Classification evaluation metrics include:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Regression

The following regression algorithms are implemented:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
XGBoost Regressor

Regression evaluation metrics include:

MAE
MSE
RMSE
R² Score
Regression Results
Initial Models
Model	MAE	MSE	RMSE	R²
Linear Regression	1.408	3.822	1.955	0.810
Decision Tree	0.924	3.152	1.775	0.843
Random Forest	0.810	1.765	1.328	0.912
XGBoost	0.853	1.685	1.298	0.916
Tuned Models
Model	MAE	MSE	RMSE	R²
Tuned Random Forest	0.768	1.538	1.240	0.923
Tuned XGBoost	0.825	1.639	1.280	0.918

The tuned Random Forest achieved the strongest test-set metrics among the regression models evaluated in the current experiment.

Hyperparameter Tuning

GridSearchCV was used for hyperparameter tuning.

Tuned Random Forest

Best parameters:

n_estimators = 100
max_depth = 5
min_samples_split = 5
min_samples_leaf = 2
Tuned XGBoost

Best parameters:

n_estimators = 100
max_depth = 2
learning_rate = 0.05
subsample = 0.8
Streamlit Application

The project includes an interactive Streamlit application.

The user can enter:

Previous Grade G1
Previous Grade G2
Weekly Study Time
Past Class Failures
Number of Absences

The application provides:

Pass/Fail prediction
Expected final grade out of 20
Run the Application

Clone the repository:

git clone https://github.com/shreem-bhargava36/Student-Performance-Predictor.git

Open the project folder:

cd Student-Performance-Predictor

Install the required libraries:

pip install -r requirements.txt

Run Streamlit:

streamlit run app.py

The application will open at:

http://localhost:8501
Project Structure
Student-Performance-Predictor/
│
├── dataset/
│   ├── student-mat.csv
│   ├── X_train_processed.csv
│   ├── X_test_processed.csv
│   ├── y_reg_train.csv
│   ├── y_reg_test.csv
│   ├── y_cls_train.csv
│   └── y_cls_test.csv
│
├── models/
│   ├── preprocessor.pkl
│   ├── classification_model.pkl
│   └── regression_model.pkl
│
├── notebooks/
│   └── student_performance.ipynb
│
├── src/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
Project Workflow
UCI Student Performance Dataset
            ↓
Data Collection
            ↓
Data Cleaning
            ↓
Feature Engineering
            ↓
Train/Test Split
            ↓
Encoding + Scaling
            ↓
Classification + Regression
            ↓
Model Evaluation
            ↓
Hyperparameter Tuning
            ↓
Model Saving
            ↓
Streamlit Application
            ↓
Student Performance Prediction
Technologies Used
Python
Pandas
NumPy
Scikit-learn
XGBoost
Joblib
Streamlit
Jupyter Notebook
Git
GitHub
Current Project Status
Data Collection - Complete
Data Preprocessing - Complete
Classification Models - Implemented
Regression Models - Implemented
Hyperparameter Tuning - Complete
Model Saving - Complete
Streamlit Application - Implemented
GitHub Repository - Available
Classification Result Note

The initial classification experiments produced perfect test-set metrics. Because unusually perfect results can indicate a data alignment or leakage issue, these classification metrics are being treated as under verification rather than presented as final performance claims.

The regression results reported above are from the current regression experiments and hyperparameter tuning.

Future Improvements
Perform additional validation of classification features and data alignment.
Improve the Streamlit interface.
Add visual analytics and EDA charts.
Add explainable AI features.
Add model confidence and prediction explanations.
Deploy the application online.
License

This project is developed for educational and academic purposes.