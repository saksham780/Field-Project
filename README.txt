DiaPredict — Diabetes Prediction Using One Machine-Learning Model
==================================================================

PROJECT IDEA
DiaPredict is a student-friendly web application that demonstrates diabetes
classification using a single machine-learning algorithm: Random Forest.

WHY ONLY ONE MODEL?
This version uses only Random Forest, so it is easier to understand, explain,
train, and maintain. Random Forest is a strong general-purpose model for
tabular data. However, no model is guaranteed to be the best for every dataset.
The application reports its actual test metrics instead of promising a fixed
accuracy.

FILES INCLUDED
- app.py: Streamlit web app with overview, assessment, performance, and project pages
- download_dataset.py: downloads the public CSV dataset
- train_model.py: trains and reports metrics for the same single Random Forest model
- requirements.txt: required Python packages
- data/: dataset folder (created/downloaded during setup)

SETUP ON WINDOWS
1. Extract the ZIP file to a folder such as Desktop\\DiaPredict_OneBestModel.
2. Install Python 3.10 or newer from https://www.python.org/downloads/
   and select "Add Python to PATH" during installation.
3. Open the project folder in File Explorer.
4. Click the address bar, type cmd, and press Enter.
5. Install dependencies:
      python -m pip install --upgrade pip
      python -m pip install -r requirements.txt
6. Download the dataset:
      python download_dataset.py
7. Start the app:
      python -m streamlit run app.py
8. Open the local address printed in Command Prompt, usually:
      http://localhost:8501

OPTIONAL MODEL REPORT
Run this command to print the Random Forest metrics in Command Prompt:
      python train_model.py

FEATURES
- Single Random Forest classifier (300 trees)
- Input form for age, gender and symptoms
- Prediction result page
- Accuracy, precision, recall, F1-score and confusion matrix
- Simple project overview and dataset explanation
- Educational-use warning

DATASET
UCI Early Stage Diabetes Risk Prediction dataset, based on age, gender and
symptom questionnaire responses. The dataset has 520 original records and 16
predictor features. It does not contain blood glucose or BMI values.

Dataset page:
https://archive.ics.uci.edu/dataset/529/early

Dataset citation:
UCI Machine Learning Repository (2020), Early Stage Diabetes Risk Prediction.
DOI: https://doi.org/10.24432/C5VG8H

IMPORTANT MEDICAL DISCLAIMER
This project is for learning and demonstration only. It is not clinically
validated and must not be used to diagnose diabetes, choose treatment, or
change medication. If you have symptoms or concerns, consult a qualified
healthcare professional. A negative prediction does not rule out diabetes.
