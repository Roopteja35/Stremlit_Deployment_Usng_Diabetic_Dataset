                                        Diabetes Prediction Web App (SVM Model)
Project Overview:
  This project is a Machine Learning based Diabetes Prediction Web Application built using:
    Python,
    Scikit-learn,
    Support Vector Machine (SVM),
    Streamlit.
  The application predicts whether a person is diabetic or not based on medical parameters such as glucose level, BMI, age, insulin, etc.

Problem Statement:
  Diabetes is one of the most common chronic diseases worldwide. Early prediction helps in preventive care and medical planning.
  This project uses the PIMA Indian Diabetes Dataset to train a machine learning model that predicts diabetes risk.

Dataset Information:
  Total Records: 768,
  Features: 8,
  Target Variable: Outcome,
    0 → Non-Diabetic,
    1 → Diabetic.

  Features Used:
    Pregnancies,
    Glucose,
    Blood Pressure,
    Skin Thickness,
    Insulin,
    BMI,
    Diabetes Pedigree Function,
    Age.

Machine Learning Approach:
  Checked for missing values.
  Split data into training and testing sets.
  Support Vector Machine (SVM) model is used with Kernel as Linear.
  For model evaluation as Accuracy Score, Classification Report and Confusion Matrix.
  Model Accuracy Achieved: 79.3%.

Web Application:
  The web app is built using Streamlit.
  Users can:
    Enter medical details.
    Click on "Diabetes Test Result".
    Get prediction instantly.

Sample Output:

  If predicted value = 0 → The person is NOT Diabetic,
  
  If predicted value = 1 → The person is Diabetic
