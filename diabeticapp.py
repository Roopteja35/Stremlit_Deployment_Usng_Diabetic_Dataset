
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# load the dataset
df=pd.read_csv('diabetes.csv')
df.head()

df.info()
df.isnull().sum()
df.describe()
df['Outcome'].value_counts()

# Visualization
df.groupby('Outcome').mean().plot(kind="bar")
df.groupby('Outcome')['Pregnancies'].mean().plot(kind="bar")
df.pivot_table(index='Outcome', columns='Pregnancies', values='Age', aggfunc='mean').fillna(0)

# Separate Features and Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9,stratify=y)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

array=np.array([[5,17,19,32,87,54,67,25.6]]).reshape(1,-1)

pred_array=model.predict(array)
if pred_array[0]==0:
  print("The person is not diabetic")
else:
  print("The person is diabetic")

def main():
  st.title("Diabetics Prediction for number of pregnancies")
  pregnancies=st.text_input("Num of pregnancies")
  glucose=st.text_input("Enter the Glucose level")
  blood_pressure=st.text_input("Enter the Blood Pressure")
  skin_thickness=st.text_input("Enter the Skin Thickness")
  insulin=st.text_input("Enter the Insulin")
  BMI=st.text_input("Enter the BMI")
  Diabetes_Pedigree_Function=st.text_input("Enter the Diabetes Pedigree Function")
  age=st.text_input("Enter the Age")

  diagnosis=" "

  if st.button("diabets test results"):
    #input_data = np.array([[
    #      float(pregnancies),
    #      float(glucose),
    #      float(blood_pressure),
    #      float(skin_thickness),
    #      float(insulin),
    #      float(BMI),
    #      float(Diabetes_Pedigree_Function),
    #      float(age)]])
    diagnosis=model.predict([[pregnancies,glucose,blood_pressure,skin_thickness,insulin,BMI,Diabetes_Pedigree_Function,age]])
    #diagnosis=model.predict(input_data)
    if diagnosis[0]==0:
      diagnosis="The person is not diabetic"
    else:
      diagnosis="The person is diabetic"
    st.success(diagnosis)

main()