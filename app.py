import streamlit as st
import numpy as np
from keras.models import load_model
import joblib
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0;  /* Light Gray */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the saved model and scaler
model_path = r"C:\Users\8858s\OneDrive\Desktop\Heart Disease\heart_disease_model.h5"
scaler_path = r"C:\Users\8858s\OneDrive\Desktop\Heart Disease\scaler.pkl"

classifier = load_model(model_path)
sc = joblib.load(scaler_path)

# Streamlit app
st.title("Heart Disease Prediction")

st.sidebar.header("Input Features")

def user_input_features():
    age = st.sidebar.number_input('Age (years)', min_value=9, max_value=77, value=54, step=1)
    sex = st.sidebar.selectbox('Sex (1=Male, 0=Female)', [1, 0])
    cp = st.sidebar.number_input('Chest Pain Type (1-4)', min_value=1, max_value=4, value=3, step=1)
    trestbps = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', min_value=94, max_value=200, value=131, step=1)
    chol = st.sidebar.number_input('Serum Cholesterol (mg/dl)', min_value=126, max_value=564, value=246, step=1)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar (>120 mg/dl, 1=Yes, 0=No)', [1, 0])
    restecg = st.sidebar.number_input('Resting ECG Results (0-2)', min_value=0, max_value=2, value=1, step=1)
    thalach = st.sidebar.number_input('Max Heart Rate Achieved', min_value=71, max_value=202, value=150, step=1)
    exang = st.sidebar.selectbox('Exercise Induced Angina (1=Yes, 0=No)', [1, 0])
    oldpeak = st.sidebar.number_input('ST Depression Induced', min_value=0.0, max_value=6.2, value=1.0, step=0.1)
    slope = st.sidebar.number_input('Slope of the Peak (1-3)', min_value=1, max_value=3, value=2, step=1)
    ca = st.sidebar.number_input('Number of Major Vessels (0-3)', min_value=0, max_value=3, value=0, step=1)
    thal = st.sidebar.number_input('Thalassemia (3=Normal, 6=Fixed, 7=Reversible)', min_value=3, max_value=7, value=3, step=1)
    
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    return features

input_data = user_input_features()

# Standardize the input data using the saved scaler
input_data_scaled = sc.transform(input_data)

# Make predictions using the model
prediction = classifier.predict(input_data_scaled)[0][0]

# Display the result
st.subheader("Prediction Result")

if prediction > 0.5:
    st.write("High chance of heart disease. (Prediction: {:.2f})".format(prediction))
else:
    st.write("Low chance of heart disease. (Prediction: {:.2f})".format(prediction))

st.write("""
### Note:
This is a demonstration app and not intended for clinical use. Please consult a doctor.
""")