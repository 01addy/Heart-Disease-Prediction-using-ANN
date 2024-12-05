import streamlit as st
import numpy as np
from keras.models import load_model
import joblib


st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f9; /* Light Gray */
    }
    .header {
        background-color: #007bff; /* Blue Background */
        padding: 10px;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .result-box {
        background-color: #f8d7da; /* Light Red for High Risk */
        padding: 15px;
        border-radius: 5px;
        color: #721c24; /* Dark Red Text */
        font-size: 18px;
        font-weight: bold;
    }
    .summary-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
    }
    .summary-table th {
        background-color: #007bff; /* Blue Header */
        color: white;
        text-align: left;
        padding: 10px;
    }
    .summary-table td {
        background-color: #ffffff; /* White Cell Background */
        color: black;
        padding: 10px;
    }
    .summary-table th, .summary-table td {
        border: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


model_path = r"C:\Users\8858s\OneDrive\Desktop\Heart Disease\heart_disease_model.h5"#Path to heart_disease_model.h5
scaler_path = r"C:\Users\8858s\OneDrive\Desktop\Heart Disease\scaler.pkl"#path to scaler.pkl

classifier = load_model(model_path)
sc = joblib.load(scaler_path)


st.markdown("<div class='header'><h1>Heart Disease Prediction</h1></div>", unsafe_allow_html=True)

st.sidebar.header("Input Features")


tooltips = {
    "Age": "The age of the individual in years.",
    "Sex": "Biological sex of the individual (1=Male, 0=Female).",
    "Chest Pain Type": "Type of chest pain experienced (1=Typical Angina, 2=Atypical Angina, 3=Non-Anginal Pain, 4=Asymptomatic).",
    "Resting Blood Pressure": "The individual's resting blood pressure (in mm Hg).",
    "Serum Cholesterol": "The individual's serum cholesterol level (in mg/dl).",
    "Fasting Blood Sugar": "Whether fasting blood sugar is >120 mg/dl (1=Yes, 0=No).",
    "Resting ECG Results": "Electrocardiogram results (0=Normal, 1=ST-T wave abnormality, 2=Left ventricular hypertrophy).",
    "Max Heart Rate Achieved": "The maximum heart rate achieved during exercise.",
    "Exercise Induced Angina": "Whether exercise-induced angina is present (1=Yes, 0=No).",
    "ST Depression Induced": "ST depression induced by exercise relative to rest.",
    "Slope of the Peak": "The slope of the peak exercise ST segment (1=Upsloping, 2=Flat, 3=Downsloping).",
    "Number of Major Vessels": "Number of major vessels colored by fluoroscopy (0-3).",
    "Thalassemia": "Blood disorder type (3=Normal, 6=Fixed defect, 7=Reversible defect).",
}
def user_input_features():
    """Collect user inputs from sidebar."""
    inputs = {}
    inputs["Age"] = int(st.sidebar.number_input("Age (years)", min_value=9, max_value=77, value=54, step=1, help=tooltips["Age"]))
    inputs["Sex"] = int(st.sidebar.selectbox("Sex (1=Male, 0=Female)", [1, 0], help=tooltips["Sex"]))
    inputs["Chest Pain Type"] = st.sidebar.number_input("Chest Pain Type (1-4)", min_value=1, max_value=4, value=3, step=1, help=tooltips["Chest Pain Type"])
    inputs["Resting Blood Pressure"] = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=94, max_value=200, value=131, step=1, help=tooltips["Resting Blood Pressure"])
    inputs["Serum Cholesterol"] = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=126, max_value=564, value=246, step=1, help=tooltips["Serum Cholesterol"])
    inputs["Fasting Blood Sugar"] = st.sidebar.selectbox("Fasting Blood Sugar (>120 mg/dl, 1=Yes, 0=No)", [1, 0], help=tooltips["Fasting Blood Sugar"])
    inputs["Resting ECG Results"] = st.sidebar.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, value=1, step=1, help=tooltips["Resting ECG Results"])
    inputs["Max Heart Rate Achieved"] = st.sidebar.number_input("Max Heart Rate Achieved", min_value=71, max_value=202, value=150, step=1, help=tooltips["Max Heart Rate Achieved"])
    inputs["Exercise Induced Angina"] = st.sidebar.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [1, 0], help=tooltips["Exercise Induced Angina"])
    inputs["ST Depression Induced"] = st.sidebar.number_input("ST Depression Induced", min_value=0.0, max_value=6.2, value=1.0, step=0.1, help=tooltips["ST Depression Induced"])
    inputs["Slope of the Peak"] = st.sidebar.number_input("Slope of the Peak (1-3)", min_value=1, max_value=3, value=2, step=1, help=tooltips["Slope of the Peak"])
    inputs["Number of Major Vessels"] = st.sidebar.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0, step=1, help=tooltips["Number of Major Vessels"])
    inputs["Thalassemia"] = st.sidebar.number_input("Thalassemia (3=Normal, 6=Fixed, 7=Reversible)", min_value=3, max_value=7, value=3, step=1, help=tooltips["Thalassemia"])

    return np.array([[value for value in inputs.values()]]), inputs


input_data, input_summary = user_input_features()


input_data_scaled = sc.transform(input_data)


prediction = classifier.predict(input_data_scaled)[0][0]
confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100


st.subheader("Prediction Result")
result_color = "background-color: #d4edda; color: #155724;" if prediction <= 0.5 else "background-color: #f8d7da; color: #721c24;"
if prediction > 0.5:
    st.markdown(
        f"<div class='result-box'>High chance of heart disease. Confidence: {confidence:.2f}%</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<div class='result-box' style='{result_color}'>Low chance of heart disease. Confidence: {confidence:.2f}%</div>",
        unsafe_allow_html=True,
    )


st.write("### Input Summary")
table_header = """
<table class="summary-table">
<thead>
<tr>
<th>Feature</th>
<th>Value</th>
</tr>
</thead>
<tbody>
"""
table_rows = "".join(f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in input_summary.items())
table_footer = "</tbody></table>"
st.markdown(table_header + table_rows + table_footer, unsafe_allow_html=True)

st.write("""
### Note:
This is not intended for clinical use.
""")
