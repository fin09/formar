import os
import joblib
import numpy as np
import streamlit as st
from pathlib import Path
import sys

# Get the current directory of the script or fallback to the current working directory
if hasattr(sys, 'frozen'):
    current_dir = Path(sys.executable).parent
elif '__file__' in globals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = Path().absolute()

# Load the trained model from the same directory
model_path = os.path.join(current_dir, "trained_model.joblib")
model = joblib.load(model_path)

# Define the prediction function
def predict_department(input_data):
    try:
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Map the prediction to department name
        department_mapping = {0: 'Swe', 1: 'Cs', 2: 'Cne', 3: 'Ai'}
        predicted_department = department_mapping[prediction[0]]

        return predicted_department

    except Exception as e:
        return str(e)

# Define the Streamlit interface
st.title("Department Predictor")

# Create input fields for all course totals
input_labels = ["CSC101_total", "CSC201_total", "CSC203_total", "CSC205_total", "CSC102_total", 
                "MAT202_total", "MAT203_total", "MAT103_total", "CSC206_total", "MAN101_total", 
                "SWE201_total", "SWE301_total", "SWE303_total", "CNE202_total", "CNE203_total", 
                "CNE304_total", "CSC301_total", "CNE302_total", "CSC309_total", "CSC302_total", 
                "CSC303_total", "CNE308_total"]

# Store user inputs in a list
input_data = []
for label in input_labels:
    value = st.number_input(f"{label}", min_value=0, max_value=100, step=1, value=3)
    input_data.append(value)

# Convert the input data to a numpy array
input_data = np.array([input_data])

# Prediction button
if st.button('Predict Department'):
    predicted_department = predict_department(input_data)
    st.write(f"Predicted Department: {predicted_department}")
