import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

# Load the scaler and model at the start to ensure they are ready when needed
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

try:
    model = tf.keras.models.load_model('model.keras')  # Load the model only once at the start
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # In case of error, we set the model to None

# Prediction function
def predict():
    if model is None:  # Check if the model is loaded
        st.error("Model is not loaded. Please try again.")
        return
    
    # Convert input values into a 2D array with shape (1, 4)
    input_array = np.array([[float(age), float(gender), float(bmi), float(glucose)]])  # Shape (1, 4)
    
    # Scale the input
    try:
        input_array = scaler.transform(input_array)  # Apply scaling
    except Exception as e:
        st.error(f"Error in scaling: {e}")
        return

    # Make the prediction
    prediction = model.predict(input_array)

    # Interpret the prediction
    if prediction[0][0] > 0.5:
        st.error('Positive Diabetes. Doctor consultation recommended.')
    else:
        st.success('No Diabetes.')

# Streamlit input fields
age = st.number_input('Age', min_value=0, max_value=100, value=30)
gender = st.selectbox('Gender', options=[('Male', 1), ('Female', 2)], format_func=lambda x: x[0])[1]  # Get the numeric value (1 or 2)
bmi = st.number_input('BMI', min_value=0, max_value=50, value=20)
glucose = st.number_input('Blood Glucose Level', min_value=0, max_value=200, value=80)

# Predict button
st.button('Predict', on_click=predict)