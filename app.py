import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model('model.keras')

def predict():
    input_array = np.array([[age, blood_pressure, bmi, glucose ]])
    prediction = model.predict(input_array)
    if prediction[0][0] > 0.5:
        st.error(f'Positive Diabetes. Doctor consultation recommended.')
    elif prediction[0][0] <= 0.5:
        st.success(f'No Diabetes.')
    else:
       st.error('Something went wrong!')

 
age = st.number_input('Age', min_value=0, max_value=100, value=30)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=120)
bmi = st.number_input('BMI', min_value=0, max_value=50, value=20)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=80)
                                

st.button('Predict', on_click=predict)
