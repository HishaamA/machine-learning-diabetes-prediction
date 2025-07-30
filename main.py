import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav', 'rb'))


def diabetes_prediction(inputdata):
    input_data = np.array(inputdata)
    input_data_reshaped = input_data.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return "This person is not diabetic"
    else:
        return "This person is diabetic"


def main():
    st.title("Diabetes Prediction ML App")
    pregnancies = st.text_input("Number of Pregnancies")
    glucose = st.text_input("Glucose Level")
    blood_pressure = st.text_input("Blood Pressure value")
    skin_thickness = st.text_input("Skin Thickness value")
    insulin = st.text_input("Insulin value")
    bmi = st.text_input("BMI value")
    dpf = st.text_input("Diabetes Pedigree Function value")
    age = st.text_input("Age of the Person")

    diagnosis = ''

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,dpf,age])

    st.success(diagnosis)


if __name__ =='__main__':
    main()