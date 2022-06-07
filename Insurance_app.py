import streamlit as st
import pickle
import numpy as np
def load_model():
    with open('steps.pkl','rb') as file:
        data = pickle.load(file)
    return data 

data = load_model()
model=data["model"]
le_sex=data["le_sex"]
le_region=data["le_region"]
le_smoker=data["le_smoker"]
st.title("Insurance Charge Prediction")
region=("northeast","northwest","southeast","southwest")
sex=("male","female")
smoker=("yes","no")
region=st.selectbox("Select your region",region)
sex=st.selectbox("Gender",sex)
smoker=st.selectbox("Smoker",smoker)
children=st.slider("No of children",0,7)
age=st.slider("Age",18,65)
bmi=st.slider("BMI",15.0,50.0)
cl=st.button("Predict Charges")
if cl:
    t=np.array([[age,sex,bmi,children,smoker,region]])
    t[:,1]=le_sex.transform(t[:,1])
    t[:,4]=le_smoker.transform(t[:,4])
    t[:,5]=le_region.transform(t[:,5])
    t=t.astype(float)
    charge=model.predict(t)
    st.subheader(f"The Cost Of Your Insurance is ${charge[0]:.2f}")