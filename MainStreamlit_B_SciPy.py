import streamlit as st
import pickle
import os
import numpy as np

model_directory = r'Dataset UTS_Gasal 2425.csv'  # Corrected assignment

# Load the models 
with open('/mnt/data/BestModel_CLF_RandomForest.pkl', 'rb') as f:
    clf_model = pickle.load(f)

with open('/mnt/data/BestModel_REG_Ridge.pkl', 'rb') as f:
    reg_model = pickle.load(f)
# Title for the app
st.title("Prediction App Using RandomForest Classifier and Ridge Regressor")

# Sidebar options for model selection
model_type = st.sidebar.selectbox("Choose Model Type", ("Classifier", "Regressor"))

# Create input fields for the user based on your model's feature requirements
if model_type == "Classifier":
    st.subheader("Classifier: RandomForest")
    
    # Input features for the classifier model (customize based on your features)
    input1 = st.number_input("Input Feature 1 (Classifier)")
    input2 = st.number_input("Input Feature 2 (Classifier)")
    input3 = st.number_input("Input Feature 3 (Classifier)")
    
    # Prepare the data for prediction
    input_data_clf = np.array([[input1, input2, input3]])  # Modify this according to your model
    
    # Predict using the classifier model
    if st.button("Predict Class"):
        prediction_clf = clf_model.predict(input_data_clf)
        st.write(f"Predicted Class: {prediction_clf[0]}")

elif model_type == "Regressor":
    st.subheader("Regressor: Ridge")
    
    # Input features for the regressor model (customize based on your features)
    input1 = st.number_input("Input Feature 1 (Regressor)")
    input2 = st.number_input("Input Feature 2 (Regressor)")
    input3 = st.number_input("Input Feature 3 (Regressor)")
    
    # Prepare the data for prediction
    input_data_reg = np.array([[input1, input2, input3]])  # Modify this according to your model
    
    # Predict using the regressor model
    if st.button("Predict Value"):
        prediction_reg = reg_model.predict(input_data_reg)
        st.write(f"Predicted Value: {prediction_reg[0]}")

