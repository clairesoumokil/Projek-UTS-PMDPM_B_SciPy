import streamlit as st
import pickle
import os
import numpy as np

# Corrected the assignment of the model directory (Dataset file is separate from the model paths)
dataset_path = 'Dataset UTS_Gasal 2425.csv'

# Load the models using correct paths
with open('BestModel_CLF_RandomForest.pkl', 'rb') as clf_file:
    clf_model = pickle.load(clf_file)

with open('BestModel_REG_Ridge.pkl', 'rb') as reg_file:
    reg_model = pickle.load(reg_file)

# Title for the app
st.title("Prediction App Using RandomForest Classifier and Ridge Regressor")

# Sidebar options for model selection
model_type = st.sidebar.selectbox("Choose Model Type", ("Classifier", "Regressor"))

# Input fields based on model type
if model_type == "Classifier":
    st.subheader("Classifier: RandomForest")

    # Input features for the classifier model
    input1 = st.number_input("Input Feature 1 (Classifier)")
    input2 = st.number_input("Input Feature 2 (Classifier)")
    input3 = st.number_input("Input Feature 3 (Classifier)")

    # Prepare data for prediction
    input_data_clf = np.array([[input1, input2, input3]])

    # Predict using the classifier model
    if st.button("Predict Class"):
        prediction_clf = clf_model.predict(input_data_clf)
        st.write(f"Predicted Class: {prediction_clf[0]}")

elif model_type == "Regressor":
    st.subheader("Regressor: Ridge")

    # Input features for the regressor model
    input1 = st.number_input("Input Feature 1 (Regressor)")
    input2 = st.number_input("Input Feature 2 (Regressor)")
    input3 = st.number_input("Input Feature 3 (Regressor)")

    # Prepare data for prediction
    input_data_reg = np.array([[input1, input2, input3]])

    # Predict using the regressor model
    if st.button("Predict Value"):
        prediction_reg = reg_model.predict(input_data_reg)
        st.write(f"Predicted Value: {prediction_reg[0]}")
