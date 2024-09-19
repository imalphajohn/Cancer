import streamlit as st
import pandas as pd
import numpy as np
import joblib  # To load the saved model

# Load the saved model
model = joblib.load('cancer_model.pkl')

# Load dataset for reference
cancer = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv')

# Define the Streamlit app
def main():
    st.title("Breast Cancer Diagnosis Prediction")
    
    # Input fields for user features
    st.write("Enter the following features to predict diagnosis:")

    # Generate input fields for each feature in the dataset
    features = cancer.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).columns
    input_data = []
    
    for feature in features:
        value = st.number_input(f"Enter value for {feature}", min_value=0.0, format="%.2f")
        input_data.append(value)
    
    # Prediction button
    if st.button("Predict"):
        # Convert the input data to a numpy array
        input_array = np.array(input_data).reshape(1, -1)
        
        # Make prediction using the trained model
        prediction = model.predict(input_array)
        
        # Display the prediction result
        if prediction[0] == 'M':
            st.error("The prediction is: Malignant (Cancerous)")
        else:
            st.success("The prediction is: Benign (Non-Cancerous)")

if __name__ == '__main__':
    main()
