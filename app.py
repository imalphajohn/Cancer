import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load and preprocess the data
cancer = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv')
y = cancer['diagnosis']
X = cancer[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']]  # Select important features

# Step 2: Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)

# Step 3: Train the model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Step 4: Define the Streamlit app
def main():
    st.title("Breast Cancer Prediction")

    # Dropdowns for user input
    radius_mean = st.selectbox("Radius Mean", np.arange(6.0, 30.0, 0.1))
    texture_mean = st.selectbox("Texture Mean", np.arange(9.0, 40.0, 0.1))
    perimeter_mean = st.selectbox("Perimeter Mean", np.arange(40.0, 190.0, 1.0))
    area_mean = st.selectbox("Area Mean", np.arange(100.0, 2500.0, 10.0))
    smoothness_mean = st.selectbox("Smoothness Mean", np.arange(0.05, 0.17, 0.001))

    # Predict button
    if st.button("Predict"):
        # Prepare data for prediction
        input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]])
        
        # Predict the outcome
        prediction = model.predict(input_data)[0]

        # Display result
        if prediction == 'M':
            st.write("The prediction is: **Malignant** (Cancerous)")
        else:
            st.write("The prediction is: **Benign** (Non-cancerous)")

# Run the Streamlit app
if __name__ == '__main__':
    main()
