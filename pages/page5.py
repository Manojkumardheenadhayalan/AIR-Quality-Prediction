# pages/page4.py
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def add_custom_css():
    # CSS for black background
    st.markdown(
        """
        <style>
        /* Set the background to black */
        body {
            background-color: black;
            color: white; /* Ensure text is visible on a black background */
        }

        /* Modify headers and links for better visibility */
        h1, h2, h3, h4, h5, h6 {
            color: #00FF00; /* Green text for headers */
        }

        a {
            color: #1E90FF; /* Blue text for links */
        }

        /* Info and button sections styling */
        .st-info {
            background-color: #333333; /* Dark gray background for info boxes */
            color: white;
        }

        button {
            background-color: #444444; /* Dark gray buttons */
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def run():
    # Add custom CSS
    add_custom_css()
    st.title("Interactive Prediction Tool")

    df = pd.read_csv("PRSA_Data_Changping_20130301-20170228.csv")

    feature_cols = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "RAIN", "WSPM"]
    target_col = "PM2.5"

    X = df[feature_cols]
    y = df[target_col]
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # Split data into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train model
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    
    user_input = {}
    for feature in feature_cols:
        user_input[feature] = st.number_input(f"Enter {feature} value:", value=float(X[feature].mean()))

    # Predict for user input
    if st.button("Predict PM2.5 Concentration"):
        user_data = np.array([list(user_input.values())])
        user_prediction = model.predict(user_data)
        st.write(f"**Predicted PM2.5 Concentration:** {user_prediction[0]:.2f}")

if __name__ == "__main__":
    run()
