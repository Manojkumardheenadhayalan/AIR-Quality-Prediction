import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: black;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #00FF00;
        }
        a {
            color: #1E90FF;
        }
        .st-info {
            background-color: #333333;
            color: white;
        }
        button {
            background-color: #444444;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def train_model(model_name, X_train, y_train):
    models = {
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Linear Regression": LinearRegression(),
        "AdaBoost Regressor": AdaBoostRegressor(),
        "Ridge Regression": Ridge()
    }
    model = models[model_name]
    model.fit(X_train, y_train)
    return model

def run():
    add_custom_css()
    st.title("Predictions & Visualizations")
    
    st.markdown("### Select Regression Model")
    model_name = st.selectbox("Choose a model:", [
        "Gradient Boosting Regressor", 
        "Random Forest Regressor", 
        "Linear Regression", 
        "AdaBoost Regressor", 
        "Ridge Regression"
    ])
    
    df = pd.read_csv("PRSA_Data_Changping_20130301-20170228.csv")
    st.markdown("#### Dataset Overview")
    st.write(df.head())

    feature_cols = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "RAIN", "WSPM"]
    target_col = "PM2.5"

    X = df[feature_cols]
    y = df[target_col]
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = train_model(model_name, X_train, y_train)
    predictions = model.predict(X_test)

    st.write("#### Example Predictions (First 10)")
    results_df = pd.DataFrame({
        "Actual": y_test[:10].values,
        "Predicted": predictions[:10]
    }).reset_index(drop=True)
    st.dataframe(results_df)

    st.markdown("### Prediction Performance")
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R-squared (R2):** {r2:.2f}")

    st.markdown("### Visualization: Actual vs Predicted")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.6, color="blue", label="Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="Perfect Prediction")
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.legend()
    st.pyplot(plt)

    st.markdown("### Visualization: Error Distribution")
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, color="purple", edgecolor="black", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--", label="Zero Error Line")
    plt.xlabel("Residual Value (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.legend()
    st.pyplot(plt)

if __name__ == "__main__":
    run()
