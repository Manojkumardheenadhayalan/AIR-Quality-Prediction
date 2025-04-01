import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    st.title("Model Training & Evaluation")
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

    X = df[["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "RAIN", "WSPM"]]
    y = df["PM2.5"]

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    model = train_model(model_name, X_train, y_train)
    
    st.markdown(f"#### Training {model_name}")
    st.write(f"{model_name} trained successfully!")
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    st.markdown("### Model Performance")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R-squared (R2):** {r2:.2f}")
    
    if hasattr(model, "feature_importances_"):
        st.markdown("### Feature Importance")
        feature_importance = model.feature_importances_
        feature_names = X.columns
        feature_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
        feature_df = feature_df.sort_values(by="Importance", ascending=False)
        st.dataframe(feature_df)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_df["Feature"], feature_df["Importance"], color="skyblue")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.gca().invert_yaxis()
        st.pyplot(plt)
    
    st.markdown("### Predicted vs Actual Values")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.6, color="purple")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="Perfect Prediction Line")
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.legend()
    st.pyplot(plt)
    
    residuals = y_test - predictions
    st.markdown("### Residual Analysis")
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)), residuals, alpha=0.6, color="green")
    plt.axhline(0, color="red", linestyle="--", label="Zero Residual Line")
    plt.xlabel("Sample Index")
    plt.ylabel("Residual")
    plt.legend()
    st.pyplot(plt)
    
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, color="orange", edgecolor="black", alpha=0.7)
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    st.pyplot(plt)

if __name__ == "__main__":
    run()
