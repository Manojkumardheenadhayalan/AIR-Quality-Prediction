import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import importlib
def set_black_background_and_sidebar():
    st.markdown(
        """
        <style>
        /* General background styling for the app */
        .stApp {
            background-color: black;
            color: white;
        }

        /* Sidebar styling */
        .css-1v3fvcr {
            background-color: black !important; /* Black background */
        }
        .css-1v3fvcr, .css-1v3fvcr div, .css-1v3fvcr div span, .css-1v3fvcr div label {
            color: black !important; 
        }

        /* Change the background color of the sidebar */
        .sidebar .sidebar-content {
            background-color: #31333F;
        }
        /* Change the text color of the sidebar */
        .sidebar .sidebar-content {
            color: blue;
        }

        [data-testid="stSidebarContent"] { color: white; background-color: black; }
        [data-testid="stHeader"] { color: white; background-color: black; }
        /* Button and interactive elements styling */
        .stButton > button {
            background-color: #333333; /* Dark gray buttons */
            color: white; /* White text */
        }

        /* Styling headers and text elements globally */
        h1, h2, h3, h4, h5, h6, p, div, span, label {
            color: white;
        }

        /* Styling dataframes */
        .st-dataframe {
            background-color: #222222;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# Set page configuration
st.set_page_config(page_title="Beijing Air Quality Analysis", layout="wide")

def main():
    # Apply custom CSS
    set_black_background_and_sidebar()

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    pages = {
        "Introduction": "page1",
        "EDA": "page2",
        "Modeling and Prediction": "page3",
        "Prediction and Visualization": "page4",
        "Interactive Prediction Dashboard": "page5"
    }

    choice = st.sidebar.radio("Go to", list(pages.keys()))

    # Dynamically load the selected page
    selected_page = pages[choice]
    page_module = importlib.import_module(f"pages.{selected_page}")
    page_module.run()
if __name__ == "__main__":
    main()
