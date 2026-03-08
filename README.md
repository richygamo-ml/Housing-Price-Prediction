# Housing Price Prediction Web App

This project is an interactive machine learning web application that predicts California 
housing prices based on economic and geographic features.

The application is built using Python and deployed with Streamlit.

## Features

- Predict housing prices from user inputs
- Interactive sliders for feature selection
- Model evaluation metrics (R², MAE, MSE)
- Actual vs Predicted visualization
- Feature importance analysis
- Dataset preview

## Machine Learning Model

The model uses Linear Regression trained on the California Housing dataset.

Target Variable:
MedHouseValue (median house value)

Key Features:
- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude
- Longitude

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Streamlit

## Run Locally

Clone the repository

git clone https://github.com/richygamo-ml/Housing-Price-Prediction.git

Navigate to the project folder

cd Housing-Price-Prediction

Install dependencies

pip install -r requirements.txt

Run the app

streamlit run Housing_Price_Prediction_App.py

## Live App

The application is deployed using Streamlit Cloud.
