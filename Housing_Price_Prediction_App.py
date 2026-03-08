import streamlit as st
import pickle
import numpy as np

import pandas as pd
from sklearn.linear_model import LinearRegression

# load dataset
data = pd.read_csv("Housing_data.csv")
print(data.columns)
X = data.drop("Predicted Price", axis=1)
y = data["Predicted Price"]

model = LinearRegression()
model.fit(X, y)

st.title("Housing Price Prediction")

MedInc = st.number_input("Median Income")
HouseAge = st.number_input("House Age")
AveRooms = st.number_input("Average Rooms")
AveBedrms = st.number_input("Average Bedrooms")
Population = st.number_input("Population")
AveOccup = st.number_input("Average Occupancy")
Latitude = st.number_input("Latitude")
Longitude = st.number_input("Longitude")

if st.button("Predict"):

    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]])

    prediction = model.predict(features)

    st.success(f"Predicted Price: ${prediction[0]*100000:.2f}")
