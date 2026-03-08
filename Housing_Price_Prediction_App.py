import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# load dataset
data = pd.read_csv("Housing_data.csv")

# features and target
X = data.drop("MedHouseValue", axis=1)
y = data["MedHouseValue"]

# Ensure model doesn't retrain every time the app refreshes
@st.cache_resource
def train_model():
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model()

st.title("Housing Price Prediction")

# Briefly describe the app
st.write("""
This machine learning app predicts California housing prices based on
economic and geographic features from the California Housing dataset.
Enter values below to estimate a house value.
""")

# Include sliders for fast and engaging adjustments
MedInc = st.slider("Median Income", 0.0, 15.0, 3.0)
HouseAge = st.slider("House Age", 1, 50, 20)
AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.slider("Population", 0, 5000, 1000)
AveOccup = st.slider("Average Occupancy", 1.0, 10.0, 3.0)
Latitude = st.slider("Latitude", 32.0, 42.0, 36.0)
Longitude = st.slider("Longitude", -125.0, -114.0, -120.0)

# User can see the data used to train the models
st.subheader("Dataset Preview")
st.dataframe(data.head())

if st.button("Predict"):

    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]])

    prediction = model.predict(features)

    st.success(f"Predicted House Value: ${prediction[0]*100000:.2f}")
