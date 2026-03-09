import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("Housing_data.csv")

# Features and target
X = data.drop("MedHouseValue", axis=1)
y = data["MedHouseValue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model and ensure model doesn't retrain every time the app refreshes
@st.cache_resource
def train_model():
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model()

st.title("Housing Price Prediction")

# Briefly describe the app
st.sidebar.title("About")

st.sidebar.write("""
This machine learning app predicts California housing prices using a Linear Regression model trained on the California Housing dataset.

Features include:
- Median income
- House age
- Average rooms
- Population
- Geographic location
""")

# Include sliders for fast and engaging adjustments
MedInc = st.slider("Median Income (× $100,000)", 0.0, 15.0, 3.0)
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
    st.caption("Prediction represents the estimated median house value based on the selected housing and geographic features.")

# Define predicted target on the test set (not to be mistaken for initial target, y)
y_predicted = model.predict(X_test)

# Metrics evaluation
st.header("Model Performance")
st.caption(
"Model performance is evaluated on a 20% test dataset that was not used during training.")

r2 = r2_score(y_test, y_predicted)
mae = mean_absolute_error(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)

# Dislay metrics in the app
st.subheader("Model Evaluation Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("R² Score", f"{r2:.3f}")
col2.metric("MAE", f"{mae:.3f}")
col3.metric("MSE", f"{mse:.3f}")

# Visualization
st.subheader("Actual vs Predicted Housing Prices")

fig, ax = plt.subplots()

ax.scatter(y_test, y_predicted, alpha=0.5)
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.set_title("Actual vs Predicted Housing Prices")

st.pyplot(fig)

# Let's check which feature(s) is/are the most important
st.subheader("Feature Importance")

importance = model.coef_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(by="Importance", ascending=False)

fig2, ax2 = plt.subplots()

ax2.barh(importance_df["Feature"], importance_df["Importance"])
ax2.set_xlabel("Coefficient Value")
ax2.set_title("Feature Importance")

st.pyplot(fig2)