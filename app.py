import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):

    st.write("🔄 Training model... Please wait")

    # Load dataset
    df = pd.read_csv("crop_production.csv")

    # Data Cleaning
    df = df.dropna()
    df = df[['State_Name', 'Crop', 'Season', 'Area', 'Production']]

    # Create Yield column
    df['Yield'] = df['Production'] / df['Area']

    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Encoding
    le_state = LabelEncoder()
    le_crop = LabelEncoder()
    le_season = LabelEncoder()

    df['State_Name'] = le_state.fit_transform(df['State_Name'])
    df['Crop'] = le_crop.fit_transform(df['Crop'])
    df['Season'] = le_season.fit_transform(df['Season'])

    # Features & Target
    X = df[['State_Name', 'Crop', 'Season', 'Area']]
    y = df['Yield']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    # Save everything
    joblib.dump(model, "model.pkl")
    joblib.dump(le_state, "le_state.pkl")
    joblib.dump(le_crop, "le_crop.pkl")
    joblib.dump(le_season, "le_season.pkl")

    st.success("✅ Model trained and saved!")

model = joblib.load("model.pkl")
le_state = joblib.load("le_state.pkl")
le_crop = joblib.load("le_crop.pkl")
le_season = joblib.load("le_season.pkl")

st.set_page_config(page_title="Crop Yield Predictor", layout="centered")

st.title("🌾 Crop Yield Prediction System")
st.write("Enter details to predict crop yield (kg/hectare)")

state = st.selectbox("Select State", le_state.classes_)
crop = st.selectbox("Select Crop", le_crop.classes_)
season = st.selectbox("Select Season", le_season.classes_)

area = st.number_input("Enter Area (in hectares)", min_value=0.1)

state_encoded = le_state.transform([state])[0]
crop_encoded = le_crop.transform([crop])[0]
season_encoded = le_season.transform([season])[0]

if st.button("Predict Yield"):
    input_data = np.array([[state_encoded, crop_encoded, season_encoded, area]])
    prediction = model.predict(input_data)

    st.success(f"🌾 Predicted Yield: {prediction[0]:.2f} kg/hectare")

if st.checkbox("Show Dataset Insights"):
    df = pd.read_csv("crop_production.csv")
    df = df.dropna()
    df['Yield'] = df['Production'] / df['Area']

    st.subheader("📊 Sample Data")
    st.write(df.head())

    st.subheader("📈 Yield Distribution")
    st.bar_chart(df['Yield'].head(50))