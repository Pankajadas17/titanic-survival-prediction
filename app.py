import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("model.pkl")

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.write("Predict whether a passenger would survive the Titanic disaster.")

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------
st.header("Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["male", "female"])
age = st.slider("Age", 1, 80, 30)
fare = st.slider("Fare Paid", 0.0, 500.0, 32.0)
sibsp = st.number_input("Siblings / Spouses Aboard", 0, 8, 0)
parch = st.number_input("Parents / Children Aboard", 0, 6, 0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Rare"])

# -----------------------------
# Feature Engineering (SAME AS TRAINING)
# -----------------------------
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# Encode categorical features
sex_male = 1 if sex == "male" else 0
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

title_miss = 1 if title == "Miss" else 0
title_mr = 1 if title == "Mr" else 0
title_mrs = 1 if title == "Mrs" else 0
title_rare = 1 if title == "Rare" else 0

# -----------------------------
# Final Input Vector (MATCH TRAINING ORDER)
# -----------------------------
input_data = np.array([[
    pclass,
    age,
    sibsp,
    parch,
    fare,
    family_size,
    is_alone,
    sex_male,
    embarked_q,
    embarked_s,
    title_miss,
    title_mr,
    title_mrs,
    title_rare
]])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"🟢 Passenger is likely to SURVIVE\n\nProbability: {probability:.2%}")
    else:
        st.error(f"🔴 Passenger is likely to NOT SURVIVE\n\nProbability: {probability:.2%}")
