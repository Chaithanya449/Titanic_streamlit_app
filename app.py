

import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("logistic_model.pkl")

st.title("🚢 Titanic Survival Predictor")
st.write("Fill in passenger details to predict survival probability.")

# --- User Input Section ---
st.header("Passenger Details")

pclass   = st.selectbox("Passenger Class", [1, 2, 3], help="1=First, 2=Second, 3=Third")
sex      = st.selectbox("Sex", ["male", "female"])
age      = st.slider("Age", 1, 80, 25)
sibsp    = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
parch    = st.number_input("Parents/Children Aboard", min_value=0, max_value=6, value=0)
fare     = st.number_input("Fare Paid ($)", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"], help="C=Cherbourg, Q=Queenstown, S=Southampton")

# Encode inputs to match training preprocessing
sex_encoded      = 1 if sex == "female" else 0
embarked_encoded = {"C": 0, "Q": 1, "S": 2}[embarked]

# Create input dataframe
input_data = pd.DataFrame({
    "Pclass"  : [pclass],
    "Sex"     : [sex_encoded],
    "Age"     : [age],
    "SibSp"   : [sibsp],
    "Parch"   : [parch],
    "Fare"    : [fare],
    "Embarked": [embarked_encoded]
})

# --- Predict ---
if st.button("Predict Survival"):
    prediction      = model.predict(input_data)[0]
    survival_prob   = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"**Survived** — Probability: {survival_prob:.2%}")
    else:
        st.error(f"**Did Not Survive** — Probability: {survival_prob:.2%}")

    st.write(f"**Survival Probability:** `{survival_prob:.4f}`")