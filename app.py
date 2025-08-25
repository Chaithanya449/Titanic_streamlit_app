# app.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib

# --- Load the Trained Model and Scaler ---
try:
    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler files not found. Please ensure 'logistic_regression_model.pkl' and 'scaler.pkl' are in the same directory as app.py.")
    st.stop()

# --- Set up the User Interface ---
st.title('Titanic Survival Prediction')
st.write("This app uses a Logistic Regression model to predict whether a passenger would have survived the Titanic disaster.")

st.header("Enter Passenger Details:")

# Create user inputs for all the features
pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', min_value=0, max_value=100, value=29, step=1)
sibsp = st.number_input('Number of Siblings/Spouses Aboard (SibSp)', min_value=0, max_value=10, value=0, step=1)
parch = st.number_input('Number of Parents/Children Aboard (Parch)', min_value=0, max_value=10, value=0, step=1)
fare = st.number_input('Fare', min_value=0.0, max_value=600.0, value=32.0, format="%.2f")
embarked = st.selectbox('Port of Embarkation', ['Southampton', 'Cherbourg', 'Queenstown'])

# --- Prediction Logic ---
if st.button('Predict Survival'):
    
    # 1. Preprocess the user input
    sex_numeric = 0 if sex == 'male' else 1
    embarked_Q = 1 if embarked == 'Queenstown' else 0
    embarked_S = 1 if embarked == 'Southampton' else 0

    # 2. Create a DataFrame from the user input in the correct order
    user_input = pd.DataFrame({
        'Pclass': [pclass], 'Age': [age], 'SibSp': [sibsp], 'Parch': [parch],
        'Fare': [fare], 'Sex': [sex_numeric], 'Embarked_Q': [embarked_Q], 'Embarked_S': [embarked_S]
    })
    
    # 3. Scale the user input
    user_input_scaled = scaler.transform(user_input)
    
    # 4. Make a prediction
    prediction = model.predict(user_input_scaled)
    prediction_proba = model.predict_proba(user_input_scaled)
    
    # 5. Display the result
    st.header("Prediction Result")
    if prediction[0] == 1:
        st.success(f"This passenger would have likely **SURVIVED**.")
        st.write(f"Probability of Survival: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.error(f"This passenger would have likely **NOT SURVIVED**.")
        st.write(f"Probability of Not Surviving: {prediction_proba[0][0]*100:.2f}%")
