import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# ✅ Rebuild model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(24, 1)),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])

# ✅ Load weights
model.load_weights(r"C:\Users\js280\traffic_lstm_model.h5")

# ✅ Load scaler
with open(r"C:\Users\js280\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# UI
st.title("Traffic Volume Prediction using LSTM")
st.write("Enter last 24 hours traffic data")

input_data = []
for i in range(24):
    val = st.number_input(f"Hour {i+1}", min_value=0.0, step=1.0)
    input_data.append(val)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(-1, 1)
    input_scaled = scaler.transform(input_array)
    input_scaled = input_scaled.reshape(1, 24, 1)

    prediction = model.predict(input_scaled)
    predicted_value = scaler.inverse_transform(prediction)

    st.success(f"Predicted Traffic: {int(predicted_value[0][0])}")