import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
layers = tf.keras.layers
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO

# Add custom CSS for background and fonts
st.markdown(
    """
    <style>
    .main { background-color: #f4f4f9; font-family: 'Roboto', sans-serif; }
    .stButton>button { background-color: #ff4b4b; color: white; border-radius: 10px; padding: 10px; }
    .stSlider>.css-14xtw13 { background-color: #007bff; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Load data function
@st.cache_resource
def load_data():
    dfs = pd.read_csv(r'D:\Dominos\Pizza_Sale_Cleaned.csv')
    dfs['order_date'] = pd.to_datetime(dfs['order_date'])
    df_agg = dfs.groupby(['order_date', 'pizza_name_id']).agg({'quantity': 'sum'}).reset_index()
    df_agg['order_date'] = pd.to_datetime(df_agg['order_date'], format='%Y-%m-%d')
    data = pd.pivot_table(data=df_agg, values='quantity', index='order_date', columns='pizza_name_id')
    data = data.asfreq('1D').fillna(0).sort_index()
    return data

# Load LSTM model function
@st.cache_resource
def load_lstm_model():
    return models.load_model(r'C:\Users\VRISHALI\lstm_pizza_model.keras')

# Initialize Streamlit app
st.title("Domino's Pizza Demand Forecast and Ingredient Planner 🍕")
st.write("Predict daily pizza sales and calculate ingredient requirements.")

# Load data and model
data = load_data()
model = load_lstm_model()

# Forecast function
def predict_next_n_days(model, data, sequence_length, scaler, n):
    predictions = []
    recent_data = data[-sequence_length:]

    for _ in range(n):
        input_sequence = np.expand_dims(recent_data, axis=0)
        pred_scaled = model.predict(input_sequence)
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, pred_scaled.shape[-1]))
        predictions.append(pred[-1])
        recent_data = np.append(recent_data[1:], np.expand_dims(pred_scaled[0], axis=0), axis=0)

    return np.array(predictions)

# Input for days to forecast
days_to_forecast = st.slider('Select number of days to forecast', 1, 365, 7)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Predict button
if st.button('Predict 🚀'):
    if days_to_forecast > 0:
        test_predictions = predict_next_n_days(model, scaled_data, 7, scaler, days_to_forecast)
        rounded_predictions = np.round(test_predictions).astype(int)
        st.success("Data predicted successfully")
