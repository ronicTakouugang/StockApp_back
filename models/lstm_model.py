import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.scraper import get_stock_data
from datetime import datetime, timedelta

def get_next_prediction_date():
    # Get the next date for the prediction
    return (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

def predict_lstm():
    # Get the stock data
    data = get_stock_data("AAPL")
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    
    # Reshape data for LSTM input
    X_train = []
    for i in range(5, len(data)):
        X_train.append(data[i-5:i, 0])
    
    X_train = np.array(X_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    # Train the model
    model.fit(X_train, data[5:], epochs=50, batch_size=32)

    # Prepare input for prediction
    last_5_days = data[-5:].reshape(1, 5, 1)  # Reshape for prediction
    prediction = model.predict(last_5_days)

    # Inverse transform the prediction to get the actual price
    predicted_price = scaler.inverse_transform(prediction)

    # Get the next date for the prediction
    prediction_date = get_next_prediction_date()

    return {
        "prediction": predicted_price.flatten().tolist(),
        "date": prediction_date
    }
