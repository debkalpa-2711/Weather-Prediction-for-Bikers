import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("weather.csv")

# ✅ WRITE YOUR LINE HERE
df = df[['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 'precip_mm']]

# Rename columns
df.columns = ['temp', 'humidity', 'pressure', 'wind_speed', 'rain']

# Handle missing values
df.ffill(inplace=True)

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

print("✅ Data cleaned successfully!")
print(df.head())

import numpy as np


def create_sequences(data, seq_length=10):
    X = []
    y = []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length][0])  # Predict temp

    return np.array(X), np.array(y)


X, y = create_sequences(data_scaled)

print("X shape:", X.shape)
print("y shape:", y.shape)