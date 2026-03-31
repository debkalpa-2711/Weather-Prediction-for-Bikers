import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess again (simple reuse)
df = pd.read_csv("weather.csv")
df = df[['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 'precip_mm']]
df.columns = ['temp', 'humidity', 'pressure', 'wind_speed', 'rain']
df.ffill(inplace=True)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# Create sequences
def create_sequences(data, seq_length=10):
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])

        # 👇 ONLY temp (0) and rain (4)
        y.append([data[i + seq_length][0], data[i + seq_length][4]])

    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled)

# Build model
model = Sequential([
    LSTM(64, input_shape=(10, 5)),
    Dense(32, activation='relu'),
    Dense(2)
])

model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X, y, epochs=10)

# Save model
model.save("weather_model.h5")

print("✅ Model trained successfully!")