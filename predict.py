import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load model
model = load_model("weather_model.h5", compile=False)

# Load data again
df = pd.read_csv("weather.csv")
df = df[['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 'precip_mm']]
df.columns = ['temp', 'humidity', 'pressure', 'wind_speed', 'rain']
df.ffill(inplace=True)

# Normalize (IMPORTANT: same as training)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# Take last 10 days
last_10_days = data_scaled[-10:]

# Reshape for model
input_data = np.reshape(last_10_days, (1, 10, 5))

# Predict
prediction = model.predict(input_data)

# Convert back to real value
predicted_temp = scaler.inverse_transform(
    [[prediction[0][0], 0, 0, 0, 0]]
)[0][0]

prediction = model.predict(input_data)[0]

# Get scaled values
temp_scaled = prediction[0]
rain_scaled = prediction[1]

# Create dummy row
dummy_row = [0, 0, 0, 0, 0]
dummy_row[0] = temp_scaled
dummy_row[4] = rain_scaled

# Convert back to real values
real_values = scaler.inverse_transform([dummy_row])[0]

predicted_temp = real_values[0]
predicted_rain = real_values[4]

print("🌡️ Predicted Temp:", predicted_temp)
print("🌧️ Predicted Rain:", predicted_rain)

# Ride logic (temp)
if predicted_temp > 35:
    print("🔥 Too hot for bike ride")
elif predicted_temp < 15:
    print("🥶 Too cold for ride")
else:
    print("✅ Perfect weather for ride")

# 🚨 REAL AI rain logic
if predicted_rain > 0.05:
    print("🌧️ Rain expected — avoid ride")
else:
    print("☀️ No rain — safe ride")