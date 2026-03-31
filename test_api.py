import requests

api_key = "ca8625ad4a78b28df948fdc89420fa75"
city = "Kolkata,IN"

url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

response = requests.get(url)
data = response.json()

#print(response.status_code)
print(data)

# ✅ NOW extract values (after data exists)
temp = data.get('main', {}).get('temp', 0)
humidity = data.get('main', {}).get('humidity', 0)
pressure = data.get('main', {}).get('pressure', 0)
wind_speed = data.get('wind', {}).get('speed', 0)
rain = data.get('rain', {}).get('1h', 0)

wind_speed_kph = wind_speed * 3.6

input_row = [temp, humidity, pressure, wind_speed_kph, rain]
print("Model Input:", input_row)

print("Temp:", temp)
print("Humidity:", humidity)
print("Pressure:", pressure)
print("Wind:", wind_speed)
print("Rain:", rain)

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load model
model = load_model("weather_model.h5", compile=False)

# Load original dataset (for scaler)
df = pd.read_csv("weather.csv")
df = df[['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 'precip_mm']]
df.columns = ['temp', 'humidity', 'pressure', 'wind_speed', 'rain']
df.ffill(inplace=True)

# Scale using same scaler
scaler = MinMaxScaler()
scaler.fit(df)

# Create fake 10-day input (temporary)
last_10_days = np.array([input_row] * 10)

# Scale input
input_scaled = scaler.transform(last_10_days)
input_scaled = np.reshape(input_scaled, (1, 10, 5))

# Predict
prediction = model.predict(input_scaled)[0]

# Convert back to real values
dummy_row = [0, 0, 0, 0, 0]
dummy_row[0] = prediction[0]
dummy_row[4] = prediction[1]

real_values = scaler.inverse_transform([dummy_row])[0]

predicted_temp = real_values[0]
predicted_rain = real_values[4]

print("🌡️ Predicted Temp:", predicted_temp)
print("🌧️ Predicted Rain:", predicted_rain)

# Ride decision
if predicted_temp > 35:
    print("🔥 Too hot for bike ride")
elif predicted_temp < 15:
    print("🥶 Too cold for ride")
else:
    print("✅ Perfect weather for ride")

# Rain decision
if predicted_rain > 0.1:
    print("🌧️ Rain expected — avoid ride")
else:
    print("☀️ No rain — safe ride")