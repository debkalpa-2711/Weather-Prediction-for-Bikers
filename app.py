import streamlit as st
import requests

st.title("🌦️ AI Weather Predictor")

api_key = "ca8625ad4a78b28df948fdc89420fa75"

city = st.text_input("Enter City", "Kolkata,IN")

if st.button("Get Weather"):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        temp = data.get('main', {}).get('temp', 0)
        humidity = data.get('main', {}).get('humidity', 0)
        pressure = data.get('main', {}).get('pressure', 0)
        wind = data.get('wind', {}).get('speed', 0)
        rain = data.get('rain', {}).get('1h', 0)

        st.subheader("🌍 Current Weather")
        st.write(f"🌡️ Temp: {temp} °C")
        st.write(f"💧 Humidity: {humidity}%")
        st.write(f"ضغط Pressure: {pressure} mb")
        st.write(f"🌬️ Wind: {wind} m/s")
        st.write(f"🌧️ Rain: {rain} mm")
    else:
        st.error(f"❌ Error: {data}")

    import numpy as np
    from tensorflow.keras.models import load_model

    model = load_model("weather_model.h5", compile=False)

    # convert wind
    wind_kph = wind * 3.6


    # create input
    input_row = [temp, humidity, pressure, wind_kph, rain]

    # fake 10 days
    input_data = np.array([input_row] * 10)
    input_data = np.reshape(input_data, (1, 10, 5))

    # predict
    from tensorflow.keras.models import load_model

    model = load_model("weather_model.h5", compile=False)

    prediction = model.predict(input_data)[0]

    scaled_temp = prediction[0]
    scaled_rain = prediction[1]

    #st.write("Raw Output:", scaled_temp, scaled_rain)

    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd

    df = pd.read_csv("weather.csv")
    df = df[['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 'precip_mm']]
    df.columns = ['temp', 'humidity', 'pressure', 'wind_speed', 'rain']

    scaler = MinMaxScaler()
    scaler.fit(df)

    dummy = [scaled_temp, 0, 0, 0, 0]
    real = scaler.inverse_transform([dummy])[0]

    real_temp = real[0]

    st.subheader("🌍 Current Weather")
    st.write(f"🌡️ Current Temp: {temp} °C")

    st.subheader("🔮 AI Prediction (Next Day)")
    st.write(f"🌡️ Predicted Temp: {real_temp:.2f} °C")

    # Use REAL current temp (not AI)
    if temp > 35:
        st.warning("🔥 Too hot for ride")
    elif temp < 15:
        st.warning("🥶 Too cold for ride")
    else:
        st.success("✅ Good weather for ride")

#streamlit run app.py