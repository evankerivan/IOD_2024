import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv('./capstone/week_data.csv', parse_dates=['datetime'], index_col='datetime')

week_data = load_data()

# Streamlit UI
st.title("Sydney Air Quality Forecast")
st.subheader("Weather")

# Display Weather Data
weather_columns = week_data[['TEMP_min', 'TEMP_max', 'RAIN_sum', 'forecast']].copy()
weather_columns.columns = ['Temp Min (C)', 'Temp Max (C)', 'Rain (mm)', 'Forecast']
st.write(weather_columns)

# Display Air Quality Data
st.subheader("Air Quality")

air_quality_columns = ['CO', 'OZONE', 'PM10', 'PM2.5', 'SO2', 'overall']

def color_air_quality(val):
    color = {'Good': 'green', 'Fair': 'yellow', 'Poor': 'orange', 'Very Poor': 'red'}.get(val, '')
    return f'background-color: {color}'

st.write(week_data[air_quality_columns].style.applymap(color_air_quality))

# Line Plots for Pollutants
pollutants = ['CO_mean', 'SO2_mean', 'OZONE_mean', 'PM10_mean', 'PM2.5_mean']
for pollutant in pollutants:
    st.subheader(f"Daily {pollutant.split('_')[0]} Concentration")
    fig, ax = plt.subplots()
    ax.plot(week_data.index, week_data[pollutant])
    ax.set_xlabel("Date")
    ax.set_ylabel("Concentration")
    ax.set_title(f"{pollutant.split('_')[0]} Levels")
    st.pyplot(fig)
