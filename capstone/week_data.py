import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('./capstone/week_data.csv', index_col=0)
    # Convert index to datetime format and normalize to remove time component
    data.index = pd.to_datetime(data.index, errors='coerce').normalize()
    return data

week_data = load_data()

# Check if the index is datetime after loading and converting
if not pd.api.types.is_datetime64_any_dtype(week_data.index):
    st.error("Index is not in datetime format. Please check the data format.")
else:
    # Round temperature columns to 0 decimal places
    week_data[['TEMP_min', 'TEMP_max']] = week_data[['TEMP_min', 'TEMP_max']].round(0)

    # Use the date index to fill the "Day" column with the day of the week
    week_data['Day'] = week_data.index.to_series().dt.day_name()

    # Streamlit UI
    st.title("Sydney Air Quality Forecast")
    st.subheader("Weather")

    # Display Weather Data with Day column
    weather_columns = week_data[['Day', 'TEMP_min', 'TEMP_max', 'RAIN_sum', 'forecast']].copy()
    weather_columns.columns = ['Day', 'Temp Min (C)', 'Temp Max (C)', 'Rain (mm)', 'Forecast']
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
        plt.xticks(rotation=45)
        st.pyplot(fig)
