import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import pickle

try:
    # Attempt to load the file
    with open('prediksi_sales.sav', 'rb') as file:
        data_close = pickle.load(file)
    # Continue with the rest of your code that uses data_close
except FileNotFoundError:
    print("File 'prediksi_sales.sav' not found. Make sure the file exists in the correct directory.")
except Exception as e:
    print(f"An error occurred while loading the file: {e}")

df = pd.read_csv("monthly-car-sales.csv")

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['Month'], infer_datetime_format=True)

# Set 'date' column as the index
df.set_index('date', inplace=True)

# Fit the model
results = data_close.fit()

# Streamlit app title
st.title('Forecasting Car Sales')

# Slider to choose the number of years for prediction
date = st.slider("Tentukan tahun", 1, 30, step=1)

# Forecast the next 'date' years
forecast = results.get_forecast(steps=date)

# Extract the predicted values and confidence interval
predicted_values = forecast.predicted_mean
confidence_interval = forecast.conf_int()

# Display the predicted values
if st.button("Predict"):
    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(confidence_interval)
    with col2:
        # Create a plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot observed data
        ax.plot(df.index, df['Sales'], label='Observasi', color='blue')

        # Plot predicted values
        ax.plot(predicted_values.index, predicted_values, label='Prediksi', color='red')

        # Plot confidence interval
        ax.fill_between(confidence_interval.index, confidence_interval.iloc[:, 0], confidence_interval.iloc[:, 1], color='pink', alpha=0.3, label='Interval Kepercayaan')

        # Add labels and title
        ax.set_xlabel('Waktu')
        ax.set_ylabel('Nilai')
        ax.set_title('Prediksi SARIMAX')
        ax.legend()

        # Display the plot
        st.pyplot(fig)
