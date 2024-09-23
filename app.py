import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

# Define the time period for which we want to fetch the stock data
start = '2009-12-31'
end = '2020-01-01'

# Streamlit app title
st.title('Stock Trend Prediction')

# Text input for the stock ticker symbol, default is 'AAPL'
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Fetch historical stock data from Yahoo Finance for the given time period
df = yf.download(user_input, start=start, end=end)

# Display a summary of the stock data (mean, std, min, etc.)
st.subheader(f'Data from {start} to {end}')
st.write(df.describe())

# Visualize the Closing Price over time
st.subheader('Closing Price vs Time Chart')

# Create a plot for the stock closing price
plt.figure(figsize=(12, 6))  # Set the figure size
plt.plot(df['Close'], label='Closing Price')  # Plot the 'Close' column

# Add labels, title, and legend to the chart
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{user_input} Closing Price')
plt.legend()

# Display the plot in the Streamlit app
st.pyplot(plt)

# Visualize the Closing Price along with 100-day Moving Average (MA100)
st.subheader('Closing Price vs Time Chart with 100MA')

# Calculate the 100-day Moving Average
ma100 = df.Close.rolling(100).mean()

# Create a new plot with the closing price and 100-day Moving Average
plt.figure(figsize=(12, 6))  # Set the figure size
plt.plot(df['Close'], label='Closing Price')  # Plot the 'Close' column

# Add labels, title, and legend to the chart
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{user_input} Closing Price')
plt.legend()

# Plot the 100-day Moving Average (MA100)
plt.plot(ma100, label='100-day MA', color='orange')

# Display the plot in the Streamlit app
st.pyplot(plt)

# Visualize the Closing Price along with both 100-day (MA100) and 200-day Moving Averages (MA200)
st.subheader('Closing Price vs Time Chart with 100MA and 200MA')

# Calculate the 200-day Moving Average
ma200 = df.Close.rolling(200).mean()

# Create a new plot with the closing price, 100MA, and 200MA
plt.figure(figsize=(12, 6))  # Set the figure size
plt.plot(df['Close'], label='Closing Price')  # Plot the 'Close' column

# Add labels, title, and legend to the chart
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{user_input} Closing Price')
plt.legend()

# Plot the 100-day (orange) and 200-day (green) Moving Averages
plt.plot(ma100, label='100-day MA', color='orange')
plt.plot(ma200, label='200-day MA', color='green')

# Display the plot in the Streamlit app
st.pyplot(plt)

# Splitting the data into Training and Testing sets
# We use 70% of the data for training and the remaining 30% for testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

# Normalize the training data using MinMaxScaler (scales values between 0 and 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler on the training data and transform it
data_training_array = scaler.fit_transform(data_training)

# Load the pre-trained model (trained on similar stock data)
model = load_model('keras_model.h5')

# Testing Phase
# Fetch the last 100 days of the training data to use in testing
past_100_days = data_training.tail(100) 

# Combine the past 100 days of training data with the testing data
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

# Apply the same scaling (fit-transform) to the combined data
input_data = scaler.fit_transform(final_df)

# Prepare the test dataset with sliding windows of 100 previous days' data
x_test = []
y_test = []

# Create testing data where x_test contains sequences of 100 days and y_test contains the actual stock prices
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])  # Append the sequence of 100 values
    y_test.append(input_data[i, 0])     # Append the corresponding next value (target)

# Convert the test data into NumPy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Use the model to make predictions on the test data
y_predicted = model.predict(x_test)

# Fetch the scale factor to inverse the scaling applied during normalization
scaler = scaler.scale_

# Scale factor for inverse transformation (scaling the predicted and actual values back to the original range)
scale_factor = 1 / scaler[0]

# Multiply by scale factor to convert the predicted values back to the original scale
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Plot: Compare Predicted vs Actual Stock Prices
st.subheader('Predictions vs Original')

# Create a plot to compare the predicted prices vs the actual prices
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')  # Plot the actual stock prices
plt.plot(y_predicted, 'r', label='Predicted Price')  # Plot the predicted stock prices

# Add labels, legend, and show the plot in Streamlit
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
