import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load the datasets
tesla_df = pd.read_csv('./datasets/Tesla.csv - Tesla.csv.csv')
hyundai_df = pd.read_csv('./datasets/005380.KS.csv')

# Explore the data
print("Tesla Stock Prices:")
print(tesla_df.head())
print("\nHyundai Stock Prices:")
print(hyundai_df.head())

# Visualize the stock prices
plt.figure(figsize=(12, 6))
plt.plot(tesla_df['Date'], tesla_df['Close'], label='Tesla')
plt.plot(hyundai_df['Date'], hyundai_df['Close'], label='Hyundai')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Tesla vs Hyundai Stock Prices')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Merge the datasets for comparison
merged_df = pd.merge(tesla_df, hyundai_df, on='Date', suffixes=('_tesla', '_hyundai'))

# Prepare the data for neural network training
data = merged_df[['Close_tesla', 'Close_hyundai']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Split the data into training and testing sets
#train_data = scaled_data[:-30]
#test_data = scaled_data[-30:]

train_data = scaled_data[:-31]  # Adjust the indexing by subtracting 31 instead of 30
test_data = scaled_data[-31:]


# Define the features and labels
X_train, y_train = train_data[:-1], train_data[1:]

# Reshape the input data for LSTM
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(1, 2)))
model.add(Dense(2))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, verbose=0)

# Visualize the training loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Make predictions with the model
X_test = test_data[:-1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
predicted_data = model.predict(X_test)
predicted_data = scaler.inverse_transform(predicted_data)

# Compare the datasets and predicted data
plt.figure(figsize=(12, 6))
plt.plot(merged_df['Date'][-30:], merged_df['Close_tesla'][-30:], label='Actual Tesla')
plt.plot(merged_df['Date'][-30:], merged_df['Close_hyundai'][-30:], label='Actual Hyundai')
plt.plot(merged_df['Date'][-30:], predicted_data[:, 0], label='Predicted Tesla')
plt.plot(merged_df['Date'][-30:], predicted_data[:, 1], label='Predicted Hyundai')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Comparison between Actual and Predicted Stock Prices')
plt.legend()
plt.xticks(rotation=45)
plt.show()