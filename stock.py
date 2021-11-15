#%% Import Library
import investpy
import datetime as dt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
from tensorflow import keras

#%% Load data
# start = '01/06/2009'
# end = dt.datetime.now().strftime("%d/%m/%Y")
company = 'FB'
# Load data frame from investpy
# df = investpy.get_stock_historical_data(stock=company, country='United States', from_date=start, to_date=end)
# df = pd.DataFrame(df)
# print(df)

# Read data from CSV file
df = pd.read_csv(f'{company}.csv')

# Add column H-L and O-C
df['H-L'] = df['High'] - df['Low']
df['O-C'] = df['Open'] - df['Close']

# Moving Average
ma_1 = 7
ma_2 = 14
ma_3 = 21

df[f'SMA_{ma_1}'] = df['Close'].rolling(window=ma_1).mean()
df[f'SMA_{ma_2}'] = df['Close'].rolling(window=ma_2).mean()
df[f'SMA_{ma_3}'] = df['Close'].rolling(window=ma_3).mean()

# Standard Deviation
df[f'SD_{ma_1}'] = df['Close'].rolling(window=ma_1).std()
df[f'SD_{ma_3}'] = df['Close'].rolling(window=ma_3).std()
df.dropna(inplace=True)

df.to_csv(f"{company}.csv")
print("Done Loading Data")

#%% Process Data
pre_day = 30
# Scale data
scala_x = MinMaxScaler(feature_range=(0, 1))
scala_y = MinMaxScaler(feature_range=(0, 1))

# Column needed
cols_x = ['H-L', 'O-C', f'SMA_{ma_1}', f'SMA_{ma_2}', f'SMA_{ma_3}', f'SD_{ma_1}', f'SD_{ma_3}']
cols_y = ['Close']

# Reshape
scaled_data_x = scala_x.fit_transform(df[cols_x].values.reshape(-1, len(cols_x)))
scaled_data_y = scala_y.fit_transform(df[cols_y].values.reshape(-1, len(cols_y)))

# Create empty list
x_total = []
y_total = []

# Add data to list
for i in range(pre_day, len(df)):
    x_total.append(scaled_data_x[i-pre_day:i])
    y_total.append(scaled_data_y[i])

# The last rows that be used to predict
test_size = 60

# Slice train data and test data
x_train = np.array(x_total[:len(x_total)-test_size])
x_test = np.array(x_total[len(x_total)-test_size:])
y_train = np.array(y_total[:len(y_total)-test_size])
y_test = np.array(y_total[len(y_total)-test_size:])

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#%% Build Model
# Use Sequential model
model = Sequential()

new_training = True
if new_training:
    # Use Long Short-Term Memory layer
    model.add(LSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60))
    model.add(Dropout(0.2))
    model.add(Dense(units=len(cols_y)))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=20, steps_per_epoch=40, use_multiprocessing=True)
    model.save(f"{company}.h5")    
else:
    model = keras.models.load_model(f"{company}.h5")

print("Done Training Model")

#%% Testing
# Predict test data
predict_prices = model.predict(x_test)

# Inverse data
predict_prices = scala_y.inverse_transform(predict_prices)

print(predict_prices)

#%% Ploting the Stat
# Reshape y_test with new name real_price
real_price = df[len(df)-test_size:]['Close'].values.reshape(-1, 1)
real_price = np.array(real_price)
real_price = real_price.reshape(real_price.shape[0], 1)

plt.plot(real_price, color="red", label=f"Real {company} Prices")
plt.plot(predict_prices, color="blue", label=f"Predicted {company} Prices")
plt.title(f"{company} Prices")
plt.xlabel("Time")
plt.ylabel("Stock Prices")
plt.ylim(bottom=0)
plt.legend()
plt.show()

#%% Make Prediction
x_predict = df[len(df)-pre_day:][cols_x].values.reshape(-1, len(cols_x))
x_predict = scala_x.transform(x_predict)
x_predict = np.array(x_predict)
x_predict = x_predict.reshape(1, x_predict.shape[0], len(cols_x))
prediction = model.predict(x_predict)
prediction = scala_y.inverse_transform(prediction)
print(prediction)

#%%
# %%
