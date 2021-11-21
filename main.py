# In[1]: Khai báo các thư viện cần thiết
import investpy  # dùng dữ liệu từ trang investing.com
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler  # dùng để xử lý dữ liệu
import numpy as np  # dùng các hàm toán
from tensorflow import keras
from tensorflow.keras.models import Sequential  # model để huấn luyện dự đoán
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt  # dùng để thể hiện kết quả trên trục tọa độ (x, y)
import os

# In[2]: Load dữ liệu
start = '01/01/2001'  # ngày bắt đầu
end = dt.datetime.now().strftime("%d/%m/%Y")  # ngày kết thúc (hôm nay)
company = 'PLX'  # mã chứng khoán của công ty
country = 'Vietnam'  # tên quốc gia
# lấy dữ liệu từ trang investing.com
df = investpy.get_stock_historical_data(stock=company, country=country, from_date=start, to_date=end)

df = pd.DataFrame(df)  # chuyển dữ liệu thành dạng bản
print(df.to_string())

df.to_csv(f"datasets/{company}-{country}.csv")
print("Done Loading Data")

# In[3]: Xử lý dữ liệu
pre_day = 30
scala_x = MinMaxScaler(feature_range=(0, 1))  # khai báo biến đặt vùng giá trị [0,1]
scala_y = MinMaxScaler(feature_range=(0, 1))  # khai báo biến đặt vùng giá trị [0,1]
cols_x = ['Open', 'High', 'Low', 'Volume']
cols_y = ['Close']
# xử lý dữ liệu các cột thành các giá trị từ 0 đến 1
scaled_data_x = scala_x.fit_transform(df[cols_x].values.reshape(-1, len(cols_x)))
scaled_data_y = scala_y.fit_transform(df[cols_y].values.reshape(-1, len(cols_y)))

x_total = []  # mảng lưu dữ liệu của feature, mỗi sample là dữ liệu của 30 ngày trước đó
y_total = []  # mảng lưu dữ liệu của label cho mỗi sample, là giá dự đoán hôm nay dựa trên 30 ngày trước 

for i in range(pre_day, len(df)):
    x_total.append(scaled_data_x[i - pre_day:i])  # chèn sample là từng bộ dữ liệu 30 ngày trước ngày i
    y_total.append(scaled_data_y[i])

test_size = 365
# kích thước tập test
x_train = np.array(x_total[:len(x_total) - test_size])  # chia tập dữ liệu tập train
x_test = np.array(x_total[len(x_total) - test_size:])  # chia label tập train
y_train = np.array(y_total[:len(y_total) - test_size])  # chia tập dữ liệu tập test
y_test = np.array(y_total[len(y_total) - test_size:])  # chia label tập train

# xem kích thước tập test
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# In[4]: Build Model

newModel = True
if os.path.isfile(f"models/{company}-{country}.h5"):
    lastModified = dt.datetime.fromtimestamp(os.path.getmtime(f"models/{company}-{country}.h5"))
    if lastModified.strftime("%Y%m%d") == dt.datetime.now().strftime("%Y%m%d"):
        newModel = False

model = Sequential()
if newModel:
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
    model.fit(x_train, y_train, epochs=120, steps_per_epoch=40, use_multiprocessing=True)
    model.save(f"models/{company}-{country}.h5")
else:
    model = keras.models.load_model(f"models/{company}-{country}.h5")

print("Done Training Model")
# print(model.summary())

# score = model.evaluate(x_test, y_test, batch_size=128)
# print('Test loss:', score)
# print('Test accuracy:', accuracy)


# In[5]: Testing
# dữ đoán trên bộ dữ liệu test
predict_prices = model.predict(x_test)  
predict_prices = scala_y.inverse_transform(predict_prices)  # chuyển giá trị về miền giá trị thực 

# lấy dữ liệu giá thực tế
real_price = scala_y.inverse_transform(y_test) # chuyển giá trị về miền giá trị thực
real_price = np.array(real_price)

# Vẽ đồ thị lên trục 
plt.plot(real_price, color="red", label=f"Real {company} Prices")
plt.plot(predict_prices, color="blue", label=f"Predicted {company} Prices")
plt.title(f"{company} Prices")
plt.xlabel("Time")
plt.ylabel("Stock Prices")
plt.ylim(bottom=0)
plt.legend()
today = dt.datetime.now().strftime("%Y%m%d")
plt.savefig(f'figures/{company}-{country}-{today}.svg')
plt.savefig(f'figures/{company}-{country}-{today}.png')
plt.show()

# In[6]: Make Prediction
x_predict = df[len(df)-pre_day:][cols_x].values.reshape(-1, len(cols_x))
x_predict = scala_x.transform(x_predict)
x_predict = np.array(x_predict)
x_predict = x_predict.reshape(1, x_predict.shape[0], len(cols_x))
prediction = model.predict(x_predict)
prediction = scala_y.inverse_transform(prediction).reshape(1)[0]
print('-----------------------------------------------------')
print('Giá dự đoán hiện tại:', predict_prices[-1][0])
print('Giá thực tế hiện tại:', real_price[-1][0])
print('-----------------------------------------------------')
change = prediction - predict_prices[-1][0]
print('Giá thay đổi:', ('+' + change) if change > 0 else change)
print('Giá dự đoán tiếp theo:', prediction)
print('Giá thực tế tiếp theo:', real_price[-1][0] + change)