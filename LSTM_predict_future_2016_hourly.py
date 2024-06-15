import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 导入数据
data_path = "C:/Users/bolee/Documents/GitHub/NTU_DA_finalProject/preprocessed_PV_ground_2016.csv"
data = pd.read_csv(data_path)

# 删除有缺失值的行
data_cleaned = data.dropna()

# 打印資料的基本資訊
print("資料樣本數 (行數):", data_cleaned.shape[0])
print("資料特徵數 (列數):", data_cleaned.shape[1])
print("\n資料型態:")
print(data_cleaned.dtypes)
print("\n前幾行資料:")
print(data_cleaned.head())

# 將TIMESTAMP欄位轉換為日期時間類型
data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])

# 按照每小時對資料進行分組並計算每小時的平均值
data_hourly_avg = data.resample('H', on='TIMESTAMP').mean()

# 印出處理後的資料
print("資料樣本數 (行數):", data_hourly_avg.shape[0])
print("資料特徵數 (列數):", data_hourly_avg.shape[1])
print("\n處理後的資料:")
print(data_hourly_avg.head())

# 删除重新采样后的 NaN 行
data_hourly_avg = data_hourly_avg.dropna()

# 提取特徵和目標列
features = data_hourly_avg.columns[data_hourly_avg.columns != 'PwrMtrRealPwrAC']
target = 'PwrMtrRealPwrAC'

# 定義函數，用於生成序列資料
def generate_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix, :-1], data[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# 將資料轉換為numpy array
data_values = data_hourly_avg.values

# 定義參數範圍
n_steps_range = [3, 5, 7, 10]
epochs_range = [50, 100, 200]
batch_size_range = [16, 32, 64]

# 記錄最佳參數和最低誤差
best_n_steps = None
best_epochs = None
best_batch_size = None
lowest_mae = float('inf')

for n_steps in n_steps_range:
    for epochs in epochs_range:
        for batch_size in batch_size_range:
            # 生成序列資料
            X, y = generate_sequences(data_values, n_steps)

            # 分割資料集為訓練集和測試集
            split = int(0.8 * len(X))
            X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

            # 構建LSTM模型
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(n_steps, X.shape[2])))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')

            # 訓練模型
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

            # 使用模型進行預測
            y_pred = model.predict(X_test).reshape(-1)

            # 計算均值絕對誤差
            mae = mean_absolute_error(y_test, y_pred)

            # 檢查是否是最低誤差
            if mae < lowest_mae:
                lowest_mae = mae
                best_n_steps = n_steps
                best_epochs = epochs
                best_batch_size = batch_size

print(f"最佳參數: n_steps={best_n_steps}, epochs={best_epochs}, batch_size={best_batch_size}")
print(f"最低均值絕對誤差: {lowest_mae}")

# 使用最佳參數重新訓練模型並繪製結果
n_steps = best_n_steps
epochs = best_epochs
batch_size = best_batch_size

# 生成序列資料
X, y = generate_sequences(data_values, n_steps)

# 分割資料集為訓練集和測試集
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# 構建LSTM模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 訓練模型
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# 使用模型進行預測
y_pred = model.predict(X_test).reshape(-1)

# 直接使用未標準化的數據
y_pred_inv = y_pred
y_test_inv = y_test

# 繪製模型預測值和實際值的比較圖
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.legend()
plt.show()
