import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

# 設定隨機種子以確保結果可重現
np.random.seed(42)

# 生成隨機數據
X = np.random.rand(35, 5)  # 35筆資料，每筆資料有5個特徵
Y = np.random.rand(35, 1)  # 35筆資料，每筆資料有1個標籤

# 創建滑動窗口數據集
def create_dataset(X, Y, time_step=7):
    dataX, dataY = [], []
    for i in range(len(X) - time_step):
        a = X[i:(i + time_step), :]
        dataX.append(a)
        dataY.append(Y[i + time_step])
    return np.array(dataX), np.array(dataY)

time_step = 7
dataX, dataY = create_dataset(X, Y, time_step)

# 建立 LSTM 模型
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(time_step, 5)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 訓練模型
model.fit(dataX, dataY, epochs=100, batch_size=1, verbose=2)

# 進行預測
predictedY = model.predict(dataX)

# 繪製實際值和預測值
plt.figure(figsize=(12, 6))
plt.plot(range(len(dataY)), dataY, label='Actual Y')
plt.plot(range(len(predictedY)), predictedY, label='Predicted Y', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Y Value')
plt.title('Actual vs Predicted Y')
plt.legend()
plt.show()
