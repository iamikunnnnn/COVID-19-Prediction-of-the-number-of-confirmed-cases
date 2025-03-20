import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import joblib
model = keras.models.load_model('./best_gru_covid_model.keras')


plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.use('Qt5Agg')
# 删除NaN行

# 加载数据
data = pd.read_csv("../TSPM-ML-0/prediction_results.csv")
data = data.loc[data['Entity'].isin(['Germany'])]
data = data.dropna()

# 创建滞后特征：前1-7天的感染规模
for i in range(1, 15):
    data[f'real_{i}'] = data.iloc[:,3].shift(i)

# 特征矩阵包含当天和前7天的感染规模
feature_cols = ['estimated_actual_infections'] + [f'real_{i}' for i in range(1, 8)]
data = data.dropna()

X = data[feature_cols].tail(1)  # 取最后一行作为最新输入
print(X)
# 对特征进行标准化
# 加载训练阶段保存的缩放器参数
scaler_X = joblib.load('scaler_X.pkl')

# 使用相同的缩放器参数对新数据进行标准化
X_scaled = scaler_X.transform(X)

# 重塑数据为 GRU 格式 [samples(样本数量), time_steps(时间步), features(每个时间步的特征数量)]
# 我们将滞后特征视为时间步
n_features = X.shape[1]
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 8, 1)
predicted_scaled = model.predict(X_reshaped)
# 反归一化预测结果
# 假设 'estimated_actual_infections' 是目标变量的列名，这里对其进行反归一化
scaler_y = joblib.load('scaler_y.pkl')  # 加载用于归一化目标变量的缩放器

# 对预测结果进行反归一化
predicted = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1))
predicted.save("prediction_results.csv")
print(f"预测的下一天感染规模: {predicted[0][0]:.2f}")