# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skopt import BayesSearchCV
import joblib
# from owid import catalog
model = "./random_forest_regressor.pkl"
data = pd.read_csv(r'D:\浏览器\covid-19-cases-tests-positive-rate-and-reproduction-rate.csv')
data = data.loc[data['Entity'].isin(['France', 'Germany', 'North America', 'South Korea'])]
data = data.dropna()
data_infections = pd.read_csv(r"F:\pycharm_COVID-19\COVID-19\prediction_results.csv")

data_infections = data_infections.dropna()
print(data.head())
for i in range(1, 8):
    data_infections['estimated_actual_infections{i}'] = data_infections['estimated_actual_infections'].shift(i)
# 当天病例数
volume_1=data['Daily COVID-19 tests per 1,000 people (7-day smoothed)'].shift(-1)
# 未来1天感染规模
real_1 =data_infections['estimated_actual_infections'].shift(-1)
# 当天感染规模
real=data_infections['estimated_actual_infections']
# 与其他列对齐
real = real[:-1].reset_index(drop=True)
# 抛弃原来的索引，否则在创建feature_X时，由于索引不同会造成自动用NaN补充样本量，导致与y不匹配
volume_1 = volume_1.reset_index(drop=True)
real_1 = real_1.reset_index(drop=True)
real = real.reset_index(drop=True)

# 删除包含 NaN 的行，防止由于shift(-1)将数据下移造成的NaN
volume_1 = volume_1.dropna()
real_1 = real_1.dropna()
real = real.dropna()

features_df = pd.DataFrame({
    "volume_1": volume_1[-1],
    "real_1": 0.05,
    "real": real[-1],
})
X = features_df

# 重塑数据为 GRU 格式 [samples(样本数量), time_steps(时间步), features(每个时间步的特征数量)]
# 我们将滞后特征视为时间步
n_features = X.shape[1]
X_reshaped = X.reshape(X.shape[0], 8, 1)
predicted_scaled = model.predict(X_reshaped)

print(f"预测的下一天的病例数: {predicted_scaled[0][0]:.2f}")