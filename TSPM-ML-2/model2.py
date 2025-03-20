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
import joblib
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.use('Qt5Agg')
# 删除NaN行
# ['France', 'Germany', 'North America', 'South Korea']
# 加载数据
data = pd.read_csv("../TSPM-ML-0/prediction_results.csv")
data = data.loc[data['Entity'].isin(['France', 'Germany', 'North America', 'South Korea'])]
data = data.dropna()
# 删除NaN行
print(data.head())

# 创建滞后特征：前1-7天的感染规模
for i in range(1, 15):
    data[f'real_{i}'] = data.iloc[:,3].shift(i)

# 设置目标变量为real_1（未来一天的实际感染规模）
data['target'] = data.iloc[:,3].shift(-1)
# 特征矩阵包含当天和前7天的感染规模
feature_cols = ['estimated_actual_infections'] + [f'real_{i}' for i in range(1, 8)]
data = data.dropna()
X = data[feature_cols]
y = data['target']

# 将数据集划分为训练集和测试集,注意：不能随机划分，会打乱时间顺序
test_size = 60
print(X.shape)
print(y.shape)
X_train = X[:-test_size]
X_test = X[-test_size:]
y_train = y[:-test_size]
y_test = y[-test_size:]


# 数据预处理
# 对特征进行标准化
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
# 保存缩放器
joblib.dump(scaler_X, 'scaler_X.pkl')
# 对目标变量进行标准化
scaler_y = MinMaxScaler(feature_range=(0, 1))
joblib.dump(scaler_y, 'scaler_y.pkl')
# 将 y_train 和 y_test 转换为 NumPy 数组并重塑为二维数组
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
# 重塑数据为 GRU 格式 [samples(样本数量), time_steps(时间步), features(每个时间步的特征数量)]
# 我们将滞后特征视为时间步
n_features = X_train.shape[1]
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 8, 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 8, 1)

print("X_train_scaled.shape is --------------------------->",X_train_scaled.shape)
print("X_train_scaled.shape is --------------------------->",X_test_scaled.shape)


# 构建GRU模型
def build_gru_model(gru_units=128, dropout_rate=0.4, lr=0.001):
    model = Sequential()

    # GRU层，注意input_shape改为(time_steps, n_features)
    model.add(GRU(units=gru_units,
                  input_shape=(8, 1),  # 8个时间步，每步1个特征
                  activation='tanh',  # or__'relu'
                  return_sequences=False))  # 表示 GRU 只返回最后一个时间步的输出 h_T，而不是每个时间步的 h_t。h_T是各时刻的隐藏状态，与输入X_t共同决定要记忆前面的的多少信息

    # 其他层保持不变
    model.add(Dropout(dropout_rate))  # dropout_rate=0.2：防止过拟合，训练时随机丢弃 20% 的神经元。
    model.add(Dense(1))

    optimizer = Adam(learning_rate=lr, clipnorm=1.0)  # 添加梯度裁剪
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


# 创建模型实例
gru_model = build_gru_model(gru_units=128, dropout_rate=0.2, lr=0.001)
print(gru_model.summary())

# 设置回调函数
early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min',
                           restore_best_weights=True)  # 如果 val_loss 连续 15 轮不下降，停止训练，并恢复最佳权重，避免过拟合。
checkpoint = ModelCheckpoint('best_gru_covid_model.keras', monitor='val_loss', save_best_only=True,
                             verbose=1)  # 只保存验证损失最小的模型权重。
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1,
                              min_lr=0.0001)  # 5 轮内 val_loss 没有改善，学习率减半，最小 0.0001

# 训练模型
history = gru_model.fit(
    X_train_reshaped,
    y_train_scaled,
    epochs=1000,
    batch_size=16,
    validation_split=0.2,  # 分出20%作为验证集
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# 评估模型
# 载入最佳模型
gru_model.load_weights('best_gru_covid_model.keras')

# 预测
train_predict = gru_model.predict(X_train_reshaped)
test_predict = gru_model.predict(X_test_reshaped)

# 反归一化
train_predict = scaler_y.inverse_transform(train_predict)
test_predict = scaler_y.inverse_transform(test_predict)
y_train_actual = scaler_y.inverse_transform(y_train_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

# 计算评估指标
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
train_mae = mean_absolute_error(y_train_actual, train_predict)
test_mae = mean_absolute_error(y_test_actual, test_predict)
train_r2 = r2_score(y_train_actual, train_predict)
test_r2 = r2_score(y_test_actual, test_predict)
MAPE = (np.mean(np.abs((y_test_actual - test_predict) / y_test_actual))) * 100
print(f"MAPE: {MAPE}")
print('训练集 RMSE: %.4f' % train_rmse)
print('测试集 RMSE: %.4f' % test_rmse)
print('训练集 MAE: %.4f' % train_mae)
print('测试集 MAE: %.4f' % test_mae)
print('训练集 R²: %.4f' % train_r2)
print('测试集 R²: %.4f' % test_r2)

# 可视化训练历史
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='训练MAE')
plt.plot(history.history['val_mae'], label='验证MAE')
plt.title('平均绝对误差')
plt.xlabel('轮次')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.show()

# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(y_train_actual, label='实际值')
plt.plot(train_predict, label='预测值')
plt.title('训练集: 实际感染规模 vs 预测感染规模')
plt.xlabel('时间点')
plt.ylabel('感染规模')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(y_test_actual, label='实际值')
plt.plot(test_predict, label='预测值')
plt.title('测试集: 实际感染规模 vs 预测感染规模')
plt.xlabel('时间点')
plt.ylabel('感染规模')
plt.legend()
plt.tight_layout()
plt.show()


# 超参数调优函数
def tune_hyperparameters():
    # 定义超参数网格
    gru_units_list = [32, 64, 128]  # GRU单元数量
    dropout_rate_list = [0.2, 0.3, 0.4]  # Dropout率
    learning_rate_list = [0.01, 0.001, 0.0001]  # 学习率
    batch_size_list = [16, 32, 64]  # Batch大小

    # 存储结果
    results = []
    best_val_loss = float('inf')
    best_params = {}

    # 迭代所有组合
    for gru_units in gru_units_list:
        for dropout_rate in dropout_rate_list:
            for lr in learning_rate_list:
                for batch_size in batch_size_list:
                    print(f"\n测试参数: units={gru_units}, dropout={dropout_rate}, lr={lr}, batch_size={batch_size}")

                    # 构建模型
                    model = build_gru_model(gru_units=gru_units, dropout_rate=dropout_rate, lr=lr)

                    # 提前停止回调
                    early_stop_tune = EarlyStopping(monitor='val_loss', patience=10, verbose=0,
                                                    restore_best_weights=True)

                    # 训练模型
                    hist = model.fit(
                        X_train_reshaped,
                        y_train_scaled,
                        epochs=50,
                        batch_size=batch_size,
                        validation_split=0.2,
                        callbacks=[early_stop_tune],
                        verbose=0
                    )

                    # 记录最佳验证损失
                    min_val_loss = min(hist.history['val_loss'])
                    results.append({
                        'gru_units': gru_units,
                        'dropout_rate': dropout_rate,
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'val_loss': min_val_loss
                    })

                    print(f"验证损失: {min_val_loss:.6f}")

                    # 更新最佳参数
                    if min_val_loss < best_val_loss:
                        best_val_loss = min_val_loss
                        best_params = {
                            'gru_units': gru_units,
                            'dropout_rate': dropout_rate,
                            'learning_rate': lr,
                            'batch_size': batch_size
                        }
                        print(f"【新最佳参数】: {best_params}")

    # 排序结果
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_loss')

    print("\n超参数调优结果 (前5名):")
    print(results_df.head())

    print("\n最佳参数组合:")
    print(best_params)

    return best_params


# # 取消下面注释运行超参数调优
# best_params = tune_hyperparameters()
# #
# # 使用最佳参数训练最终模型
# best_model = build_gru_model(
#     gru_units=best_params['gru_units'],
#     dropout_rate=best_params['dropout_rate'],
#     lr=best_params['learning_rate']
# )
# best_model.fit(
#     X_train_reshaped,
#     y_train_scaled,
#     epochs=100,
#     batch_size=best_params['batch_size'],
#     validation_split=0.2,
#     callbacks=[early_stop, checkpoint, reduce_lr],
#     verbose=1
# )
#
# # 模型保存与加载
# best_model.save('covid_gru_final.keras')
# # from tensorflow.keras.models import load_model
# # loaded_model = load_model('covid_gru_final.h5')
