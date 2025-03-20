# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skopt import BayesSearchCV

# from owid import catalog

data = pd.read_csv(r'D:\浏览器\covid-19-cases-tests-positive-rate-and-reproduction-rate.csv')
data = data.loc[data['Entity'].isin(['France', 'Germany', 'North America', 'South Korea'])]
data = data.dropna()
data_infections = pd.read_csv("../TSPM-ML-0/prediction_results.csv")
infections=data_infections.iloc[:,2]
infections = infections.dropna()
print(data.head())
for i in range(1, 8):
    data[f'positive_rate_lag_{i}'] = data['COVID-19 positivity rate'].shift(i)
y = infections
X = data[['COVID-19 positivity rate'] + [f'positive_rate_lag_{i}' for i in range(1, 8)] + [
    'Daily COVID-19 tests per 1,000 people (7-day smoothed)']]

# 将数据集划分为训练集和测试集
# 使用最近的20%数据作为测试集
test_size = 60

X_train = X[:-test_size]
X_test = X[-test_size:]
y_train = y[:-test_size]
y_test = y[-test_size:]
if __name__ == '__main__':
    # rfr =RandomForestRegressor(n_estimators=25)
    # rfr_s = cross_val_score(rfr, X, y, cv=10)
    # plt.plot(range(1, 11), rfr_s, label="RandomForest")
    #
    # plt.legend()
    # plt.show()
    # 2. 特征缩放
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)  # 训练集标准化
    # X_test = scaler.transform(X_test)  # 测试集标准化
    # scaler_minmax = MinMaxScaler()
    # X_train_minmax = scaler_minmax.fit_transform(X_train)  # 训练集归一化
    # X_test_minmax = scaler_minmax.transform(X_test)  # 测试集归一化
    # 3. 随机森林回归模型
    rf = RandomForestRegressor(random_state=42)

    # 4. 超参数调优：使用 GridSearchCV 来选择最优的超参数
    # 随机搜索，快
    # param_distributions = {
    # 网格搜索，准确但慢
    # param_grid = {
    # 贝叶斯搜索
    param_space = {

        'n_estimators': [100, 200, 300],  # 树的数量
        'max_depth': [10, 20, 30,],  # 树的最大深度
        'min_samples_split': [2, 5, 10],  # 内部节点再划分所需的最小样本数
        'min_samples_leaf': [1, 2, 4],  # 叶子节点最小样本数
        'max_features': ['sqrt', 'log2',None],  # 每棵树的最大特征数
        'bootstrap': [True, False]  # 是否使用自助法
    }

    # 创建GridSearchCV/RandomizedSearchCV对象，使用3折交叉验证
    # 创建 RandomizedSearchCV 对象，使用3折交叉验证
    # 网格搜索
    # search = GridSearchCV(estimator=rf,
    # 贝叶斯搜索
    bayes_search = BayesSearchCV(estimator=rf,
                                 search_spaces=param_space,  # 贝叶斯搜索
                                 # param_grid=param_grid,  # 网格搜索
                                 # param_distributions=param_distributions,  # 随机搜索
                                 cv=5,
                                 n_jobs=-1,
                                 verbose=2,
                                 scoring='neg_mean_squared_error')

    # 训练模型
    bayes_search.fit(X_train, y_train)

    # 输出最优参数
    print(f"Best parameters: {bayes_search.best_params_}")

    # 使用最佳参数创建最终模型
    best_rf = bayes_search.best_estimator_

    # 5. 评估模型：在测试集上进行预测并评估
    y_pred = best_rf.predict(X_test)

    # 计算均方误差和R²

    # 计算真实值的平均值
    y_mean = y_test.mean()

    # 计算分子：预测值与真实值之间的绝对误差之和
    absolute_errors = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    MAPE = (np.mean(np.abs((y_test - y_pred) / y_test))) * 100
    print(f"MAPE: {MAPE}")
    # 计算分母：真实值与真实值平均值之间的绝对误差之和
    denominator = mean_absolute_error(y_test, [y_mean] * len(y_test), multioutput='raw_values')

    # 计算RAE
    rae = absolute_errors / denominator

    print(f"Relative Absolute Error (RAE): {rae}")
    r2 = r2_score(y_test, y_pred)

    print(f"R-squared: {r2}")

    # 获取特征重要性
    feature_importances = best_rf.feature_importances_

    # 归一化特征重要性
    normalized_importances = feature_importances / np.sum(feature_importances)
    pd_X_train = pd.DataFrame(X_train)
    # 打印归一化后的权重
    for feature_name, importance in zip(pd_X_train.columns, normalized_importances):
        print(f"Feature: {feature_name}, Normalized Importance: {importance:.4f}")
    # 6. 保存模型（如果需要）
    import joblib

    joblib.dump(best_rf, 'random_forest_model.pkl')

    # Best parameters: {'bootstrap': True, 'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
