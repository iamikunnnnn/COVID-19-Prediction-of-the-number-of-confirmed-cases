import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib

# 设置字体为支持中文的字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 或者使用 'Arial Unicode MS'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
matplotlib.use('TkAgg')  # 或者使用 'Qt5Agg'

def preprocess_data(data, smoothing_window=7):
    """预处理输入数据，处理缺失值和异常值"""
    # 解析数据并进行适当的错误处理
    dates = pd.to_datetime(data.iloc[:, 1])
    confirmed_cases = data.iloc[:, 2].values
    tests_per_thousand = data.iloc[:, 3].values

    # 处理阳性率可能的缺失值
    positivity_rate = data.iloc[:, 4]

    # 平滑Rt值并处理异常值
    raw_Rt = data.iloc[:, 5].values
    Rt = pd.Series(raw_Rt).clip(0.1, 10).rolling(window=smoothing_window, min_periods=1, center=True).mean().values

    return dates, confirmed_cases, tests_per_thousand, positivity_rate, Rt

def estimate_actual_infections(confirmed_cases, positivity_rate, total_population, tests_per_thousand):
    """基于检测数据估算实际感染人数，使用动态修正因子"""
    # 计算总检测量并进行适当处理
    total_tests = tests_per_thousand * total_population / 1000

    # 基于阳性率的动态修正因子（阳性率越高，低报率越高）
    f_base = 1.2
    f_dynamic = f_base + (positivity_rate * 2)  # 随阳性率增加从1.2到3.2

    # 使用改进公式估算感染人数
    estimated_infections = confirmed_cases / np.maximum(positivity_rate, 0.001) * f_dynamic  # np.maximum(positivity_rate, 0.001)确保不会除以一个非常小的数或零


    # 根据检测能力应用现实约束
    max_possible_cases = total_tests * 0.8  # 假设最多80%的检测可能呈阳性
    estimated_infections = np.minimum(estimated_infections, max_possible_cases)

    # 确保没有负值并平滑估算结果
    estimated_infections = pd.Series(np.maximum(estimated_infections, confirmed_cases)).rolling(
        window=7, min_periods=1, center=True).mean().values

    return estimated_infections


# ------------------------------
# SEIR模型实现
# ------------------------------

def seir_model(y, t, beta_t, sigma, gamma, N):
    """具有时变传染率的SEIR微分方程"""
    S, E, I, R = y

    # 获取当前时间的beta值并进行边界检查
    if t < len(beta_t):
        beta = beta_t[int(t)]
    else:
        beta = beta_t[-1]  # 使用最后可用的beta值

    # 改进的模型方程，确保人口守恒
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

    return [dSdt, dEdt, dIdt, dRdt]


def run_seir_model(total_population, estimated_infections, Rt, time_range):
    """运行SEIR模型并进行适当的初始化"""
    # 使用现实范围的SEIR参数
    sigma = 1 / 3.5  # 潜伏期：近期变种为3-4天
    gamma = 1 / 5  # 康复期：5天

    # 使用改进公式从Rt计算beta
    beta_t = Rt * gamma  # beta = R * gamma

    # 改进的初始条件
    E0 = estimated_infections[0] * 0.8  # 假设80%的估计病例仍处于暴露状态
    I0 = estimated_infections[0]
    R0 = estimated_infections[0] * 2  # 初始康复人口
    S0 = total_population - E0 - I0 - R0

    # 初始状态向量
    y0 = [S0, E0, I0, R0]

    # 求解ODE并改进错误处理

    try:
        sol = odeint(seir_model, y0, time_range, args=(beta_t, sigma, gamma, total_population))
        S, E, I_pred, R = sol.T
        return S, E, I_pred, R
    except Exception as e:
        print(f"ODE求解器错误: {e}")
        # 返回合理的后备值
        return np.ones(len(time_range)) * S0, np.ones(len(time_range)) * E0, \
               np.ones(len(time_range)) * I0, np.ones(len(time_range)) * R0

def build_enhanced_features(data, confirmed_cases, I_pred, estimated_infections, Rt):
    """为ML模型构建增强特征集"""
    features_df = pd.DataFrame({
        # 原始特征
        "confirmed_cases": confirmed_cases,  # 确诊病例数
        "estimated_infections": estimated_infections,  # 估计的感染人数
        "I_pred": I_pred,  # SEIR模型预测的感染人数
        "Rt": Rt,  # 有效传染数

        # 增强特征 - 移动平均
        "confirmed_7d_avg": pd.Series(confirmed_cases).rolling(window=7, min_periods=1).mean().values,  # 7天确诊平均值
        "confirmed_growth": pd.Series(confirmed_cases).pct_change(7).fillna(0).values,  # 7天确诊增长率

        # 比率特征
        "seir_to_confirmed_ratio": I_pred / np.maximum(confirmed_cases, 1),  # SEIR预测与确诊比
        "estimated_to_confirmed_ratio": estimated_infections / np.maximum(confirmed_cases, 1),  # 估计感染与确诊比

        # 时间特征
        "day_of_week": pd.to_datetime(data.iloc[:, 1]).dt.dayofweek,  # 周几（捕捉周期性报告模式）
    })

    # 添加滞后特征
    for lag in [3, 7, 14]:
        features_df[f"confirmed_lag_{lag}"] = pd.Series(confirmed_cases).shift(lag).fillna(0).values  # 确诊病例滞后值
        features_df[f"Rt_lag_{lag}"] = pd.Series(Rt).shift(lag).fillna(Rt[0]).values  # Rt滞后值

    return features_df


def train_ml_model(features_df, target_values, test_size=0.2):
    """训练ML模型，进行适当的验证和特征重要性分析"""
    # 定义要使用的特征列
    feature_columns = features_df.columns.tolist()

    # 分割数据 - 对于预测问题使用基于时间的分割
    train_size = int(len(features_df) * (1 - test_size))
    X_train, X_test = features_df.iloc[:train_size], features_df.iloc[train_size:]
    y_train, y_test = target_values[:train_size], target_values[train_size:]

    # 缩放特征以提高收敛性
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 使用适当超参数训练XGBoost
    model = XGBRegressor(
        objective="reg:squarederror",  # 回归目标函数
        n_estimators=300,  # 树的数量
        max_depth=10,  # 树的最大深度
        learning_rate=0.001,  # 学习率
        subsample=0.8,  # 样本子采样
        colsample_bytree=0.8,  # 特征子采样
        min_child_weight=5,  # 最小子节点权重
        random_state=42  # 随机种子
    )

    # 拟合模型
    model.fit(X_train_scaled, y_train)

    # 评估模型
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    print(f"训练集RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train))}")
    print(f"测试集RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test))}")
    print(f"R²分数(测试集): {r2_score(y_test, y_pred_test)}")

    # 打印特征重要性
    importance = model.feature_importances_
    for i, col in enumerate(feature_columns):
        print(f"{col}: {importance[i]:.4f}")

    # 返回模型和缩放器以供后续使用
    return model, scaler


def predict_infections(model, scaler, features_df):
    """使用训练好的模型进行预测"""
    X_scaled = scaler.transform(features_df)
    predictions = model.predict(X_scaled)

    # 确保预测结果合理
    predictions = np.maximum(predictions, 0)  # 没有负感染数

    return predictions

def run_hybrid_model(data, total_population=1_000_000):
    """运行完整的SEIR+ML混合模型"""
    # 预处理数据
    dates, confirmed_cases, tests_per_thousand, positivity_rate, Rt = preprocess_data(data)

    # 估计实际感染人数
    estimated_infections = estimate_actual_infections(
        confirmed_cases, positivity_rate, total_population, tests_per_thousand
    )

    # 运行SEIR模型
    time_range = np.arange(len(dates))
    S, E, I_pred, R = run_seir_model(total_population, estimated_infections, Rt, time_range)

    # 创建结合SEIR和检测数据的初步调整估计
    I_adjusted_simple = np.maximum(I_pred, estimated_infections)
    plt.figure(figsize=(7,7))
    plt.plot(time_range, I_adjusted_simple)
    plt.show()
    # 构建增强特征集
    # features_df = build_enhanced_features(data, confirmed_cases, I_pred, estimated_infections, Rt)
    features_df = pd.DataFrame({
        "confirmed_cases": confirmed_cases,
        "estimated_infections": estimated_infections,
        "I_pred": I_pred,
        "Rt": Rt
    })

    # 训练ML模型以优化预测
    # 使用I_adjusted_simple作为ML模型的目标
    model, scaler = train_ml_model(features_df, I_adjusted_simple)

    # 进行最终预测
    I_ml_adjusted = predict_infections(model, scaler, features_df)

    # 返回结果用于可视化和分析
    results = {
        "dates": dates,
        "confirmed_cases": confirmed_cases,
        "estimated_infections": estimated_infections,
        "I_pred": I_pred,
        "I_adjusted_simple": I_adjusted_simple,
        "I_ml_adjusted": I_ml_adjusted,
        "Rt": Rt
    }

    return results

def visualize_results(results):
    """使用多子图在同一画布上显示不同模型与确诊病例的对比"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    # 子图1: 确诊病例 vs SEIR模型预测
    axes[0].plot(results["dates"], results["confirmed_cases"], label="确诊病例", color="black", linestyle=":",
                 linewidth=2)
    axes[0].plot(results["dates"], results["I_pred"], label="SEIR模型预测", color="blue", linestyle="--", linewidth=2)
    axes[0].set_title("确诊病例 vs SEIR模型预测", fontsize=14)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # 子图2: 确诊病例 vs 基于检测的估计
    axes[1].plot(results["dates"], results["confirmed_cases"], label="确诊病例", color="black", linestyle=":",
                 linewidth=2)
    axes[1].plot(results["dates"], results["estimated_infections"], label="基于检测的估计", color="green",
                 linestyle="-.", linewidth=2)
    axes[1].set_title("确诊病例 vs 基于检测的估计", fontsize=14)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    # 子图3: 确诊病例 vs 简单混合估计
    axes[2].plot(results["dates"], results["confirmed_cases"], label="确诊病例", color="black", linestyle=":",
                 linewidth=2)
    axes[2].plot(results["dates"], results["I_adjusted_simple"], label="简单混合估计", color="purple", linewidth=2)
    axes[2].set_title("确诊病例 vs 简单混合估计", fontsize=14)
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    # 子图4: 确诊病例 vs ML增强预测
    axes[3].plot(results["dates"], results["confirmed_cases"], label="确诊病例", color="black", linestyle=":",
                 linewidth=2)
    axes[3].plot(results["dates"], results["I_ml_adjusted"], label="ML增强预测", color="red", linewidth=2.5)
    axes[3].set_title("确诊病例 vs ML增强预测", fontsize=14)
    axes[3].legend(loc="upper left")
    axes[3].grid(True, alpha=0.3)

    # 为所有子图设置共同的格式
    for ax in axes:
        ax.set_ylabel("感染人数", fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis='x', rotation=45)

    # 只为底部子图添加x轴标签
    for i in [2, 3]:
        axes[i].set_xlabel("日期", fontsize=12)

    plt.suptitle("COVID-19不同预测模型与确诊病例对比", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 为总标题留出空间

    return plt

def main():
    # 加载数据

    data = pd.read_csv(r"F:\pycharm_COVID-19\COVID-19\Data\covid-19-cases-tests-positive-rate-and-reproduction-rate.csv")
    data = pd.read_csv(r'D:\浏览器\covid-19-cases-tests-positive-rate-and-reproduction-rate.csv')
    data = data.loc[data['Entity'].isin(['France', 'Germany', 'North America', 'South Korea'])]
    data = data.dropna()
    print(f"成功加载数据，共{len(data)}行")


    # 设置参数
    total_population = 10_000_000  # 设置适当的人口规模
    print(f"使用总人口: {total_population:,}")

    # 运行混合模型
    print("正在运行混合模型...")
    results = run_hybrid_model(data, total_population)
    print("模型运行完成！")

    # 可视化结果
    print("生成可视化...")
    plt = visualize_results(results)
    plt.savefig("covid_prediction_results.png", dpi=300)
    print("可视化已保存为 'covid_prediction_results.png'")
    plt.show()

    # 保存结果数据
    results_df = pd.DataFrame({
        'Entity':data['Entity'],
        'date': results['dates'],
        'confirmed_cases': results['confirmed_cases'],
        'estimated_actual_infections': results['estimated_infections'],
        'seir_predicted_infections': results['I_pred'],
        'hybrid_prediction': results['I_adjusted_simple'],
        'ml_enhanced_prediction': results['I_ml_adjusted'],
        'reproduction_number': results['Rt']
    })
    results_df.to_csv("prediction_results.csv", index=False)
    print("预测结果已保存为 'prediction_results.csv'")

    # 打印一些关键统计信息
    print("\n模型预测摘要:")
    print(f"预测期间估计最高日感染人数: {int(np.max(results['I_ml_adjusted']))}")
    print(f"预测期间平均Rt值: {np.mean(results['Rt']):.2f}")
    print(f"确诊病例与预测实际感染比例: 1:{np.mean(results['I_ml_adjusted'] / results['confirmed_cases']):.2f}")

    # 完成
    print("\n分析完成！您可以查看生成的图表和CSV文件以获取详细结果。")


if __name__ == "__main__":
    main()