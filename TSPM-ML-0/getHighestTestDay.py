import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 删除NaN行

# 加载数据
data = pd.read_csv(r'D:\浏览器\covid-19-cases-tests-positive-rate-and-reproduction-rate.csv')

data = data.loc[data['Entity'].isin(['South Korea'])]  # 2022-03-14 for S-Kr
data = data.dropna()
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.use('Qt5Agg')

plt.figure(figsize=(8, 8), dpi=200)
plt.xlabel("Day")
plt.ylabel("test")

x = data.iloc[:, 1]
y = data.iloc[:, 3]

# 绘制折线图
plt.plot(x, y)

# 找到y的最大值和对应的x值
max_y = y.max()
max_x = x[y.idxmax()]

# 在最大值处画圆圈
plt.scatter(max_x, max_y, color='red', s=100, edgecolor='black', label=f"Max y ({max_x}, {max_y})")

# 显示图例
plt.legend()

# 展示图形
plt.show()

data['Day'] = pd.to_datetime(data['Day'])  # 将日期列转换为 datetime 类型
# 设置截断日期
cutoff_date = pd.to_datetime("2022-03-14")

# 截断数据：保留 'date' 列不晚于 2022-03-14 的数据
filtered_data = data[data['Day'] <= cutoff_date]

# 显示结果
print(filtered_data)


# 对 'COVID-19 positivity rate' 列按从小到大的顺序进行排序
filtered_data_sorted = filtered_data.sort_values(by='COVID-19 positivity rate')
plt.figure(figsize=(8, 8), dpi=200)
plt.xlabel("test")
plt.ylabel("pos")

# 获取排序后的 x 和 y
x_sorted = filtered_data_sorted['Daily COVID-19 tests per 1,000 people (7-day smoothed)']  # 假设 x 对应的是日期列
y_sorted = filtered_data_sorted['COVID-19 positivity rate']
# 绘制折线图
plt.plot(x_sorted, y_sorted)

# 找到y的最大值和对应的x值
max_y = y.max()  # 16.869 ,即Pi
max_x = x[y.idxmax()]


# 显示图例
plt.legend()

# 展示图形
plt.show()

# 计算总天数
total_days = len(filtered_data['Day'].unique())

# 计算 'Daily COVID-19 tests per 1,000 people (7-day smoothed)' 的总和
total_tests = filtered_data['Daily COVID-19 tests per 1,000 people (7-day smoothed)'].sum()

# 计算平均检测量Ti
average_tests = total_tests / total_days

# 打印平均检测量
print(f"The average daily COVID-19 tests per 1,000 people is: {average_tests}")