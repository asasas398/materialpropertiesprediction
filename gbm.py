#设置超参数objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20, verbosity=2
#LGB核心超参数：task，指定任务，可选train，predict等。boosting_type，提升方法，可选gbdt、rf等。objective，目标（函数），如果是回归任务，可l2、l1、huber、quantile等；如果是分类任务，可选binary、multiclass等。max_depth，树的最大深度，控制过拟合的有效手段。num_leaves，树的最大叶子节点数。feature_fraction，特征的随机采样率，指bagging_fraction，样本的随机采样率bagging_freq,是否启用bagging并设置迭代轮次，如启用，上述的特征与样本的的随机采样需要设置。learning_rate，学习率lambda_l1，L1正则化lambda_l2，L2正则化


# Step1:导入标准库，Importing the Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# 导入数据集
dataset = pd.read_excel(r"D:\data\A1500.xlsx",sheet_name=1)

# 打印数据
#print(tabulate(dataset, headers='keys'))

# 自变量与因变量分离,自变量是拉伸应变，拉伸应力，因变量为真实应变or真实应力
# 自变量对应第4,5列 因变量对应第6or7列；行（第3行-3273行）-无缺失值
# 根据预测的是真实应变or真实应力调整y值
x = dataset.iloc[2:3276,[3,4]].values
y =  dataset.iloc[2:3276,6].values

#切分训练和测试集,test_size=0.2 train_size默认0.75
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# 调用LightGBM模型，使用训练集数据进行训练（拟合）
# Add verbosity=2 to print messages while running boosting
my_model = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20,
                             verbosity=2)
my_model.fit(x_train, y_train, verbose=False)

# 使用模型对测试集数据进行预测
predictions = my_model.predict(x_test)

# 对模型的预测结果进行评判（平均绝对误差）
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))
print('均方误差: %.2f' % mean_squared_error(y_test, predictions))
print('确定系数(R^2): %.2f' % r2_score(y_test, predictions))
