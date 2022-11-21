# Step1:导入标准库，Importing the Libraries
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import train_test_split
import lightgbm as LGB
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error,r2_score
from sklearn.metrics import make_scorer
from  sklearn.model_selection import GridSearchCV


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

#1.建立lightgbm基础模型
#先利用一些常规超参数设置，使用较低的学习率获得一个基础模型，方便后续的调参效率。使用callback回调方法early stopping控制过拟合风险，当验证集上的精度若干轮不下降，提前停止训练。
#定义回归模型评估误差指标
def median_absolute_percentage_error(y_true,y_pred):
    return np.median(np.abs((y_pred-y_true)/y_true))
def regression_metrics(true,pred):
    print('回归模型评估指标结果:')
    print('均方误差【MSE】:', mean_squared_error(true, pred))
    print('均方根误差【RMSE】:',np.sqrt(mean_squared_error(true,pred)))
    print('平均绝对误差【MAE】:',mean_absolute_error(true,pred))
    print('绝对误差中位数【MedianAE】:',median_absolute_error(true,pred))
    print('平均绝对百分比误差【MAPE】:',mean_absolute_percentage_error(true,pred))
    print('绝对百分比误差中位数【MedianAPE】:',median_absolute_percentage_error(true,pred))
    print('确定系数(R^2): %.2f' % r2_score(true,pred))

#建立LGB的dataset格式数据
lgb_train = LGB.Dataset(x_train, y_train)
lgb_eval = LGB.Dataset(x_test, y_test, reference=lgb_train)
#定义超参数dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
 #   'objective': 'regression',
 #   'max_depth': 7,
 #   'num_leaves': 31,
    'objective': 'regression_l2',
    'max_depth': 8,
    'num_leaves':14,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
  #  'bagging_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': -1,
    #新加
    'metric':'mae',
    'reg_alpha':0.01,
    'reg_lamba':0.3,
    'min_child_samples':8,
    'min_child_weight':0.00005

}
#定义callback回调
callback=[LGB.early_stopping(stopping_rounds=10,verbose=True),
          LGB.log_evaluation(period=10,show_stdv=True)]
# 训练 train
m1 = LGB.train(params,lgb_train,num_boost_round=2000,
               valid_sets=[lgb_train,lgb_eval],callbacks=callback)
#预测数据集
y_pred = m1.predict(x_test)
#评估模型
regression_metrics(y_test,y_pred)

#未调参前
#真实应变预测误差：平均绝对百分比误差【MAPE】: 0.028034048382236155，绝对百分比误差中位数【MedianAPE】: 0.002982211508658856
#真实应力预测误差：平均绝对百分比误差【MAPE】: 0.03304580917320333，绝对百分比误差中位数【MedianAPE】: 0.0003463161603396225

#调参后
#真实应变预测误差：平均绝对百分比误差【MAPE】: 0.023860036261903063，绝对百分比误差中位数【MedianAPE】: 0.0031388127855464577
#真实应力预测误差：平均绝对百分比误差【MAPE】: 0.011036745948315704，绝对百分比误差中位数【MedianAPE】: 0.0004910448646712003









#2.模型优化
#目标函数（objective）与评估函数（metrics）在GBM模型中至关重要，目标函数影响模型的学习，而callback中的early_stopping受评估函数的控制。根据数据的分布情况以及最终的目标选择合适的目标函数与评估函数的组合，对模型效果有较大的影响。
#其他重要的核心超参数，均通过俩俩网格搜索（Grid Search）的方法进行调参，选择最优参数组合，此处注意，因为使用了5折交叉验证来选择超参数，所以所有GridSearch过程均在全量样本上来fit

# #2.1目标函数和评估函数的选择
# #选择5种目标函数，5种评估函数，俩俩组合后观测模型误差
# objective=['regression_l2','regression_l1','quantile','poisson','mape']
# metrics=['l2','mae','quantile','poisson','mape']
# metrics_test_data=pd.DataFrame(columns=['objective','metric','MAPE','Median APE','MAE'])
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
#               '开始目标函数与评估函数评估')
# for i in objective:
#     for k in metrics:
#         size=metrics_test_data.size
#         params = {
#             'task': 'train',
#             'boosting_type': 'gbdt',
#             'objective': i,
#             'metric':k,
#             'max_depth': 7,
#             'num_leaves': 31,
#             'learning_rate': 0.1,
#             'feature_fraction': 0.8,
#             'bagging_fraction': 0.8,
#             'bagging_freq': 5,
#             'verbose': -1
#         }
#         callback=LGB.early_stopping(stopping_rounds=10,verbose=0)
#         gbm = LGB.train(params,lgb_train,num_boost_round=2000,
#                 valid_sets=lgb_eval,callbacks=[callback])
#         y_pred = gbm.predict(x_test)
#         metrics_test_data.loc[size]=[i,k,mean_absolute_percentage_error(y_test,y_pred),
#                                      median_absolute_percentage_error(y_test,y_pred),
#                                      mean_absolute_error(y_test,y_pred)
#                                     ]
#         print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),i,'+',k,' 完成评估',
#               ' best iteration is:',gbm.best_iteration)
#
# #metrics_test_data
# #选择regression_l2+mae的函数组合。







# #2.2 树深度与叶子节点数超参数GridSearch
# #我们使用网格搜索进行调参，因sklearn中的GridSearchCV过程中的scoring未提供我们重点观测的Median APE指标，所以需要自定义。
neg_median_absolute_percentage_error = make_scorer(median_absolute_percentage_error,
                                                   greater_is_better=False)
#
#
# #注意，在使用网格搜索过程中，因为引入了N折交叉验证的方法，所以所有GridSearch过程均在全量样本上来fit
# #另外，sklearn接口的参数名称与LGB有差异，例如subsample（bagging_fraction）等等，虽然用LGB默认叫法也能run，但是可能会有waring。
# #网格搜索过程：首先声明一个基础模型model_lgb设置需要搜索的参数params_test1，支持range()方法，也可以写list设置GridSearchCV()，scoring为评估函数，cv=5为五折交叉验证
#
# #开始gridsearch
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'开始树深度和叶子结点数GridSearch')
# model_lgb = LGB.LGBMRegressor(objective='regression_l2',
#                               metric='mae',
#                               learning_rate=0.1,
#                               subsample = 0.8,
#                               colsample_bytree = 0.8,
#                               subsample_freq = 5)
# params_test1={
#     'max_depth': range(7,11,1),
#     #'num_leaves':range(10,90,10)
#     'num_leaves':range(10,30,1)
# }
# gsearch1 = GridSearchCV(estimator=model_lgb,
#                         param_grid=params_test1,
#                         scoring=neg_median_absolute_percentage_error,
#                         cv=5,
#                         verbose=1,
#                         n_jobs=-1)
# gsearch1.fit(x, y)
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'完成树深度和叶子结点数GridSearch')
# print('Best parameters found by grid search are:', gsearch1.best_params_)
#
# #当'num_leaves':range(10,90,10)
# #结果：Best parameters found by grid search are: {'max_depth': 7, 'num_leaves': 20}
#
# #调整直至'num_leaves':range(10,30,1)
# #Best parameters found by grid search are: {'max_depth': 8, 'num_leaves': 14}

#2.3搜索其他超参数
# #叶子结点最小数据数量与最小hessian GridSearch
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'开始叶子结点最小数据数量与最小hessian GridSearch')
# model_lgb = LGB.LGBMRegressor(objective='regression_l2',
#                               metric='mae',
#                               learning_rate=0.1,
#                               subsample = 0.8,
#                               colsample_bytree = 0.8,
#                               subsample_freq = 5,
#                               max_depth=8,num_leaves=14)
#
# params_test3={
#     'min_child_samples':[6,7,8,9,10],
#     'min_child_weight':[0.00005,0.00001,0.0001]
#  }
#
# gsearch1 = GridSearchCV(estimator=model_lgb,
#                         param_grid=params_test3,
#                         scoring=neg_median_absolute_percentage_error,
#                         cv=5,
#                         verbose=1,
#                         n_jobs=-1)
# gsearch1.fit(x, y)
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'完成叶子结点最小数据数量与最小hessian GridSearch')
# print('Best parameters found by grid search are:', gsearch1.best_params_)
#
# #Best parameters found by grid search are: {'min_child_samples': 8, 'min_child_weight': 0.00005}



# #特征与样本的随机采样率
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'开始特征与样本的随机采样率 GridSearch')
# model_lgb = LGB.LGBMRegressor(objective='regression_l2',
#                               metric='mae',
#                               learning_rate=0.1,
# #                              subsample = 0.8,
# #                              colsample_bytree = 0.8,
#                               subsample_freq = 5,
#                               max_depth=8,num_leaves=14,
#                               min_child_samples=8, min_child_weight=0.00005)
#
# params_test4={
#     'subsample': [0.7,0.8,0.9],
#     'colsample_bytree': [0.7,0.8,0.9]
# }
#
# gsearch1 = GridSearchCV(estimator=model_lgb,
#                         param_grid=params_test4,
#                         scoring=neg_median_absolute_percentage_error,
#                         cv=5,
#                         verbose=1,
#                         n_jobs=-1)
# gsearch1.fit(x, y)
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'完成特征与样本的随机采样率 GridSearch')
# print('Best parameters found by grid search are:', gsearch1.best_params_)
# #Best parameters found by grid search are: {'colsample_bytree': 0.8, 'subsample': 0.9}
#

# #正则化参数
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'开始正则化参数 GridSearch')
# model_lgb = LGB.LGBMRegressor(objective='regression_l2',
#                               metric='mae',
#                               learning_rate=0.1,
#                               subsample = 0.9,
#                               colsample_bytree = 0.8,
#                               subsample_freq = 5,
#                               max_depth=8,num_leaves=14,
#                               min_child_samples=8, min_child_weight=0.00005)
#
# params_test5={
#     'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3],
#     'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3]
# }
#
# gsearch1 = GridSearchCV(estimator=model_lgb,
#                         param_grid=params_test5,
#                         scoring=neg_median_absolute_percentage_error,
#                         cv=5,
#                         verbose=1,
#                         n_jobs=-1)
# gsearch1.fit(x, y)
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'完成正则化参数 GridSearch')
# print('Best parameters found by grid search are:', gsearch1.best_params_)
# #Best parameters found by grid search are: {'reg_alpha': 0.01, 'reg_lambda': 0.3}
#
#
