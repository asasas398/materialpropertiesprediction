import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import RidgeCV, LassoCV,ElasticNetCV,Lasso,LinearRegression
from sklearn.preprocessing import StandardScaler



dataset = pd.read_excel(r"D:\data\A1500.xlsx",sheet_name=1)
x = dataset.iloc[2:3276,[3,4]].values
y =  dataset.iloc[2:3276,6].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#正则化
scaler2 = StandardScaler()
X_train = scaler2.fit_transform(x_train)
X_test = scaler2.transform(x_test)


#调用线性回归模型
regr = LinearRegression(fit_intercept=True, n_jobs=-1)

# 调用其他模型，使用训练集数据进行训练,其中alpha的值为岭系数，scikit-learn提供了类RidgeCV，它可以自动执行网格搜索，来寻找最佳值
model = LassoCV(alphas=np.logspace(-10,-2,200,base=10))
#model = RidgeCV(alphas=np.logspace(-10,-2,200,base=10))
#model= ElasticNetCV(alphas=np.logspace(-10,-2,200,base=10), l1_ratio=(0.1, 0.25, 0.5, 0.75, 0.8))



#线性回归模型拟合和预测并且进行评价
regr.fit(X_train,y_train)
y_pred = regr.predict(X_test)
print('系数: \n', regr.coef_)
print("Mean Absolute Error : " + str(mean_absolute_error(y_pred, y_test)))
print('均方误差: %.2f'% mean_squared_error(y_test, y_pred))
print('确定系数(R^2): %.2f'% r2_score(y_test, y_pred))




#使用其他模型拟合
model.fit(X_train, y_train)
print(model.alpha_)
#print('ElasticNet optimal alpha: %.3f and L1 ratio: %.4f' % (model.alpha_, model.l1_ratio_))



# 使用其他模型对测试集数据进行预测
predictions = model.predict(X_test)

# 对模型的预测结果进行评价
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))
print('均方误差: %.2f' % mean_squared_error(y_test, predictions))
print('确定系数(R^2): %.2f' % r2_score(y_test, predictions))
print('绝对百分比误差中位数【MedianAPE】:',mean_absolute_percentage_error(y_test,predictions))



#ridge预测结果：optimal alpha:1e-06，Mean Absolute Error : 0.3476181806426796
#lasso预测结果：optimal alpha:0.0006222570836730231，Mean Absolute Error : 0.3529421061228235
#ElasticNet预测结果 optimal alpha: 0.000 and L1 ratio: 0.8000,Mean Absolute Error : 0.35314399286252485

#绘图
t = np.arange(len(x_test))
plt.plot(t, y_test, 'r-', linewidth=2, label=u'原值')
plt.plot(t, predictions, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='upper right')
plt.grid()
plt.show()


