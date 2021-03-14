import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sklearn
data = pd.read_csv('/home/liushaozhe/strategy_data/0510050.csv',index_col='Time')
data = data[['y_pred','target']]
y_pred = data['y_pred']
y_test = data['target']
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data.iloc[:,:-1].values,data.iloc[:,-1].values, test_size=0.3, random_state=42)
# knn = KNeighborsRegressor(n_neighbors=8, weights='uniform', algorithm='auto', metric='minkowski')
# knn.fit(X_train,y_train)
# y_pred = knn.predict(X_test)
# print(np.sqrt(mean_squared_error(y_test, y_pred)))#计算均方差根判断效果
# print(r2_score(y_test,y_pred))#计算均方误差回归损失，越接近于1拟合效果越好
# 趋势准确率
test_results_all = np.sign(y_pred * y_test)
cor = 0
for x in test_results_all:
    if x > 0:
        cor += 1
acc_all = cor * 1.0 / len(test_results_all)
print("The test acc is %f" % acc_all)
#绘图展示预测效果
plt.figure(figsize=(10, 10), dpi=200)
plt.plot(y_pred[:50],label = 'y_pred',color='red',linewidth=1.0, linestyle='--')
plt.plot(y_test[:50],label = 'y_true',color='blue', linewidth=1.0, linestyle='-.')
plt.title('y_pred and y_true', fontsize=12)
plt.legend()
plt.xlabel("time(minute)")  # X轴标签
plt.ylabel("midprice")  # Y轴标签
# plt.savefig(save_dir + '/' + 'y_pred and y_true value.jpg')
plt.show()

