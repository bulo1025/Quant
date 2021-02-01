# =============神经网络用于分类=============
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV


def report(results,n_top=5):
    for i in range(1,n_top+1):
        candidates=np.flatnonzero(results['rank_test_score']==i)
        for candidate in candidates:
            print("Model with rank:{0}".format(i))
            # print("Mean train score:{0:.3f}(std:{1:.3f})".format(
            #     results['mean_train_score'][candidate],
            #     results['std_train_score'][candidate]))
            print("Mean validation score:{0:.3f}(std:{1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters:{0}".format(results['params'][candidate]))
            print("")
def sample_from_label(data,num):
    ret_data = pd.DataFrame()
    for i in data['label'].unique():
        temp_data = data[data['label'] == i].sample(num,random_state = 1,axis = 0)
        ret_data = pd.concat([ret_data,temp_data])
    return ret_data
data = pd.read_csv('/home/liushaozhe/clean_data_10year_range_2_day/000002.SZ.CSV',encoding = 'gbk')
features = ['前收盘价(元)','开盘价(元)','最高价(元)','最低价(元)','成交量(股)','成交金额(元)','均价(元)','换手率(%)','A股流通市值(元)'
            ,'总市值(元)','A股流通股本(股)','总股本(股)','市盈率','市净率','市销率','市现率']
data = sample_from_label(data,300)
data.sort_index(inplace = True)
# print(data.head())
label = ['label']
data_feature = data[features]
data_label = data['label']
# print(data.shape)
scaler = StandardScaler() # 标准化转换
scaler.fit(data_feature)  # 训练标准化对象
data_feature= scaler.transform(data_feature)   # 转换数据集
train_data,test_data,train_label,test_label = train_test_split(data_feature,data_label,random_state=1, train_size=0.8,test_size=0.2,shuffle = True)
print(train_data.shape)
print(test_data.shape)


# -------------------------模型部分-----------------------#
# 神经网络输入为2，第一隐藏层神经元个数为5，第二隐藏层神经元个数为2，输出结果为2分类。
# solver='lbfgs',  MLP的求解方法：L-BFGS 在小数据上表现较好，Adam 较为鲁棒，
# SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）,SGD标识随机梯度下降。
clf = MLPClassifier(solver='adam',activation='relu',alpha=1e-10,hidden_layer_sizes=(588,100),learning_rate='adaptive', random_state=1,shuffle=True,max_iter=5000, verbose = True)
# selector = SelectFromModel(estimator=MLPClassifier()).fit(data_feature, data_label)
clf.fit(train_data,train_label)
predict_results = clf.predict(test_data)
# # print(clf.score())
# # 绘制混淆矩阵
sns.set()
f,ax=plt.subplots()
C2= confusion_matrix(test_label, predict_results, labels=[0, 1, 2])
print(C2) #打印出来看看
print(clf.score(test_data,test_label))
sns.heatmap(C2,annot=True,ax=ax) #画热力图
ax.set_title('confusion matrix') #标题
ax.set_xlabel('predict') #x轴
ax.set_ylabel('true') #y轴
plt.show()



# -----------------超参数优化--------------------#
# clf = MLPClassifier(max_iter=100,verbose=True)
# #设置想要优化的超参数以及他们的取值分布
# parameter_space = { 'hidden_layer_sizes': [(sp_randint.rvs(100,600,1),sp_randint.rvs(100,600,1),),
#                                            (sp_randint.rvs(100,600,1),),
#                                            (sp_randint.rvs(100,600,1),sp_randint.rvs(100,600,1),sp_randint.rvs(100,600,1)),
#                                            (sp_randint.rvs(100,600,1),sp_randint.rvs(100,600,1),sp_randint.rvs(100,600,1),sp_randint.rvs(100,600,1))],
#                     'activation': ['tanh', 'relu', 'logistic'],
#                     'solver': ['sgd', 'adam', 'lbfgs'],
#                       'tol':[0.0000000001],
#                     'alpha': uniform(0.0001, 0.9),
#                     'learning_rate': ['constant','adaptive']}
# #开启超参数空间的随机搜索
# n_iter_search=200
# random_search=RandomizedSearchCV(clf,param_distributions=parameter_space,n_iter=n_iter_search)
# start=time()
# # random_search.fit(train_data,train_label)  # X,y
# random_search.fit(data_feature,data_label)
# print("RandomizedSearchCV took %.3f seconds for %d candidates"
#       "parameter settings."%((time()-start),n_iter_search))
# report(random_search.cv_results_)