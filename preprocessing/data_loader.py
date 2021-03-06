import pandas as pd
from sklearn.feature_selection import SelectKBest,f_regression

def load_data_with_featureselect(dir,flag = True,k = 20):
    data = pd.read_csv(dir, index_col='Time')
    indexsave = data.reset_index().Time
    # data = data.drop('midprice', axis=1)
    if flag == True:
        X = data.iloc[:,:-1].values
        y = data.iloc[:,-1].values
        X_new = SelectKBest(score_func=f_regression, k=k).fit_transform(X, y)
        X_new = pd.DataFrame(X_new)
        y = pd.DataFrame(y)
        data = pd.concat([X_new,y],axis = 1)
        data['Time'] = indexsave
        data = data.set_index('Time')   # 通过set index 保留 时间戳
    return data


