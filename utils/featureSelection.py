import pandas as pd
from sklearn.feature_selection import SelectKBest,f_regression

from sklearn.feature_selection import SelectKBest,f_regression

def load_data_with_featureselect(data,flag = True,k = 20):
    cor = data.corr().iloc[:,-1]
    if flag == True:
        X = data.iloc[:,:-1].values
        y = data.iloc[:,-1].values
        X_new = SelectKBest(score_func=f_regression, k=k).fit_transform(X, y)
        X_new = pd.DataFrame(X_new)
        y = pd.DataFrame(y)
        data = pd.concat([X_new,y],axis = 1)
    return data,cor
if __name__ == "__main__":
    data = pd.read_csv('/home/liushaozhe/dataSet/clean_data_save/0510050.csv', index_col='Time')
    data = data.drop('weighted_price', axis=1)
    # print(data.head())
    load_data_with_featureselect(data)