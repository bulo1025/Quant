import numpy as np
import pandas as pd
import os
import datetime
from scipy.special import expit




data_ETF = pd.read_csv('../dataSet/0512880_Hist.csv')
data_ETF['Time'] = pd.to_datetime(data_ETF.Time)
data_ETF.set_index(['Time'],inplace= True)
data_futures = pd.read_csv('../dataSet/FIH2103_Hist.csv')
data_futures['Time'] = pd.to_datetime(data_futures.Time)
data_futures.set_index(['Time'],inplace= True)
data_index = pd.read_csv('../dataSet/1399975_Hist.csv')
data_index['Time'] = pd.to_datetime(data_index.Time)
data_index.set_index(['Time'],inplace= True)
## 去掉重复值
data_futures = data_futures[~data_futures.index.duplicated(keep='first')]
data_index = data_index[~data_index.index.duplicated(keep='first')]
# 存在不一致的，丢掉，因为第一行是9:29
data_futures = data_futures.iloc[1:,:]
new_features = pd.DataFrame()
# 生成15s的采样
future_close = data_futures['Close'].resample('15s').last().dropna()
index_close = data_index['Close'].resample('15s').last().dropna()
ETF_close = data_ETF['Close'].resample('15s').last().dropna()

# 计算基差一阶差分
basis = (future_close - index_close).dropna()
new_features['basis_diff'] = basis.diff().dropna()

# 计算期货-ETF价差 2H- 1day
C = (future_close.rolling(475).mean() / ETF_close.rolling(475).mean()).dropna()
new_features['spread'] = (future_close - C * ETF_close).dropna()
new_features = new_features.dropna()

if __name__ == '__main__':
    print(new_features.head())
    data_fetures.to_csv('../dataSet/Clean_data/1159949_clean.csv')

