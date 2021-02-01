import numpy as np
import pandas as pd


data_target = '510050.XSHG.csv'
time_window = 60
future_target = 40
STEP = 6
data = pd.read_csv('/home/liushaozhe/dataSet/' + data_target)
TRAIN_SPLIT_RATIO = 0.7
def data_transfer(data):
    """
    提取有效的列，这里，有几列去掉了，需要
    :param data:
    :return:
    """
    # feature target生成
    feature_list = list(data.columns)
    feature_list.pop(0)
    # del feature_list['factor2_4']
    # del feature_list['factor2_6']
    # del feature_list['factor2_12']
    # del feature_list['factor26']
    target = feature_list.pop(1)
    feature_list.pop(13)
    feature_list.pop(13)
    feature_list.pop(13)
    feature_list.pop(-1)
    features = data[feature_list]
    target = data[target]
    data = pd.concat([features,target],axis=1)
    data.astype('float64')
    return data.dropna()


