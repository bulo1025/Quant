import pandas as pd
import numpy as np



def Preprocess_data(df, index=[], drop_num=21):
    # drop useless columns
    df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    df['average_price'] = (df['askPrice1']+df['bidPrice1'])/2
    df['ABVolRatioIndex_1'] = (df['askVolume1']+df['askVolume2'])/(df['bidVolume1']+df['bidVolume2'])
    df['ABVolRatioIndex_2'] = (df['askVolume1']+df['askVolume2']+df['askVolume3']+df['askVolume4']+df['askVolume5']) / \
        (df['bidVolume1']+df['bidVolume2']+df['bidVolume3']+df['bidVolume4']+df['bidVolume5'])
    df['askX'] = (df['value']-df['volume']*df['bidPrice1'])/(df['askPrice1']-df['bidPrice1'])
    df['bidY'] = df['volume']-df['askX']
    df['moving_average'] = (df['average_price']+df['average_price'].shift(1)+df['average_price'].shift(2)+df['average_price'].shift(3)+df['average_price'].shift(4)+df['average_price'].shift(5))/6
    df['askprice1_diff'] = df['askPrice1'] - df['askPrice1'].shift(1)
    df['bidprice1_diff'] = df['bidPrice1'] - df['bidPrice1'].shift(1)
    df['askvolume1_diff'] = df['askVolume1'] - df['askVolume1'].shift(1)
    df['bidvolume1_diff'] = df['bidVolume1'] - df['bidVolume1'].shift(1)
    df.dropna(inplace=True)

    # df = df.drop(columns=index, axis=1)
    # if drop_num != 0:
    #     df = df.iloc[1:]
    #     df = df.iloc[:-drop_num]
    # # reindex
    # df.index = range(0, len(df))

    index = []
    eps = 0.02
    data = {"volume": [], "moment": [], "value": [], "Close" : [] ,"price": [], "price_change": [], "gain": [],
            'ABVolRatioIndex_1': [], 'ABVolRatioIndex_2': [], "OrderImbalance1": [], "askX": [], "bidY": [], 'PosPercentIndex': []}
    last_price = df.iloc[0]["average_price"]

    for i, row in df.iterrows():
        delta = (row["average_price"] - last_price)/last_price
        index.append(row['Time'])
        data["volume"].append(row["volume"])
        data["Close"].append(row["Close"])
        data["askX"].append(row["askX"])
        data["bidY"].append(row["bidY"])
        data["ABVolRatioIndex_1"].append(row["ABVolRatioIndex_1"])
        data["ABVolRatioIndex_2"].append(row["ABVolRatioIndex_2"])
        data["moment"].append(row["volume"] * delta)
        data["value"].append(row["value"])
        data["price"].append(row["average_price"])
        data["price_change"].append(row["average_price"])
        data["gain"].append(row["average_price"])
        data['PosPercentIndex'].append((row["average_price"]-row["moving_average"])/row["moving_average"])
        delta_bidvolume1 = row['bidVolume1']*(row['bidprice1_diff'] > 0) + row['bidvolume1_diff']*(row['bidprice1_diff'] == 0)
        delta_askvolume1 = row['askVolume1']*(row['askprice1_diff'] < 0) + row['askvolume1_diff']*(row['askprice1_diff'] == 0)
        data["OrderImbalance1"].append(delta_bidvolume1-delta_askvolume1)

        last_price = row["average_price"]

    index = pd.to_datetime(index)
    df = pd.DataFrame(data, index=index)
    df['Time'] = index

    # calculate difference of close price
    df['close_diff'] = df['Close'] / df['Close'].shift(1) - 1
    df['price_change'] = df['price'] / df['price'].shift(1) - 1
    df['close_diff'] = df['close_diff']-df['price_change']

    # predict interval gain
    df['gain'] = df['price'].shift(-20) / df['price'] - 1

    df.dropna(inplace=True)

    # df['price_change'] = df['price_change'].apply(lambda x: np.where(x >= eps, eps, np.where(x > -eps, x, -eps)))  # 去极值
    df['gain'] = df['gain'].apply(lambda x: np.where(x >= eps, eps, np.where(x > -eps, x, -eps)))  # 去极值
    df['gain'] = df['gain'] * 10  # 适当增大return范围，利于LSTM模型训练
    df['close_diff'] = df['close_diff'] * 100  # 适当增大return范围，利于LSTM模型训练

    df.reset_index(drop=True, inplace=True)

    return df

a = pd.read_csv('/home/liushaozhe/dataSet/0510050_Hist.csv')
a = Preprocess_data(a,['ticker', 'exchangeCD'])
b = pd.read_csv('/home/liushaozhe/dataSet/clean_data_save/0510050.csv')
res = pd.merge(b,a)
print(res.info())



import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import preprocessing
from keras.layers import Input, Dense, LSTM, merge
from keras.models import Model
from keras.models import load_model

import pandas as pd
import numpy as np
import os
import logging
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, core, Embedding
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


class conf:
    train_days = 20
    test_days = 1
    t = time.time()
    start_t = t - train_days * 24 * 60 * 60
    split_t = t - test_days * 24 * 60 * 60
    seq_len = 240  # 每个input的长度
    fields = ["close_diff", "volume", "moment", "value", "deal", "price", 'OrderImbalance1', 'ABVolRatioIndex_1', 'ABVolRatioIndex_2', 'askX', 'bidY', 'PosPercentIndex']
    train_proportion = 0.8  # 训练数据占总数据量的比值，其余为测试数据
    normalise = True  # 数据标准化
    epochs = 3  # LSTM神经网络迭代次数
    batch = seq_len  # 整数，指定进行梯度下降时每个batch包含的样本数,训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步
    validation_split = 0.5  # 0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。
    lr = 0.001  # 学习效率
    price_step = 0.001


def load_data(path):
    df = pd.read_csv(path, encoding='gbk')
    return df


def Preprocess_data(hist_IF, df, index=[], drop_num=21):

    # drop useless columns
    df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    df['average_price'] = (df['askprice1']+df['bidprice1'])/2
    df['ABVolRatioIndex_1'] = (df['askvolume1']+df['askvolume2'])/(df['bidvolume1']+df['bidvolume2'])
    df['ABVolRatioIndex_2'] = (df['askvolume1']+df['askvolume2']+df['askvolume3']+df['askvolume4']+df['askvolume5']) / \
        (df['bidvolume1']+df['bidvolume2']+df['bidvolume3']+df['bidvolume4']+df['bidvolume5'])
    df['askX'] = (df['value']-df['volume']*df['bidprice1'])/(df['askprice1']-df['bidprice1'])
    df['bidY'] = df['volume']-df['askX']
    df['moving_average'] = (df['average_price']+df['average_price'].shift(1)+df['average_price'].shift(2)+df['average_price'].shift(3)+df['average_price'].shift(4)+df['average_price'].shift(5))/6
    df['askprice1_diff'] = df['askprice1'] - df['askprice1'].shift(1)
    df['bidprice1_diff'] = df['bidprice1'] - df['bidprice1'].shift(1)
    df['askvolume1_diff'] = df['askvolume1'] - df['askvolume1'].shift(1)
    df['bidvolume1_diff'] = df['bidvolume1'] - df['bidvolume1'].shift(1)
    df.dropna(inplace=True)

    df = df.drop(columns=index, axis=1)
    if drop_num != 0:
        df = df.iloc[1:]
        df = df.iloc[:-drop_num]
    # reindex
    df.index = range(0, len(df))

    index = []
    eps = 0.02
    data = {"volume": [], "moment": [], "value": [], "deal": [], "price": [], "price_change": [], "gain": [],
            'ABVolRatioIndex_1': [], 'ABVolRatioIndex_2': [], "OrderImbalance1": [], "askX": [], "bidY": [], 'PosPercentIndex': []}
    last_price = df.iloc[0]["average_price"]

    for i, row in df.iterrows():
        delta = (row["average_price"] - last_price)/last_price
        index.append(row['datetime'])
        data["volume"].append(row["volume"])
        data["askX"].append(row["askX"])
        data["bidY"].append(row["bidY"])
        data["ABVolRatioIndex_1"].append(row["ABVolRatioIndex_1"])
        data["ABVolRatioIndex_2"].append(row["ABVolRatioIndex_2"])
        data["moment"].append(row["volume"] * delta)
        data["value"].append(row["value"])
        data["deal"].append(row["deal"])
        data["price"].append(row["average_price"])
        data["price_change"].append(row["average_price"])
        data["gain"].append(row["average_price"])
        data['PosPercentIndex'].append((row["average_price"]-row["moving_average"])/row["moving_average"])
        delta_bidvolume1 = row['bidvolume1']*(row['bidprice1_diff'] > 0) + row['bidvolume1_diff']*(row['bidprice1_diff'] == 0)
        delta_askvolume1 = row['askvolume1']*(row['askprice1_diff'] < 0) + row['askvolume1_diff']*(row['askprice1_diff'] == 0)
        data["OrderImbalance1"].append(delta_bidvolume1-delta_askvolume1)

        last_price = row["average_price"]

    index = pd.to_datetime(index)
    df = pd.DataFrame(data, index=index)
    df['datetime'] = index

    df = pd.merge(df, hist_IF, on='datetime')

    # calculate difference of close price
    df['close_diff'] = df['Close'] / df['Close'].shift(1) - 1
    df['price_change'] = df['price'] / df['price'].shift(1) - 1
    df['close_diff'] = df['close_diff']-df['price_change']

    # predict interval gain
    df['gain'] = df['price'].shift(-20) / df['price'] - 1

    df.dropna(inplace=True)

    # df['price_change'] = df['price_change'].apply(lambda x: np.where(x >= eps, eps, np.where(x > -eps, x, -eps)))  # 去极值
    df['gain'] = df['gain'].apply(lambda x: np.where(x >= eps, eps, np.where(x > -eps, x, -eps)))  # 去极值

    df['gain'] = df['gain'] * 10  # 适当增大return范围，利于LSTM模型训练
    df['close_diff'] = df['close_diff'] * 100  # 适当增大return范围，利于LSTM模型训练

    df.reset_index(drop=True, inplace=True)
    scaledata = df[conf.fields]

    # 数据处理：设定每个input（30time series×6features）以及数据标准化
    train_input = []
    train_output = []

    for i in range(conf.seq_len - 1, len(scaledata)):
        scaler = preprocessing.StandardScaler()
        a = scaler.fit_transform(scaledata[i + 1 - conf.seq_len:i + 1])
        train_input.append(a)
        c = df['gain'][i]
        train_output.append(c)

    return train_input, train_output


def binddata(hist_IF, date, stock_list):
    total_input = []
    total_output = []
    for stock in stock_list:
        for item in date:
            df1 = load_data(r'/root/lstm/stock/2020/%s/%s/%s.csv' % (item[0:2], item[2:], stock))
            train_input, train_output = Preprocess_data(hist_IF, df1, ['ticker', 'exchangecd', 'shortnm', 'prevcloseprice', 'openprice'])
            total_input.extend(train_input)
            total_output.extend(train_output)
            print('successfully bind date:', item)
    return total_input, total_output


def train(train_input, train_output):

    # LSTM接受数组类型的输入
    train_x = np.array(train_input)
    train_y = np.array(train_output)

    # 构建神经网络层 1层LSTM层+3层Dense层
    # 用于1个输入情况

    lstm_input = Input(shape=(conf.seq_len, len(conf.fields)), name='lstm_input')
    lstm_output1 = LSTM(conf.seq_len, activation='tanh', dropout=0.2, recurrent_dropout=0.1, return_sequences=True)(lstm_input)
    lstm_output2 = LSTM(conf.seq_len, activation='tanh', dropout=0.2, recurrent_dropout=0.1, return_sequences=False)(lstm_output1)
    Dense_output_1 = Dense(64, activation='linear')(lstm_output2)
    Dense_output_2 = Dense(16, activation='linear')(Dense_output_1)
    predictions = Dense(1, activation='tanh')(Dense_output_2)

    model = Model(inputs=lstm_input, outputs=predictions)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    history = model.fit(train_x, train_y, batch_size=conf.batch, epochs=conf.epochs, validation_split=conf.validation_split, verbose=2, shuffle=True)
    model.save(r'/root/lstm/stock/my_model.h5')

    predict_y = model.predict(train_x)
    distribution = {}
    for i in range(len(predict_y)):
        count = 1
        if predict_y[i] * train_y[i] < 0:
            count = 0
        level = int(predict_y[i] / conf.price_step) * conf.price_step

        if level in distribution:
            distribution[level]["all"] += 1
            distribution[level]["right"] += count
        else:
            distribution[level] = {"all": 1, "right": count}

    print("Distribution of right:\n", distribution)

    correct = 0
    for level, value in distribution.items():
        correct += value['right']
    print("total accuracy:", correct/len(predict_y))


def predict(test_input, test_output):
    test_x = np.array(test_input)
    test_y = np.array(test_output)

    model = load_model(r'/root/lstm/stock/my_model.h5')

    predict_y = model.predict(test_x)
    distribution = {}
    for i in range(len(predict_y)):
        count = 1
        if predict_y[i] * test_y[i] < 0:
            count = 0
        level = int(predict_y[i] / conf.price_step) * conf.price_step

        if level in distribution:
            distribution[level]["all"] += 1
            distribution[level]["right"] += count
        else:
            distribution[level] = {"all": 1, "right": count}
    print("Distribution of right:\n", distribution)
    correct = 0
    for level, value in distribution.items():
        correct += value['right']
    print("total accuracy:", correct/len(predict_y))


hist_IF = pd.read_csv(r'/root/lstm/hist_index_futures/FIF2012_Hist.csv', encoding='gbk')


hist_IF = hist_IF.drop(columns=['Open', 'High', 'Low', 'Volume'], axis=1)
hist_IF['Time'] = pd.to_datetime(hist_IF['Time'])
hist_IF.set_index(['Time'], inplace=True)
hist_IF = hist_IF.resample(rule='3S', label='right', closed='right').mean()
hist_IF.dropna(inplace=True)
hist_IF['datetime'] = hist_IF.index
hist_IF.reset_index(drop=True, inplace=True)

# stock_list = ['002945.XSHE', '002939.XSHE', '601698.XSHG', '601236.XSHG']#76%
#stock_list = ['300003.XSHE', '300122.XSHE', '300142.XSHE', '300347.XSHE', '300413.XSHE', '300628.XSHE']
#stock_list = ['600438.XSHG','600703.XSHG','600745.XSHG','601108.XSHG','601878.XSHG','603501.XSHG','603799.XSHG','603986.XSHG','000066.XSHE','000723.XSHE','000977.XSHE']


stock_list = [
    '600760.XSHG',
    '600928.XSHG',
    '600968.XSHG',
    '600989.XSHG',
    '601138.XSHG',
    '601298.XSHG',
    '601319.XSHG',
    '601577.XSHG',
    '601816.XSHG',
    '002958.XSHE',
    '300059.XSHE'
]

# total_train_input,total_train_output=binddata(hist_IF,['1009','1013','1014','1015'])
total_train_input, total_train_output = binddata(hist_IF, ['0901', '0902', '0903', '0904', '0907', '0908', '0909', '0910', '0911', '0914',
                                                           '0915', '0916', '0917', '0918', '0921', '0922', '0923', '0924', '0925', '0928', '0929', '0930'], stock_list)

train(total_train_input, total_train_output)

total_test_input, total_test_output = binddata(hist_IF, ['1009', '1013', '1014', '1015', '1016', '1019', '1020'], stock_list)
# total_test_input,total_test_output=binddata(['1009','1012','1013','1014','1015','1016','1019','1020'])
#total_test_input, total_test_output = binddata(['1009', '1013', '1014', '1015', '1016', '1019', '1020'])
predict(total_test_input, total_test_output)
