import pandas as pd
import matplotlib.pyplot as plt
import talib
import os
from tqdm import tqdm

dir = '../dataSet/'
# target_list = ['0512880','0510050','0510500','1150040','1159995']
# target_list = ['0510500','0510050','0512880','1159995']
target_list = ['0510050']


def load_Origin_Data(dir):
    """
    加载原始数据
    :param dir: 股票数据的相对位置   例如：'./dataSet/510050.XSHG/'
    :return:    构造好的dataframe
    """
    csv_ls = []
    for root, dirs, files in os.walk(dir):  # 可以不用写绝对路径只是为了方便
        for filename in files:
            if filename[:7] == target:
                csv_ls.append(filename)
    csv_ls = sorted(csv_ls)
    # print(csv_ls)
    for i in csv_ls:  # 这里只有一个元素啊，懒得改了，到时候多个文件再来改
        data = pd.read_csv(dir + i)
        data['Time'] = pd.to_datetime(data.Time)  # 昂妻案
        data.set_index(['Time'], inplace=True)
        # data = pd.read_csv(dir + i)
    return data.drop(['exchangeCD', 'ticker'], axis=1).astype('float')
for target in tqdm(target_list):
    print('start loading ' + target)

    data = load_Origin_Data(dir)
    data['midprice'] = (data['askPrice1'] + data['bidPrice1']) / 2
    # data = data.iloc[:8000]
    #  rolling
    # bar_data = data['Close'].resample('1T', closed='right', how='ohlc')
    # bar_data_re = bar_data.resample('1S').bfill()[['high','low']]
    # bar_list = [1,3,5,10,15]
    res_data = pd.DataFrame()
    bar_list = [1,3,5,10,15]
    for bar in tqdm(bar_list):
        data_generate = data.resample(str(bar) + 'T').last().dropna()
        # 不能先resample，要不然后面这两个的值都是一样的
        data_generate['high_' + str(bar) + 'minbar'] = data['Close'].resample(str(bar) + 'T', closed='right', how='ohlc')['high']  # 生成1分钟close price 的high bar
        data_generate['low_' + str(bar) + 'minbar'] = data['Close'].resample(str(bar) + 'T', closed='right', how='ohlc')['low']
        # 记录待会要drop掉的
        colums_drop = data_generate.columns
        # （1，3，5，10，15分钟）  最后都需要resample成1分钟的，并用前值填充
        dif, dea, macd1min = talib.MACD(data_generate['Close'].values, 12, 26, 9)
        # 计算MACD
        data_generate['macd' + str(bar) + 'min'] = macd1min
        # 计算bolling线指标
        data_generate['upper20_'+str(bar)], data_generate['middle20_'+str(bar)], data_generate['lower20_'+str(bar)] = talib.BBANDS(data_generate['Close'], 20)
        data_generate['upper30_'+str(bar)], data_generate['middle30_'+str(bar)], data_generate['lower30_'+str(bar)] = talib.BBANDS(data_generate['Close'], 30)
        data_generate['upper50_'+str(bar)], data_generate['middle50_'+str(bar)], data_generate['lower50_'+str(bar)] = talib.BBANDS(data_generate['Close'], 50)
        # 波动率（相对）
        data_generate['relative_flucrate_'+str(bar)] = 2 * (data_generate['high_' + str(bar) + 'minbar'] - data_generate['low_' + str(bar) + 'minbar']) / (data_generate['high_' + str(bar) + 'minbar'] + data_generate['low_' + str(bar) + 'minbar'])
        data_generate['obv_'+str(bar)] = talib.OBV(data_generate['Close'],data_generate['Volume'])
        data_generate['slowk_'+str(bar)], data_generate['slowd' + str(bar)] = talib.STOCH(data_generate['high_' + str(bar) + 'minbar'].values,
                                            data_generate['low_' + str(bar) + 'minbar'].values,
                                            data_generate['Close'].values,
                                            fastk_period=9,
                                            slowk_period=3,
                                            slowk_matype=0,
                                            slowd_period=3,
                                            slowd_matype=0)
        # 日内相对位置指标
        if bar == 1:
            data_generate['relative_pos'] = (data_generate['midprice'] - data_generate['Low'].min()) / (
                        data_generate['High'] - data_generate['Low'].min())
            # 标的 midprice的变化率
            data_generate['midpricemin_ratechange'] = (data_generate['midprice'].shift(bar) - data_generate['midprice']) / data_generate['midprice']
            data_generate['midpricesave'] = data_generate['midprice']
        data_generate = data_generate.drop(colums_drop,axis = 1)
        data_generate = data_generate.dropna()
        if bar != 1:
            data_generate = data_generate.resample('1T').bfill()
        res_data = pd.concat([res_data,data_generate],axis = 1)
    LSTM_traindata = pd.read_csv('/home/liushaozhe/saved_models/0510050/train_pred.csv')
    LSTM_traindata['Time'] = pd.to_datetime(LSTM_traindata.Time)
    LSTM_traindata.set_index(['Time'], inplace=True)
    LSTM_testdata = pd.read_csv('/home/liushaozhe/saved_models/0510050/test_pred.csv')
    LSTM_testdata['Time'] = pd.to_datetime(LSTM_testdata.Time)
    LSTM_testdata.set_index(['Time'], inplace=True)
    LSTM_data = pd.concat([LSTM_traindata,LSTM_testdata],axis = 0)
    # res_data = pd.concat([res_data,LSTM_traindata],axis = 1)
    res_data = pd.concat([res_data, LSTM_data], axis=1)
    res_data = res_data.dropna()
    # res_data['y_pred'] = res_data['y_pred'] / res_data['midpricesave']
    # res_data['target'] = res_data['midpricemin_ratechange']
    # res_data.drop(['midpricemin_ratechange','midpricesave'],axis =1,inplace = True)
    # res_data.to_csv('/home/liushaozhe/strategy_data/' + target + '.csv')

