import pandas as pd
import os
import datetime
import numpy as np
from scipy.special import expit
import seaborn as sns

origin_data = '510050.XSHG'

def load_Origin_Data(dir):
    """
    加载原始数据
    :param dir: 股票数据的相对位置   例如：'./dataSet/510050.XSHG/'
    :return:    构造好的dataframe
    """
    csv_ls = []
    for root, dirs, files in os.walk(dir):    # 可以不用写绝对路径只是为了方便
        for filename in files:
            if filename[-3:]=='csv':
                csv_ls.append(filename)
    csv_ls = sorted(csv_ls)
    # print(csv_ls)
    csv_ls = csv_ls[:-1]
    data50ETF =[]
    for i in csv_ls:
        day_data = pd.read_csv(dir + i)
        data50ETF.append(day_data)
    data50ETF = pd.concat(data50ETF, ignore_index=True)
    data50ETF['datetime'] = [datetime.datetime.strptime(i, '%Y%m%d %H:%M:%S') for i in data50ETF.datetime]
    data50ETF.index=data50ETF.datetime
    del data50ETF['datetime']
    del data50ETF['exchangecd']
    del data50ETF['ticker']
    return data50ETF

data = load_Origin_Data('./dataSet/' + origin_data + '/')

# 变量转换函数
def variable_shift(data):
    open = data.openprice
    close = data.lastprice
    high = data.highprice
    low = data.lowprice
    a5 = data.askprice5
    a4 = data.askprice4
    a3 = data.askprice3
    a2 = data.askprice2
    a1 = data.askprice1
    b1 = data.bidprice1
    b2 = data.bidprice2
    b3 = data.bidprice3
    b4 = data.bidprice4
    b5 = data.bidprice5
    v_a5 = data.askvolume5
    v_a4 = data.askvolume4
    v_a3 = data.askvolume3
    v_a2 = data.askvolume2
    v_a1 = data.askvolume1
    v_b1 = data.bidvolume1
    v_b2 = data.bidvolume2
    v_b3 = data.bidvolume3
    v_b4 = data.bidvolume4
    v_b5 = data.bidvolume5
    askprice = pd.concat([a5,a4,a3,a2,a1], axis=1)   # 卖家喊出来的价格
    bidprice = pd.concat([b1,b2,b3,b4,b5], axis=1)   # 买家提供的价格
    askvolume = pd.concat([v_a5,v_a4,v_a3,v_a2,v_a1], axis=1)
    bidvolume = pd.concat([v_b1,v_b2,v_b3,v_b4,v_b5], axis=1)
    return open,close,high,low,askprice,bidprice,askvolume,bidvolume,a1,a2,a3,a4,a5,b1,b2,b3,b4,b5,v_a1,v_a2,v_a3,v_a4,v_a5,v_b1,v_b2,v_b3,v_b4,v_b5

open,close,high,low,askprice,bidprice,askvolume,bidvolume,a1,a2,a3,a4,a5,b1,b2,b3,b4,b5,v_a1,v_a2,v_a3,v_a4,v_a5,v_b1,v_b2,v_b3,v_b4,v_b5 = variable_shift(data)   # 前面几个因子用到了构建因子
# # 因子2 加权价格WP
# # WP1因子
def weighted_middle_price(a1, b1, v_a1, v_b1):
    return (b1 * v_b1 + a1 * v_a1) / (v_b1 + v_a1)
# midprc 因子
def mid_price(a1, b1):
    return (a1 + b1) / 2
temp = weighted_middle_price(a1, b1, v_a1, v_b1)
data['weighted_midprice'] = temp

class mBar(object):
    def __init__(self):
        """Constructor"""
        self.open = None
        self.close = None
        self.high = None
        self.low = None
        self.datetime = None


bar = None
m_bar_list = list()

for datetime, last in data[['lastprice']].iterrows():
    new_minute_flag = False

    if not bar:  # 第一次进循环
        bar = mBar()
        new_minute_flag = True
    elif bar.datetime.minute != datetime.minute:
        bar.datetime = bar.datetime.replace(second=0, microsecond=0)  # 将秒和微秒设为0
        m_bar_list.append(bar)
        # 开启新的一个分钟bar线
        bar = mBar()
        new_minute_flag = True


    if new_minute_flag:
        bar.open, bar.high, bar.low = last['lastprice'], last['lastprice'], last['lastprice']
    else:
        bar.high, bar.low = max(bar.high, last['lastprice']), min(bar.low, last['lastprice'])

    bar.close = last['lastprice']
    bar.datetime = datetime

# 根据分钟线进行匹配
pk_df = pd.DataFrame(data=[[bar.datetime for bar in m_bar_list],
                           [bar.close for bar in m_bar_list],
                           [bar.open for bar in m_bar_list],
                           [bar.high for bar in m_bar_list],
                           [bar.low for bar in m_bar_list]],
                     index = ['datetime', 'close', 'open','high', 'low'])
pk_df = pd.DataFrame(pk_df.values.T,columns= pk_df.index)
pd.to_datetime(pk_df['datetime'],unit = 's')
pk_df['YearMonthDayMinute'] = pk_df['datetime'].map(lambda x: x.strftime('%Y-%m-%d %H:%M'))
del pk_df['datetime']
# print(pk_df.head(10))
data['YearMonthDayMinute'] = data.index.map(lambda x: x.strftime('%Y-%m-%d %H:%M'))
# for i in range(data.shape[0]):
#     for j in range(pk_df.shape[0]):
#         if pk_df['YearMonthDayMinute'].iloc[j] == data[['YearMonthDayMinute'].iloc[i]]:
#             data['high'].iloc[i] = pk_df['high'].iloc[j]
# print(data.head())
data['datesave'] = data.index
new_df = pd.DataFrame(data.values,columns=data.columns)
new_df = pd.merge(new_df,pk_df,on = 'YearMonthDayMinute',how='outer')   # 这里连接方式要注意整理 这里存在一个nan值没有解决
# data = pd.merge(data,pk_df,on = 'YearMonthDayMinute',how='outer',left_index=True)
new_df = new_df.set_index('datesave')
del new_df['YearMonthDayMinute']
del new_df['shortnm']
data = new_df
# print(data.info())
data = ((data.diff() + 0.0001) / (data + 0.0001)).dropna()


# -----------------------------因子构建----------------------------#
final_DF = pd.DataFrame()
# 因子1 买卖压力
def pressure_Factor(close, askprice, bidprice, askvolume, bidvolume):
    def weight(close, price):
        numerator = price.apply(lambda x: close / (x - close))
        denominator = numerator.sum(1)
        return numerator.apply(lambda x: x / denominator)

    weight_ask = weight(close, askprice)
    weight_ask[~np.isfinite(weight_ask)] = 0.2
    weight_bid = weight(close, bidprice)
    weight_bid[~np.isfinite(weight_bid)] = 0.2
    pressure_ask = (askvolume.values * weight_ask.values).sum(1)
    pressure_bid = (bidvolume.values * weight_bid.values).sum(1)
    return pd.Series(expit(pressure_bid / pressure_ask), index=close.index)

temp = pressure_Factor(close,askprice,bidprice,askvolume,bidvolume)
final_DF['pressure'] = temp
#
# # 因子2 加权价格WP
# # WP1因子
# def weighted_middle_price(a1, b1, v_a1, v_b1):
#     return (b1 * v_b1 + a1 * v_a1) / (v_b1 + v_a1)
# # midprc 因子
# def mid_price(a1, b1):
#     return (a1 + b1) / 2
# temp = weighted_middle_price(a1, b1, v_a1, v_b1)
final_DF['weighted_price'] = data['weighted_midprice']
#
# # 3.挂单量的买卖不均衡
def volume_gap(askvolume,bidvolume):
    """
    统一返回一个0，1，2，3，4 时间节点的dataframe
    """
    time_index = askvolume.index
    temp = (askvolume.values - bidvolume.values)/ (askvolume.values + bidvolume.values)
    return pd.DataFrame(temp,index = time_index)
temp = volume_gap(askvolume,bidvolume)
final_DF[['volume_gap0','volume_gap1','volume_gap2','volume_gap3','volume_gap4']] = temp
# # print(final_DF.head())
# # print(final_DF.shape)
# # print(final_DF.info())
#
# # 4.挂单增量的买卖不平衡（买减卖)
def newcome_buy_vol(cols,bidvolume,bidprice):
    ret_df = pd.DataFrame()
    for col in cols:
        newcome_vol = []
        for i in range(1,bidvolume.shape[0]):
            if bidprice['bidprice'+col].iloc[i] < bidprice['bidprice' + col].iloc[i-1]:
                newcome_vol.append(0)
            elif bidprice['bidprice' + col].iloc[i] == bidprice['bidprice' + col].iloc[i-1]:
                newcome_vol.append(bidvolume['bidvolume' + col].iloc[i] - bidvolume['bidvolume' + col].iloc[i-1])
            else:
                newcome_vol.append(bidvolume['bidvolume' + col].iloc[i])
        ret_df['buy_col' + col] = pd.Series(newcome_vol)
    return ret_df.set_index(bidvolume.index[1:])
#
def newcome_sell_vol(cols,askvolume,askprice):
    ret_df = pd.DataFrame()
    for col in cols:
        newcome_vol = []
        for i in range(1, askvolume.shape[0]):
            if askprice['askprice' + col].iloc[i] < askprice['askprice' + col].iloc[i - 1]:
                newcome_vol.append(0)
            elif askprice['askprice' + col].iloc[i] == askprice['askprice' + col].iloc[i - 1]:
                newcome_vol.append(askvolume['askvolume' + col].iloc[i] - askvolume['askvolume' + col].iloc[i - 1])
            else:
                newcome_vol.append(askvolume['askvolume' + col].iloc[i])
        ret_df['buy_col' + col] = pd.Series(newcome_vol)
    return ret_df.set_index(askvolume.index[1:])
#
def VOI(col_buy,col_sell,askvolume,askprice,bidvolume,bidprice):
    return newcome_buy_vol(['1','2','3','4','5'],bidvolume,bidprice) - newcome_sell_vol(['1','2','3','4','5'],askvolume,askprice)
temp = VOI(['1','2','3','4','5'],['1','2','3','4','5'],askvolume,askprice,bidvolume,bidprice)
final_DF = pd.concat([final_DF,temp],axis = 1,join= 'inner')

# 7.趋势因子
def trend_strength(time_window,midprice_series):
    trend_strength = []
    for i in range(time_window,midprice_series.shape[0]):
        window_series = midprice_series.iloc[i-time_window:i]
#         print(window_series.shape[0])
        delta = window_series.diff()
        trend_strength.append(delta.sum()/abs(delta+0.01).sum())
    return pd.Series(trend_strength,index = midprice_series.index[time_window:])
temp = mid_price(askprice['askprice1'],bidprice['bidprice1'])
temp = trend_strength(10,temp)
final_DF['trend_strength'] = temp
# # print(final_DF.info())
# # print(final_DF.head())
# # print(temp.head())
#
#
#
#
# # --------------------------------国泰长期因子------------------------------#
# # 窗口均值因子
def time_window_mean(data,col,window_list):
    """
    col:  字符串
    window_list 国泰长线因子选择了4个时间窗口
    """
    return ((data['close'].rolling(2).mean() +
            data['close'].rolling(4).mean() +
            data['close'].rolling(6).mean() +
            data['close'].rolling(12).mean())/data['close']/4)

temp = time_window_mean(data,'close',[2,4,6,12])
final_DF['long_term1'] = temp
# print(temp.shape)
# print(final_DF.shape)
# final_DF = pd.concat([final_DF,temp],axis=1,join = 'outer')
# final_DF['long_term1'] = temp
# print(final_DF.info())
# # print(final_DF.head())

# ---------------长线因子------------------------#
# # 长线2-4 因子
def factor2(data,window):
    return pd.Series(((data['high'] + data['low'] + data['close'])/3 -
             ((data['high'] + data['low'] + data['close'])/3).rolling(window).mean()/
             (abs(close -(data['high'] + data['low'] + data['close'])/3)).rolling(window).mean()/0.015).dropna())

window_list = [4,6,12]
for i in window_list:
    temp = factor2(data,i)
    final_DF['factor2_' + str(i)] = temp

def factor4(data,window_paras):
    return ((data['close'] - data['low'].rolling(window_paras[0]).min()) /
    data['high'].rolling(window_paras[0]).max() - data['low'].rolling(window_paras[0]).min()*100).rolling(window_paras[1]).mean().dropna()
window_list = [[4,1],[6,3]]
for i in window_list:
    temp = factor4(data,i)
    final_DF['factor4_' + str(i)] = temp

def factor5(data,window_paras):
    part1 = data['close'] - data['low'].rolling(window_paras[0]).min()
    part2 = (data['high'].rolling(window_paras[0]).max() - data['low'].rolling(window_paras[0]).min()) * 100
    res = 3 * (part1/part2).rolling(window_paras[1]).mean() - (2 * (part1/part2).rolling(1).mean().rolling(1).mean())
    return res.dropna()
window_list = [[4,1],[6,3]]
for i in window_list:
    temp = factor5(data,i)
    final_DF['factor5_' + str(i)] = temp

def factor11(data,window):
    difflist = []
    for i in range(window,data.shape[0]):
        difflist.append(data['close'][i] - data['close'][i-window])
    difflist = pd.Series(difflist,index = data.index[window:])
    res = difflist.values / data['close'][window:].values
    return pd.Series(res,index = data.index[window:])
window_list = [5,10]
for i in window_list:
    temp = factor11(data,i)
    final_DF['factor11_' + str(i)] = temp


def factor12(data,window):
    return data['close'] / data['close'].rolling(window).mean().dropna()
window_list = [2,6,12,24,48,72]
for i in window_list:
    temp = factor12(data,i)
    final_DF['factor12_'+str(i)] = temp
#
#
#-------------不必要的傻逼函数，之前自己写了就写了吧------------------------#
def delay(series,window):
    ret_list = []
    for i in range(window,series.shape[0]):
        ret_list.append(series[i-window])
    return pd.Series(ret_list,index = series.index[window:]).dropna()
def ema(series,N):
    alpha = 2/(N+1)
    data = np.zeros(series.shape[0])
    for i in range((series.shape[0])):
        data[i] = series[i] if i==0 else alpha*series[i]+(1-alpha)*data[i-1]  #从首开始循环
    return pd.Series(data,index = series.index).dropna()
def sma(series,n,m):
    series = series.dropna()
    data = np.zeros(series.shape[0])
    for i in range(series.shape[0]):
        data[i] = series[i] if i == 0 else m * series[i] + (n-m) * data[i-1]
    return pd.Series(data,index = series.index).dropna()
# -----------------------------傻逼结束---------------------------#

def factor13(series,N_list):
    last_part = ema(ema(series,N_list[2]) - ema(series,N_list[3]),N_list[4])
    return (2 * (ema(series,N_list[0]) - ema(series,N_list[1]) - last_part)).dropna()
window_list = [[1,3,1,3,2],[2,6,2,6,4]]
for i in window_list:
    temp = factor13(data['close'],i)
    final_DF['factor13_'+str(i)] = temp
#
def factor14(series,N_list):
    return (ema(series,N_list[0]) - ema(series,N_list[1])).dropna()
window_list = [[1,5],[5,10]]
for i in window_list:
    temp = factor14(data['close'],i)
    final_DF['factor14_'+str(i)] = temp
def factor15(series,N_list):
    return (ema(ema(series,N_list[0]) - ema(series,N_list[1]),N_list[2])).dropna()
window_list = [[1,5,3],[5,10,7]]
for i in window_list:
    temp = factor15(data['close'],i)
    final_DF['factor15_'+str(i)] = temp
# def factor16(series,paras):
#     up_part = sma(series - delay(series,paras[0]),paras[1],paras[2])
#     down_part = sma(abs(series - delay(series,paras[0])),paras[1],paras[2])
#     return (up_part / (down_part+0.01) * 100).dropna()
# window_list = [[1,2,1],[1,6,1],[1,12,1]]
# for i in window_list:
#     temp = factor16(data['close'],i)
#     final_DF['factor16_'+str(i)] = temp
def factor17(series_list,paras):
    """
    series_list 长度为3
    paras 长度为4
    """
    return ((delay(series_list[0],paras[0]) + delay(series_list[1],paras[1]) + 2 * delay(series_list[2],paras[2])) / 4).dropna()
# window_list = [[1,1,1]]
# for i in window_list:
#     temp = factor17(data['close'],i)
#     final_DF['factor17_'+str(i)] = temp
def factor18(series,paras):
    return (series - delay(series,paras)).dropna()
# temp = factor18(data['close'],6)
# final_DF['factor18_'+str(6)] = temp
def factor19(series,paras):
    return ((series - delay(series,paras[0])) / delay(series,paras[1])).dropna()
# temp = factor19(data['close'],[1,1])
# final_DF['factor19_'+str(1)] = temp
def factor20(series,paras):
    return (series.rolling(paras).mean()).dropna()
# temp = factor20(data['volume'],20)
# final_DF['factor20_'+str(20)] = temp
def factor21(series,paras):
    return series.rolling(paras).std().dropna()
# temp = factor21(data['volume'],21)
# final_DF['factor21_'+str(21)] = temp
def factor22(series,paras):
    return ((-1 * series) / series.rolling(paras).mean()).dropna()
# temp = factor22(data['volume'],20)
# final_DF['factor22_'+str(20)] = temp
def factor23(series_list):
    return ((series_list[0] + series_list[1] + series_list[2]) / 3 * series_list[3]).dropna()
temp = factor23([data['close'],data['high'],data['low'],data['volume']])
final_DF['factor23'] = temp
def factor24(series_list,paras):
    return ((series_list[0] - delay(series_list[0],paras)) / delay(series_list[0],paras) * series_list[1]).dropna()
temp = factor24([data['close'],data['volume']],25)
final_DF['factor24'] = temp
def factor25(series_list,paras):
    up = (2 * series_list[0] - series_list[1] + series_list[2]) * series_list[3]
    down = (series_list[2] - series_list[1])
    final = up / down
    return (final.rolling(paras).sum()).dropna()
window_list = [5,20]
for i in window_list:
    temp = factor25([data['close'], data['low'], data['high'], data['volume']], i)
    final_DF['factor25_'+str(i)] = temp

def factor26(series_list,paras):
    """
    paras:包含delay的参数
    """
    first_part = (series_list[0] + series_list[1]) / 2 - delay(series_list[0],paras[0]) + delay(series_list[1],paras[0]) / 2
    snd_part = (series_list[0] - series_list[1]) / series_list[2]
    # 这里会出现由于index 不一致导致的nan值
    res = (first_part * snd_part).dropna()
    res = sma(res,paras[1],paras[2])
    return res.dropna()
temp = factor26([data['high'],data['low'],data['volume']],[1,7,2])
final_DF['factor26'] = temp
final_DF = final_DF.dropna()
print(final_DF.info())
# print(final_DF[final_DF.isnull()].head(50))
# print(final_DF.head(10))
final_DF.to_csv('./dataSet/' + origin_data + '.csv')