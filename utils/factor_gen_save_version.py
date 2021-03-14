import numpy as np
import pandas as pd
import os
import datetime
from scipy.special import expit
from tqdm import tqdm
# origin_data = '510050.XSHG'
dir = '../dataSet/'
# target_list = ['0512880','0510050','0510500','1150040','1159995']
# target_list = ['0510500','0510050','0512880','1159995']
target_list = ['0512880']
ref_dict = {'0510050':['FIH2103_Hist','0000016_Hist'],
            '0512880':['FIH2103_Hist','1399975_Hist'],
            '0510500':['FIC2103_Hist','0000905_Hist'],
            '1159995':['FIC2103_Hist','1980017_Hist'],
            '1150040':['FIC2103_Hist','1399673_Hist']}
for target in tqdm(target_list):
    print('start loading ' + target)
    def load_Origin_Data(dir):
        """
        加载原始数据
        :param dir: 股票数据的相对位置   例如：'./dataSet/510050.XSHG/'
        :return:    构造好的dataframe
        """
        if target in ref_dict:
            future = ref_dict[target][0]
            index = ref_dict[target][1]
        data_futures = pd.read_csv(dir + future + '.csv')
        data_futures['Time'] = pd.to_datetime(data_futures.Time)
        data_futures.set_index(['Time'], inplace=True)
        data_index = pd.read_csv(dir + index + '.csv')
        data_index['Time'] = pd.to_datetime(data_index.Time)
        data_index.set_index(['Time'], inplace=True)
        data_futures = data_futures[~data_futures.index.duplicated(keep='first')]
        data_index = data_index[~data_index.index.duplicated(keep='first')]
        data_futures = data_futures.iloc[1:, :]
        csv_ls = []
        for root, dirs, files in os.walk(dir):    # 可以不用写绝对路径只是为了方便
            for filename in files:
                if filename[:7] == target:
                    csv_ls.append(filename)
        csv_ls = sorted(csv_ls)
        # print(csv_ls)
        for i in csv_ls:    # 这里只有一个元素啊，懒得改了，到时候多个文件再来改
            data = pd.read_csv(dir + i)
            data['Time'] = pd.to_datetime(data.Time)   # 昂妻案
            data.set_index(['Time'], inplace=True)
            # data = pd.read_csv(dir + i)
        return data.drop(['exchangeCD','ticker'],axis = 1).astype('float'),data_futures,data_index

    data,data_futures,data_index = load_Origin_Data(dir)
    data['Value'] = data['value'].diff()
    data['Value'] = data['Value'].fillna(method='ffill')
    data['pt'] = data['Value'] / data['Volume']
    data['pt'] = data['pt'].fillna(method='ffill')
    data = data.dropna()
    print(data.shape)
    # 变量转换函数
    def variable_shift(data):
        open = data.Open
        close = data.Close
        high = data.High
        low = data.Low
        a5 = data.askPrice5
        a4 = data.askPrice4
        a3 = data.askPrice3
        a2 = data.askPrice2
        a1 = data.askPrice1
        b1 = data.bidPrice1
        b2 = data.bidPrice2
        b3 = data.bidPrice3
        b4 = data.bidPrice4
        b5 = data.bidPrice5
        v_a5 = data.askVolume5
        v_a4 = data.askVolume4
        v_a3 = data.askVolume3
        v_a2 = data.askVolume2
        v_a1 = data.askVolume1
        v_b1 = data.bidVolume1
        v_b2 = data.bidVolume2
        v_b3 = data.bidVolume3
        v_b4 = data.bidVolume4
        v_b5 = data.bidVolume5
        askprice = pd.concat([a5,a4,a3,a2,a1], axis=1)   # 卖家喊出来的价格
        bidprice = pd.concat([b1,b2,b3,b4,b5], axis=1)   # 买家提供的价格
        askvolume = pd.concat([v_a5,v_a4,v_a3,v_a2,v_a1], axis=1)
        bidvolume = pd.concat([v_b1,v_b2,v_b3,v_b4,v_b5], axis=1)
        return open,close,high,low,askprice,bidprice,askvolume,bidvolume,a1,a2,a3,a4,a5,b1,b2,b3,b4,b5,v_a1,v_a2,v_a3,v_a4,v_a5,v_b1,v_b2,v_b3,v_b4,v_b5

    open,close,high,low,askprice,bidprice,askvolume,bidvolume,a1,a2,a3,a4,a5,b1,b2,b3,b4,b5,v_a1,v_a2,v_a3,v_a4,v_a5,v_b1,v_b2,v_b3,v_b4,v_b5 = variable_shift(data)   # 前面几个因子用到了构建因子

    # ----------------------------------因子构建部分--------------------------------#
    # # 因子2 加权价格WP
    # # WP1因子
    def weighted_middle_price(a1, b1, v_a1, v_b1):
        return (b1 * v_b1 + a1 * v_a1) / (v_b1 + v_a1)
    # midprc 因子d
    def mid_price(a1, b1):
        return (a1 + b1) / 2
    # temp = weighted_middle_price(data['askPrice1'], data['bidPrice1'],data['askVolume1'],data['bidVolume1'])  # 加权价格
    temp = mid_price(data['askPrice1'],data['bidPrice1'])   # 直接的midprice
    data['midprice'] = temp
    bar_data = data['Close'].resample('1T', closed='right', how='ohlc')
    #  rolling
    bar_data_re = bar_data.resample('1S').bfill()[['high','low']]
    data['high_1minbar'] = bar_data_re['high']    # 生成1分钟close price 的high bar
    data['low_1minbar'] = bar_data_re['low']      # 1分钟总close price 的 low bar
    data = data.dropna()
    # -----------------------------因子构建----------------------------#
    print('feature generation start')
    final_DF = pd.DataFrame()
    final_DF['midprice'] = data['midprice']
    # # 因子1 买卖压力
    # def pressure_Factor1(close, askprice, bidprice, askvolume, bidvolume):
    #     def weight(close, price):
    #         numerator = price.apply(lambda x: close / (x - close))
    #         denominator = numerator.sum(1)
    #         return numerator.apply(lambda x: x / denominator)
    #
    #     weight_ask = weight(close, askprice)
    #     weight_ask[~np.isfinite(weight_ask)] = 0.2
    #     weight_bid = weight(close, bidprice)
    #     weight_bid[~np.isfinite(weight_bid)] = 0.2
    #     pressure_ask = (askvolume.values * weight_ask.values).sum(1)
    #     pressure_bid = (bidvolume.values * weight_bid.values).sum(1)
    #     return pd.Series(np.log(pressure_bid / pressure_ask), index=close.index)
    def pressure_Factor(midprice,askprice, bidprice, askvolume, bidvolume):
        numerator_buy = bidvolume.values / (bidprice.sub(midprice,axis = 0).values + 0.000001)
        # 防止无穷大值出现
        numerator_buy[~np.isfinite(numerator_buy)] = 0.2
        denominator_buy = (1 / (bidprice.sub(midprice,axis = 0).values).sum(axis = 1).reshape(-1,1) + 0.000001)
        denominator_buy[~np.isfinite(denominator_buy)] = 0.2
        pressure_buy = (numerator_buy / denominator_buy).sum(axis = 1)
        numerator_sell = askvolume.values / (askprice.sub(midprice, axis=0).values + 0.000001)
        numerator_sell[~np.isfinite(numerator_buy)] = 0.2
        denominator_sell = (1 / (askprice.sub(midprice, axis=0).values).sum(axis=1).reshape(-1, 1) + 0.000001)
        pressure_sell = (numerator_sell / denominator_sell).sum(axis=1)
        numerator_sell[~np.isfinite(numerator_buy)] = 0.2
        return pd.Series(np.log(pressure_buy / (pressure_sell + 0.000001)), index=close.index)
    temp = pressure_Factor(data['midprice'],askprice,bidprice,askvolume,bidvolume)
    final_DF['pressure'] = temp
    #
    # # # 3.挂单量的买卖不均衡
    def volume_gap(askvolume,bidvolume):
        """
        统一返回一个0，1，2，3，4 时间节点的dataframe
        """
        time_index = askvolume.index
        temp = (bidvolume.values - askvolume.values)/ (bidvolume.values + askvolume.values)
        return pd.DataFrame(temp,index = time_index)
    temp = volume_gap(askvolume,bidvolume)
    final_DF[['volume_gap0','volume_gap1','volume_gap2','volume_gap3','volume_gap4']] = temp
    # def price_gap(askprice,bidprice):
    #     #
    #     time_index = askprice.index
    #     HR2 = (bidprice['bidPrice1'] - bidprice['bidPrice2'] - (askprice['askPrice2'] - askprice['askPrice1'])) / (
    #             bidprice['bidPrice1'] - bidprice['bidPrice2'] + (askprice['askPrice2'] - askprice['askPrice1']))
    #     HR3 = (bidprice['bidPrice2'] - bidprice['bidPrice3'] - (askprice['askPrice3'] - askprice['askPrice2'])) / (
    #             bidprice['bidPrice2'] - bidprice['bidPrice3'] + (askprice['askPrice2'] - askprice['askPrice1']))
    #     HR2 = pd.Series(HR2,index= time_index)
    #     HR3 = pd.Series(HR3, index=time_index)
    #     return HR2,HR3   # youwenti
    # HR2,HR3 = price_gap(askprice, bidprice)
    # final_DF['price_gap2'] = HR2
    # final_DF['price_gap3'] = HR3
    # # # 4.挂单增量的买卖不平衡（买减卖)
    tick_price = data['pt']
    bid_ratio = (askprice['askPrice1'] - tick_price) / (askprice['askPrice1'] - bidprice['bidPrice1'])
    bid_vol = data['Volume'] * bid_ratio
    ask_vol = data['Volume'] - bid_vol
    def newcome_buy_vol(cols,bidvolume,bidprice):
        ret_df = pd.DataFrame()
        for col_idx in range(len(cols[:-1])):
            newcome_vol = []
            for i in range(1,bidvolume.shape[0]):
                if bidprice['bidPrice' + cols[col_idx]].iloc[i] < bidprice['bidPrice' + cols[col_idx]].iloc[i-1]:
                    if bidvolume['bidVolume' + cols[col_idx]].iloc[i] <= bidvolume['bidVolume' + cols[col_idx + 1]].iloc[i-1]:
                        newcome_vol.append(0)
                    else:
                        newcome_vol.append(bidvolume['bidVolume' + cols[col_idx]].iloc[i] - bidvolume['bidVolume' + cols[col_idx+1]].iloc[i-1])
                elif bidprice['bidPrice' + cols[col_idx]].iloc[i] == bidprice['bidPrice' + cols[col_idx]].iloc[i-1]:
                    newcome_vol.append(bidvolume['bidVolume' + cols[col_idx]].iloc[i] - bidvolume['bidVolume' + cols[col_idx]].iloc[i-1] + bid_vol.iloc[i])
                else:
                    newcome_vol.append(bidvolume['bidVolume' + cols[col_idx]].iloc[i] + ask_vol.iloc[i])
            ret_df['buy_col' + cols[col_idx]] = pd.Series(newcome_vol)
        return ret_df.set_index(bidvolume.index[1:])

    def newcome_sell_vol(cols,askvolume,askprice):
        ret_df = pd.DataFrame()
        for col_idx in range(len(cols[:-1])):
            newcome_vol = []
            for i in range(1, askvolume.shape[0]):
                if askprice['askPrice' + cols[col_idx]].iloc[i] < askprice['askPrice' + cols[col_idx]].iloc[i - 1]:
                    newcome_vol.append(askvolume['askVolume' + cols[col_idx]].iloc[i] + bid_vol.iloc[i])
                elif askprice['askPrice' + cols[col_idx]].iloc[i] == askprice['askPrice' + cols[col_idx]].iloc[i - 1]:
                    newcome_vol.append(askvolume['askVolume' + cols[col_idx]].iloc[i] - askvolume['askVolume' + cols[col_idx]].iloc[i - 1] + ask_vol.iloc[i])
                else:
                    if askvolume['askVolume' + cols[col_idx]].iloc[i] <= askvolume['askVolume' + cols[col_idx+1]].iloc[i-1]:
                        newcome_vol.append(0)
                    else:
                        newcome_vol.append(askvolume['askVolume' + cols[col_idx]].iloc[i] - askvolume['askVolume' + cols[col_idx+1]].iloc[i-1])
            ret_df['buy_col' + cols[col_idx]] = pd.Series(newcome_vol)
        return ret_df.set_index(askvolume.index[1:])
    #
    def VOI(col_buy,col_sell,askvolume,askprice,bidvolume,bidprice):
        a = newcome_buy_vol(col_buy, bidvolume, bidprice)
        b = newcome_sell_vol(col_sell,askvolume,askprice)
        return newcome_buy_vol(col_buy,bidvolume,bidprice) - newcome_sell_vol(col_sell,askvolume,askprice)
    temp = VOI(['1','2','3','4','5'],['1','2','3','4','5'],askvolume,askprice,bidvolume,bidprice)
    final_DF = pd.concat([final_DF,temp],axis = 1,join= 'inner')

    # 6. 均值回归因子MPB
    def average_trade_price():
        TP = []
        for i in range(data.shape[0]):
            if i == 0:
                TP.append(data['midprice'].iloc[i])
            elif data['Volume'].iloc[i] > 0:
                TP.append(tick_price.iloc[i])
            elif data['Volume'].iloc[i] == 0:
                TP.append(TP[i-1])
        TP = pd.Series(TP,index = data.index)
        return TP
    def MPB(tp_series,midprice):
        return tp_series - midprice
    tp_series = average_trade_price()
    temp = MPB(tp_series,data['midprice'])
    final_DF['MPB'] = temp
    # 7.趋势因子
    def trend_strength(time_window,midprice_series):
        trend_strength = []
        for i in range(time_window,midprice_series.shape[0]):
            window_series = midprice_series.iloc[i-time_window:i]
    #         print(window_series.shape[0])
            delta = window_series.diff()
            trend_strength.append(delta.sum()/(abs(delta).sum() + 0.0000001))
        return pd.Series(trend_strength,index = midprice_series.index[time_window:])
    temp = mid_price(askprice['askPrice1'],bidprice['bidPrice1'])
    temp = trend_strength(60,temp)
    retdf = pd.DataFrame()
    retdf['trend_strength'] = temp
    final_DF = pd.concat([final_DF,retdf],axis = 1,join='inner')
    print('finish half')
    # 2.roll model
    def half_bid_ask_spread(data, timewindow):
        def pt(data):   #    value  value 要做差分
            return data['pt']
        spread_list = []
        sigma_u_list = []
        # for i in tqdm(range(800, data.shape[0]-1)):  # 滑动窗口遍历   i从当前时刻一直到最后一个时刻
        for i in range(timewindow,data.shape[0] - 1):
            print(i)
            timewindowdata_t = data.iloc[i - timewindow:i, :]
            # timewindowdata_t1 = data.iloc[i - timewindow - 1:i - 1, :]
            timewindowdata_t1 = timewindowdata_t.shift(1).dropna()   # 这里用shift后就需要iloc选取
            price_cov = np.cov(pt(timewindowdata_t).diff().dropna().iloc[1:], pt(timewindowdata_t1).diff().dropna())[0][1]
            price_cov = (-price_cov if -price_cov > 0 else 0)
            c = price_cov ** 0.5
            # sigmau = np.var(pt(timewindowdata_t)) + 2 * np.cov(pt(timewindowdata_t).diff().dropna().iloc[1:], pt(timewindowdata_t1).diff().dropna())[0][1]
            spread_list.append(c)
            # sigma_u_list.append(sigmau)
        return pd.Series(spread_list, index=data.index[timewindow:-1])
        # return pd.Series(spread_list,index = data.index[timewindow:]),pd.Series(sigma_u_list,index = data.index[timewindow:])
    # temp,temp2 = half_bid_ask_spread(data,21)
    temp = half_bid_ask_spread(data, 21)
    final_DF['roll_model_c'] = temp
    # final_DF['roll_model_sigma'] = temp2
    # 3.high-low volatility
    def high_low_volatility(data,window):
        hl_ratio = np.log(data['high_1minbar'] / data['low_1minbar'])
        hl_volatility = hl_ratio.rolling(window).sum() / window
        hl_volatility_2 = (hl_ratio ** 2).rolling(window).sum() / window / (4 * np.log(2))
        return hl_volatility.dropna(),hl_volatility_2.dropna()
    window_list = [100,200,300]
    for i in window_list:
        temp1,temp2 = high_low_volatility(data,i)
        final_DF['hl_volatility_' + str(i)] = temp1
        final_DF['hl_volatility2_' + str(i)] = temp2
    final_DF = final_DF.dropna()   # 其实也可以不用写
    # 4.corwin and schultz
    def getGamma(data):
        h2 = data['high_1minbar'].rolling(2).max()
        l2 = data['low_1minbar'].rolling(2).min()
        gamma = np.log(h2.values / l2.values) ** 2
        gamma = pd.Series(gamma, index=h2.index)
        return gamma.dropna()
    def getBeta(data, sl):
        hl = data[['high_1minbar', 'low_1minbar']].values
        hl = np.log(hl[:, 0] / hl[:, 1]) ** 2
        hl = pd.Series(hl, index=data.index)
        beta = hl.rolling(2).sum()  # 这里窗口2为公式给定
        beta = beta.rolling(sl).mean()
        return beta.dropna()
    def getAlpha(beta, gamma):
        """
        两者样本维度可能不一致，到时候需要dropna
        """
        den = 3 - 2 * 2 ** 0.5
        alpha = (2 ** 0.5 - 1) * (beta ** 0.5) / den
        alpha -= (gamma / den) ** 0.5
        alpha[alpha < 0] = 0  # 将负数的alpha置0
        return alpha.dropna()
    def corwinSchultz(data, sl):
        beta = getBeta(data, sl)
        gamma = getGamma(data)
        alpha = getAlpha(beta, gamma)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        return spread
    temp = corwinSchultz(data, 21)
    final_DF['corwinSchultz'] = temp
    # # # --------------------------------国泰长期因子------------------------------#
    print("Start long-term factor generation")
    # # 窗口均值因子
    def time_window_mean(data,col,window_list):
        """
        col:  字符串
        window_list 国泰长线因子选择了4个时间窗口
        """
        return ((data['Close'].rolling(2).mean() +
                data['Close'].rolling(4).mean() +
                data['Close'].rolling(6).mean() +
                data['Close'].rolling(12).mean())/data['Close']/4)

    temp = time_window_mean(data,'Close',[2,4,6,12])
    final_DF['long_term1'] = temp
    final_DF = final_DF.dropna()
    #
    # # ---------------长线因子------------------------#
    # 长线2-4 因子
    # 还会出现-inf
    # def factor2(data,window):
    #     return pd.Series(((data['high_1minbar'] + data['low_1minbar'] + data['Close'])/3 -
    #              ((data['high_1minbar'] + data['low_1minbar'] + data['Close'])/3).rolling(window).mean()/
    #              (abs(close -(data['high_1minbar'] + data['low_1minbar'] + data['Close'])/3)).rolling(window).mean()/0.015).dropna())
    #
    # window_list = [4,6,12]
    # for i in window_list:
    #     temp = factor2(data,i)
    #     final_DF['factor2_' + str(i)] = temp

    def factor4(data,window_paras):
        return ((data['Close'] - data['low_1minbar'].rolling(window_paras[0]).min()) /
        data['high_1minbar'].rolling(window_paras[0]).max() - data['low_1minbar'].rolling(window_paras[0]).min()*100).rolling(window_paras[1]).mean().dropna()
    window_list = [[4,1],[6,3]]
    for i in window_list:
        temp = factor4(data,i)
        final_DF['factor4_' + str(i)] = temp

    def factor5(data,window_paras):
        part1 = data['Close'] - data['low_1minbar'].rolling(window_paras[0]).min()
        part2 = (data['high_1minbar'].rolling(window_paras[0]).max() - data['low_1minbar'].rolling(window_paras[0]).min()) * 100
        res = 3 * (part1/part2).rolling(window_paras[1]).mean() - (2 * (part1/part2).rolling(1).mean().rolling(1).mean())
        return res.dropna()
    window_list = [[4,1],[6,3]]
    for i in window_list:
        temp = factor5(data,i)
        final_DF['factor5_' + str(i)] = temp

    def factor11(data,window):
        difflist = []
        for i in range(window,data.shape[0]):
            difflist.append(data['Close'][i] - data['Close'][i-window])
        difflist = pd.Series(difflist,index = data.index[window:])
        res = difflist.values / data['Close'][window:].values
        return pd.Series(res,index = data.index[window:])
    window_list = [5,10]
    for i in window_list:
        temp = factor11(data,i)
        final_DF['factor11_' + str(i)] = temp

    def factor12(data,window):
        return data['Close'] / data['Close'].rolling(window).mean().dropna()
    window_list = [2,6,12,24,48,72]
    for i in window_list:
        temp = factor12(data,i)
        final_DF['factor12_'+str(i)] = temp
    # #-------------不必要的傻逼函数，之前自己写了就写了吧------------------------#
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
    # # -----------------------------傻逼结束---------------------------#
    def factor13(series,N_list):
        last_part = ema(ema(series,N_list[2]) - ema(series,N_list[3]),N_list[4])
        return (2 * (ema(series,N_list[0]) - ema(series,N_list[1]) - last_part)).dropna()
    window_list = [[1,3,1,3,2],[2,6,2,6,4]]
    for i in window_list:
        temp = factor13(data['Close'],i)
        final_DF['factor13_'+str(i)] = temp

    def factor14(series,N_list):
        return (ema(series,N_list[0]) - ema(series,N_list[1])).dropna()
    window_list = [[1,5],[5,10]]
    for i in window_list:
        temp = factor14(data['Close'],i)
        final_DF['factor14_'+str(i)] = temp
    def factor15(series,N_list):
        return (ema(ema(series,N_list[0]) - ema(series,N_list[1]),N_list[2])).dropna()
    window_list = [[1,5,3],[5,10,7]]
    for i in window_list:
        temp = factor15(data['Close'],i)
        final_DF['factor15_'+str(i)] = temp

    def factor17(series_list):
        """
        series_list 长度为3
        paras 长度为4
        """
        return ((series_list[0].shift(1) + series_list[1].shift(1) + 2 * series_list[2].shift(1)) / 4).dropna()
    series_list = [data['high_1minbar'],data['low_1minbar'],data['Close']]
    temp = factor17(series_list)
    final_DF['factor17_'+str(i)] = temp

    def factor18(series,paras):
        return (series - series.shift(paras)).dropna()
    temp = factor18(data['Close'],6)
    final_DF['factor18_'+str(6)] = temp

    def factor19(series,paras):
        return ((series - delay(series,paras[0])) / delay(series,paras[1])).dropna()
    temp = factor19(data['Close'],[1,1])
    final_DF['factor19_'+str(1)] = temp

    def factor20(series,paras):
        return (series.rolling(paras).mean()).dropna()
    temp = factor20(data['Volume'],20)
    final_DF['factor20_'+str(20)] = temp

    def factor21(series,paras):
        return series.rolling(paras).std().dropna()
    temp = factor21(data['Volume'],21)
    final_DF['factor21_'+str(21)] = temp

    def factor22(series,paras):
        return ((-1 * series) / series.rolling(paras).mean()).dropna()
    temp = factor22(data['Volume'],20)
    final_DF['factor22_'+str(20)] = temp

    def factor23(series_list):
        return ((series_list[0] + series_list[1] + series_list[2]) / 3 * series_list[3]).dropna()
    temp = factor23([data['Close'],data['high_1minbar'],data['low_1minbar'],data['Volume']])
    final_DF['factor23'] = temp

    def factor24(series_list,paras):
        return ((series_list[0] - delay(series_list[0],paras)) / delay(series_list[0],paras) * series_list[1]).dropna()
    temp = factor24([data['Close'],data['Volume']],25)
    final_DF['factor24'] = temp

    def factor25(series_list,paras):
        up = (2 * series_list[0] - series_list[1] + series_list[2]) * series_list[3]
        down = (series_list[2] - series_list[1])
        final = up / down
        return (final.rolling(paras).sum()).dropna()
    window_list = [5,20]
    for i in window_list:
        temp = factor25([data['Close'], data['low_1minbar'], data['high_1minbar'], data['Volume']], i)
        final_DF['factor25_'+str(i)] = temp

    def factor26(series_list,paras):
        first_part = (series_list[0] + series_list[1]) / 2 - series_list[0].shift(paras[0]) + series_list[1].shift(
            paras[0]) / 2
        snd_part = (series_list[0] - series_list[1]) / series_list[2]
        res = (first_part * snd_part).dropna()
        # res = sma(res,paras[1],paras[2])
        res = res.rolling(window=7, min_periods=2).mean()
        return res.dropna()
    # temp = factor26([data['high_1minbar'],data['low_1minbar'],data['Volume']],[1,7,2])
    temp = factor26([data['high_1minbar'],data['low_1minbar'],data['Volume']],[1,7,2])
    final_DF['factor26'] = temp
    # final_DF = final_DF.dropna()
    # final_DF = final_DF.diff().dropna()
    # target序列的单独构建
    # final_DF['target'] = final_DF['midprice'].diff() / final_DF['midprice']
    # final_DF['target'] = final_DF['midprice'].diff()
    final_DF = final_DF.dropna()
    final_DF =final_DF.resample('15s').last().dropna()

#-----------------------------------期货和指数因子----------------------#
    # 生成15s的采样
    future_close = data_futures['Close'].resample('15s').last().dropna()
    index_close = data_index['Close'].resample('15s').last().dropna()
    ETF_close = data['Close'].resample('15s').last().dropna()
    # 计算基差一阶差分
    basis = (future_close - index_close).dropna()
    final_DF['basis_diff'] = basis.diff().dropna()
    # 计算期货-ETF价差 2H- 1day
    C = (future_close.rolling(475).mean() / ETF_close.rolling(475).mean()).dropna()
    final_DF['spread'] = (future_close - C * ETF_close).dropna()
    final_DF['target'] = final_DF['midprice']
    # temp_target = final_DF['target']
    final_DF.drop(labels=['midprice'], axis=1, inplace=True)
    # final_DF.insert(len(final_DF.columns), 'target', temp_target)
    print(final_DF.shape)
    final_DF = final_DF.diff().dropna()
    final_DF.to_csv('../Clean_data/' + target + '.csv')
    print('Finish' + target + 'feature construction')
