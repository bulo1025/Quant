import pandas as pd


def weighted_middle_price(a1, b1, v_a1, v_b1):
    return (b1 * v_b1 + a1 * v_a1) / (v_b1 + v_a1)
data = pd.read_csv('/home/liushaozhe/Clean_data/0510050.csv',index_col='Time')
corr = data.corr()
# print(data.head())