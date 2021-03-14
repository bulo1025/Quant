import pandas as pd
from matplotlib import pyplot as plt
data = pd.read_csv('/home/liushaozhe/dataSet/0510050_Hist.csv')
data['midprice'] = (data['askPrice1'] + data['bidPrice1']) / 2
