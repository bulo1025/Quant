import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
import pandas as pd
from preprocessing import data_loader
from model.LSTM_model import *
from model.TCN import *
save_dir = '/home/liushaozhe/saved_models'   # 工作目录
save_dir = '/home/liushaozhe/paper_saved_models'   # 论文目录
model_name = 'keras_ETF_trained_model.h5'
data_origin_dir = '/home/liushaozhe/Clean_data/'   # 构造好的因子数据
# data_list = ['0512880.csv','0510050.csv','0510500.csv','1159995.csv']
data_list = ['0510050.csv']
sw_width = 3   # 三分钟数据
batch_size = 1024
epochs_num = 25
verbose_set = 1
TRAIN_TEST_RATIO = 0.8
num_of_feature = 42
# num_of_feature = 130
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"