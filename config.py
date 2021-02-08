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
save_dir = '/home/liushaozhe/saved_models_layer1'
data_origin_dir = '/home/liushaozhe/clean_data_save/'   # 构造好的因子数据
# data_list = ['0512880.csv','0510050.csv','0510500.csv','1159995.csv']
data_list = ['0512880.csv']
sw_width = 60   # 三分钟数据
batch_size = 512
epochs_num = 30
verbose_set = 1
TRAIN_TEST_RATIO = 0.8
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"