from preprocessing import data_loader
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from model import LSTM_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
import pandas as pd


data_target = '510050.XSHG.csv'
# data_target = 'my.csv'
# time_window = 60
# future_target = 40
features_num = 35
TRAIN_TEST_RATIO = 0.8

data = pd.read_csv('/home/liushaozhe/dataSet/' + data_target)
print(data.shape)
data = data_loader.data_transfer(data)
print(data.shape)
# data = data_loader.stock_data_transfer(data)
# print(data.shape)
# features,target = data_loader.input_data_generator(data,time_window)
# features = np.reshape(features, (features.shape[0],features.shape[1],features_num))
# features = features.reshape(-1,time_window,features_num)
# print(features[0:5])
train_seq = data.iloc[:int(TRAIN_TEST_RATIO * data.shape[0]),:].values
test_seq = data.iloc[int(TRAIN_TEST_RATIO * data.shape[0]):,:].values


class MultiInputModels:
    '''
    时间序列LSTM模型
    '''

    def __init__(self, train_seq, test_seq, sw_width, batch_size, epochs_num, verbose_set):
        '''
        初始化变量和参数
        '''
        self.train_seq = train_seq
        self.test_seq = test_seq
        self.sw_width = sw_width
        self.batch_size = batch_size
        self.epochs_num = epochs_num
        self.verbose_set = verbose_set

        self.train_X, self.train_y = [], []
        self.test_X, self.test_y = [], []

    def split_sequence_multi_input(self):
        '''
        该函数实现多输入序列数据的样本划分
        '''
        # ------- 训练输入数据的样本划分------------#
        scaler = StandardScaler()  # 标准化转换
        for i in range(len(self.train_seq)):
            # 找到最后一个元素的索引，因为for循环中i从1开始，切片索引从0开始，切片区间前闭后开，所以不用减去1；
            end_index = i + self.sw_width
            # 如果最后一个滑动窗口中的最后一个元素的索引大于序列中最后一个元素的索引则丢弃该样本；
            # 这里len(self.sequence)没有减去1的原因是：保证最后一个元素的索引恰好等于序列数据索引时，能够截取到样本；
            if end_index > len(self.train_seq):
                break
            # 实现以滑动步长为1（因为是for循环），窗口宽度为self.sw_width的滑动步长取值；
            # [i:end_index, :-1] 截取第i行到第end_index-1行、除最后一列之外的列的数据；
            # [end_index-1, -1] 截取第end_index-1行、最后一列的单个数据，其实是在截取期望预测值y；
            seq_x, seq_y = self.train_seq[i:end_index, :-1], self.train_seq[end_index - 1, -1]
            scaler_data = scaler.fit(seq_x)
            seq_x = scaler.transform(seq_x)
            self.train_X.append(seq_x)
            self.train_y.append(seq_y)
        self.train_X, self.train_y = np.array(self.train_X), np.array(self.train_y)
        self.features = self.train_X.shape[2]
        # ------- 测试输入数据的样本划分------------#
        for i in range(len(self.test_seq)):
            # 找到最后一个元素的索引，因为for循环中i从1开始，切片索引从0开始，切片区间前闭后开，所以不用减去1；
            end_index = i + self.sw_width
            # 如果最后一个滑动窗口中的最后一个元素的索引大于序列中最后一个元素的索引则丢弃该样本；
            # 这里len(self.sequence)没有减去1的原因是：保证最后一个元素的索引恰好等于序列数据索引时，能够截取到样本；
            if end_index > len(self.test_seq):
                break
            # 实现以滑动步长为1（因为是for循环），窗口宽度为self.sw_width的滑动步长取值；
            # [i:end_index, :-1] 截取第i行到第end_index-1行、除最后一列之外的列的数据；
            # [end_index-1, -1] 截取第end_index-1行、最后一列的单个数据，其实是在截取期望预测值y；
            seq_x, seq_y = self.test_seq[i:end_index, :-1], self.test_seq[end_index - 1, -1]
            scaler_data = scaler.fit(seq_x)
            seq_x = scaler.transform(seq_x)
            self.test_X.append(seq_x)
            self.test_y.append(seq_y)
        self.test_X, self.test_y = np.array(self.test_X), np.array(self.test_y)
        print('train_X.shape:{}, train_y.shape:{}\n'.format(self.train_X.shape, self.train_y.shape))
        print('test_X.shape:{}, test_y.shape:{}\n'.format(self.test_X.shape, self.test_y.shape))
        return self.train_X,self.train_y,self.test_X,self.test_y,self.features


    def vanilla_lstm(self,save_dir):
        model = Sequential()
        # model.add(LSTM(50, activation='relu',
        #                input_shape=(self.sw_width, self.features)))
        model.add(LSTM(20, activation='relu',
                       input_shape=(self.sw_width, self.features),return_sequences=True))
        model.add(LSTM(64, activation='tanh',return_sequences=False))
        # model.add(LSTM(64, activation='relu',return_sequences=False))
        model.add(Dense(1))
        # def trend_accuracy(y_true,y_pred):
        #     train_results = K.sign(y_true*y_pred)
        #     return K.sum(train_results)
        # def r_square(y_true, y_pred):
        #     SSR = K.mean(K.square(y_pred - K.mean(y_true)), axis=-1)
        #     SST = K.mean(K.square(y_true - K.mean(y_true)), axis=-1)
        #     return SSR / SST
        keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer='adam', loss='mse',metrics=['mean_absolute_error'])
        print(model.summary())
        # filepath = "model_{epoch:02d}-{val_loss:.2f}-{val_trend_accuracy:.2f}.hdf5"
        filepath = "model_{epoch:02d}-{val_loss:.2f}-{val_mean_absolute_error:.2f}.hdf5"
        checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_mean_absolute_error', verbose=1,
                                     save_best_only=True)

        history = model.fit(self.train_X, self.train_y,batch_size= 512 ,epochs=self.epochs_num, validation_split=0.2 ,verbose=self.verbose_set,callbacks=[checkpoint])
        # y_pred = model.predict(self.train_X).squeeze(axis = 1)  # 降低维度
        # y_true = self.train_y
        # # print(y_pred)
        # train_results = np.sign(y_pred * y_true)
        # cor = 0
        # for x in train_results:
        #     if x > 0:
        #         cor += 1
        # acc = cor * 1.0 / len(train_results)
        # print("The train acc is %f" % acc)
        y_pred = model.predict(self.test_X).squeeze(axis=1)  # 降低维度
        y_true = self.test_y
        # print(y_pred)
        test_results = np.sign(y_pred * y_true)
        cor = 0
        for x in test_results:
            if x > 0:
                cor += 1
        acc = cor * 1.0 / len(test_results)
        print("The test acc is %f" % acc)
        # ------------------save_model-----------------#
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            # 再存一次
        model_path = os.path.join(save_dir, model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)
        # --------------------plt loss-------------------#
        plt.figure(figsize=(8, 8), dpi=200)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

    def load_model_test(self,save_dir,model_name,train_flag = 0):
        model = keras.models.load_model(save_dir +'/'+model_name)
        if train_flag:
            model.fit(self.train_X,self.train_y)
        else:
            y_pred = model.predict(self.test_X).squeeze(axis=1)  # 降低维度
            y_true = self.test_y
            target_DF = pd.DataFrame()
            target_DF['y_pred'] = y_pred
            target_DF['y_true'] = y_true
            # target_DF = target_DF[(target_DF['y_pred'] > 0.0015) & (target_DF['y_pred'] < 0.015)]
            # print(len(y_true[y_true > 0]))
            # print(len(y_true[y_true < 0]))
            # print(len(y_pred[y_pred > 0]))
            # print(len(y_pred[y_pred < 0]))
            plt.scatter(target_DF['y_pred'], target_DF['y_true'], s=30)
            plt.title('y_pred VS y_true', fontsize=24)
            plt.xlabel('y_pred', fontsize=14)
            plt.ylabel('y_true', fontsize=14)
            plt.axis([-0.001, 0.002, -0.001,0.002])
            plt.show()
            test_results = np.sign(y_pred * y_true)
            cor = 0
            for x in test_results:
                if x > 0:
                    cor += 1
            acc = cor * 1.0 / len(test_results)
            print("The test acc is %f" % acc)






sw_width = 60
batch_size = 512
epochs_num = 50
verbose_set = 1
TRAIN_RATIO = 0.7
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_ETF_trained_model.h5'
MultiInputLSTM = MultiInputModels(train_seq,test_seq,sw_width,batch_size,epochs_num,verbose_set)
MultiInputLSTM.split_sequence_multi_input()
# MultiInputLSTM.vanilla_lstm(save_dir)
# MultiInputLSTM.load_model_test(save_dir,model_name)


