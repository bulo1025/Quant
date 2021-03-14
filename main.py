from config import *

for i in data_list:
    save_dir = '/home/liushaozhe/saved_models'
    file = Path(save_dir + '/' + str(i.split('.')[0]))
    if file.exists():
        pass
    else:
        os.mkdir(file)
    save_dir = os.path.join(save_dir,str(i.split('.')[0]))
    data = data_loader.load_data_with_featureselect(data_origin_dir + i,True,num_of_feature)
    # with open(save_dir + '/' + "factor_corrCheck.txt", "w") as f:
    #     f.write("The factor corr is %f" % cor)
    # data = data.drop('mid_price', axis=1)
    train_seq = data.iloc[:int(TRAIN_TEST_RATIO * data.shape[0]), :].values
    train_seq_idx = data.iloc[:int(TRAIN_TEST_RATIO * data.shape[0]), :].reset_index().Time
    test_seq = data.iloc[int(TRAIN_TEST_RATIO * data.shape[0]):, :].values
    test_seq_idx = data.iloc[int(TRAIN_TEST_RATIO * data.shape[0]):, :].reset_index().Time
    # MultiInputLSTM = MultiInputModels(train_seq, test_seq, sw_width, batch_size, epochs_num, verbose_set,train_seq_idx,test_seq_idx)
    # MultiInputLSTM.split_sequence_multi_input()
    # MultiInputLSTM.vanilla_lstm(save_dir,model_name)
    # MultiInputLSTM.train_test_predict_tocsv(save_dir,model_name)
    # MultiInputLSTM.load_model_test(save_dir,model_name)
    MultiInputTCN = MultiInputTCNModels(train_seq, test_seq, sw_width, batch_size, epochs_num, verbose_set, train_seq_idx,
                                      test_seq_idx)
    MultiInputTCN.split_sequence_multi_input()
    MultiInputTCN.vanilla_TCN(save_dir, model_name)
    MultiInputTCN.train_test_predict_tocsv(save_dir, model_name)
    MultiInputTCN.load_model_test(save_dir, model_name)
print('end')

