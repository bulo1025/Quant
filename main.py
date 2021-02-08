from config import *

for i in data_list:
    save_dir = '/home/liushaozhe/saved_models_layer1'
    file = Path(save_dir + '/' + str(i.split('.')[0]))
    if file.exists():
        pass
    else:
        os.mkdir(file)
    save_dir = os.path.join(save_dir,str(i.split('.')[0]))
    data = data_loader.load_data_with_featureselect(data_origin_dir + i,True,20)
    data = data.drop('weighted_price', axis=1)
    train_seq = data.iloc[:int(TRAIN_TEST_RATIO * data.shape[0]), :].values
    test_seq = data.iloc[int(TRAIN_TEST_RATIO * data.shape[0]):, :].values
    model_name = 'keras_ETF_trained_model.h5'
    MultiInputLSTM = MultiInputModels(train_seq, test_seq, sw_width, batch_size, epochs_num, verbose_set)
    MultiInputLSTM.split_sequence_multi_input()
    MultiInputLSTM.vanilla_lstm(save_dir,model_name)
print('end')

