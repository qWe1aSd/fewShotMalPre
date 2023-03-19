#!/usr/bin/env python
# encoding: utf-8
# @time: 2023/2/11 16:30
# @author: Bai (๑•̀ㅂ•́)و✧
# @file: 20_mlp_training.py
# @software: PyCharm

"""
    drebin/androzoo * part, all features, rf
        - drebin_part (1512, 223)/(195, 223), features all 221, family 105
        - androzoo_part (1472, 223)/(198, 223), features all 221, family 108
        - ['_layer_3']

    loss function:
        - csdn 论文阅读，https://blog.csdn.net/zhaoyin214/article/details/94396243
        - 知乎论文阅读，https://zhuanlan.zhihu.com/p/390653624
        - 原文，http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    path in: 'F:/12.soj_few_shot/9_mlp/*/feature'
             'F:/12.soj_few_shot/9_mlp/*/train'
             'F:/12.soj_few_shot/9_mlp/*/test'
    path out: 'F:/12.soj_few_shot/9_mlp/*/result'

"""

import os
import gc
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Embedding, Conv1D, LSTM, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


# get current time
def get_current_time():
    """
    :return: current time
    """
    current_time = '{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    return current_time


# get top-k feature set
def get_feature_set(path_in, k):
    """
    get top-k feature set
        - 选择 top-k 个 feature
    :param path_in: 特征选择排序后的 feature
    :param k: top k
    :return:
    """
    feature_list = pd.read_csv(path_in)['method name'].tolist()
    feature_list = feature_list[:k]

    return feature_list


# get train and test for the selected features
def get_train_test(path_train_in, path_test_in, select_feature):
    """
    切掉 app name 后，前 n-2 列为 features，第 n-1 列为 label
    在前 n-2 列中选择 select_feature 列
    :param path_train_in:
    :param path_test_in:
    :return:
    """
    data_train = pd.read_csv(path_train_in, encoding='utf-8')
    data_test = pd.read_csv(path_test_in, encoding='utf-8')
    data_x_train, data_y_train = data_train.loc[:, select_feature].values, data_train.iloc[:, -2].values
    data_x_test, data_y_test = data_test.loc[:, select_feature].values, data_test.iloc[:, -2].values

    return data_train, data_test, data_x_train, data_y_train, data_x_test, data_y_test


###############################################################################################


# mlp network
def train_mlp(input_shape, output_shape, x_train, y_train, x_test, y_test, batch_sizes, epochs, patience, path_model):
    """
    mlp network
    :param input_shape:
    :param output_shape:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param batch_sizes:
    :param epochs:
    :param patience:
    :param path_model:
    :return:
    """
    print('in mlp, three layer')
    # network, three layers
    input = Input(shape=input_shape, name='input')
    dense1 = Dense(1024, activation='relu', name='dense1')(input)
    dense2 = Dense(512, activation='relu', name='dense2')(dense1)
    dense3 = Dense(256, activation='relu', name='dense3')(dense2)
    output = Dense(output_shape, activation='softmax')(dense3)
    # calculate distance of dense3_a and dense3_b
    model = Model(inputs=input, outputs=output)  # create model and optimize distance
    model.summary()
    # train
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=['acc'])
    history = model.fit(
        x_train, y_train, validation_data=(x_test, y_test),
        batch_size=batch_sizes, epochs=epochs, verbose=2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience, verbose=2),
                   ModelCheckpoint(filepath=path_model, monitor='val_loss', save_best_only='True', verbose=2)])
    #
    return model, history


# save result and figure
def get_results(log_note, log_shape, log_time, history, path_result_out, path_figure_out):
    """
    save notes, times
    figure acc and loss of training
    :param log_note:
    :param log_shape:
    :param log_time:
    :param history:
    :param path_result_out:
    :param path_figure_out:
    :return:
    """
    # result
    with open(path_result_out, encoding='utf-8', mode='a') as f:
        f.write(log_note)
        f.write(log_shape)
        f.write(log_time)
        f.write('**********************************************************************\n')

    # figure of training
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    #
    axs[0].plot(history.history["loss"], label="train_loss")
    axs[0].plot(history.history["val_loss"], label="val_loss")
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(range(len(history.history["loss"])))
    axs[0].set_xticklabels(range(1, len(history.history["loss"]) + 1), rotation=90)
    axs[0].legend()
    #
    axs[1].plot(history.history["acc"], label="train_acc")
    axs[1].plot(history.history["val_acc"], label="val_acc")
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(range(len(history.history["acc"])))
    axs[1].set_xticklabels(range(1, len(history.history["acc"]) + 1), rotation=90)
    axs[1].legend()
    #
    plt.tight_layout()
    plt.savefig(path_figure_out, dpi=1200)
    # plt.show()


if __name__ == '__main__':
    ############################## drebin, part, all feature, rf ##############################
    # # drebin_part (1512, 223)/(195, 223), features all 221
    # path_in = 'F:/12.soj_few_shot/9_mlp/drebin_part/'
    # path_feature_in = os.path.join(path_in, 'feature/5_drebin_MI.csv')
    # paths_train_in = [os.path.join(path_in, 'train/train_part_{}.csv'.format(_)) for _ in range(10)]
    # paths_test_in = [os.path.join(path_in, 'test/test_part_{}.csv'.format(_)) for _ in range(10)]
    # paths_model_out = [os.path.join(path_in, 'result/part_model_{}.h5'.format(_)) for _ in range(10)]
    # paths_result_out = [os.path.join(path_in, 'result/part_result_{}.txt'.format(_)) for _ in range(10)]
    # paths_figure_out = [os.path.join(path_in, 'result/part_figure_{}.pdf'.format(_)) for _ in range(10)]
    #
    # # para
    # # name_data, name_model, name_scale = ['drebin', 'androzoo'], ['mlp'], ['all', 'part']  # log
    # # k_max, k_skip = [221, 146, 133], 10  # each 10 feature training a model
    # # num_classes, num_features = [105, 108], [(221,), (146,), (133,)]  # drebin/androzoo, all/hyperlink/similarity
    # name_data, name_model, name_scale = 'drebin', 'mlp', 'part'
    # k_max, k_skip = 221, 10  # each 10 feature training a model
    # num_classes, num_features = 105, (221,)  # drebin, all
    # # epoch, batch_size, patience = 2, 128, 1  # 100, 64, 3
    # epoch, batch_size, patience = 100, 128, 3  # 100, 128, 3
    # # metrics
    # metrics_label = ['accuracy', 'precision', 'recall', 'f1score', 'matthews correlation coefficient']
    ############################## drebin, part, all feature, rf ##############################

    ############################## androzoo, part, all feature, rf ##############################
    # androzoo_part (1472, 223)/(198, 223), features all 221
    path_in = 'F:/12.soj_few_shot/9_mlp/androzoo_part/'
    path_feature_in = os.path.join(path_in, 'feature/5_androzoo_MI.csv')
    paths_train_in = [os.path.join(path_in, 'train/train_part_{}.csv'.format(_)) for _ in range(10)]
    paths_test_in = [os.path.join(path_in, 'test/test_part_{}.csv'.format(_)) for _ in range(10)]
    paths_model_out = [os.path.join(path_in, 'result/part_model_{}.h5'.format(_)) for _ in range(10)]
    paths_result_out = [os.path.join(path_in, 'result/part_result_{}.txt'.format(_)) for _ in range(10)]
    paths_figure_out = [os.path.join(path_in, 'result/part_figure_{}.pdf'.format(_)) for _ in range(10)]

    # para
    # name_data, name_model, name_scale = ['drebin', 'androzoo'], ['mlp'], ['all', 'part']  # log
    # k_max, k_skip = [221, 146, 133], 10  # each 10 feature training a model
    # num_classes, num_features = [105, 108], [(221,), (146,), (133,)]  # drebin/androzoo, all/hyperlink/similarity
    name_data, name_model, name_scale = 'androzoo', 'mlp', 'part'
    k_max, k_skip = 221, 10  # each 10 feature training a model
    num_classes, num_features = 108, (221,)  # androzoo, all
    # epoch, batch_size, patience = 2, 128, 1  # 100, 64, 3
    epoch, batch_size, patience = 100, 128, 3  # 100, 128, 3
    # metrics
    metrics_label = ['accuracy', 'precision', 'recall', 'f1score', 'matthews correlation coefficient']
    ############################## androzoo, part, all feature, rf ##############################

    ############################## train, save ##############################
    for index, (path_train_in, path_test_in, path_model_out, path_result_out, path_figure_out) in enumerate(
            zip(paths_train_in, paths_test_in, paths_model_out, paths_result_out, paths_figure_out)):

        # 3 layers
        print('************************ Fold {}, {} ************************'.format(index, get_current_time()))
        print('Choose {} features by mutual_info...'.format(k_max))
        feature_set = get_feature_set(path_in=path_feature_in, k=k_max)
        train, test, x_train, y_train, x_test, y_test = get_train_test(path_train_in, path_test_in, feature_set)
        y_train_nn, y_test_nn = to_categorical(y_train), to_categorical(y_test)
        # y_train_, y_test_ = np.argmax(y_train_nn, axis=-1), np.argmax(y_test_nn, axis=-1)  # one hot to number
        print(x_train.shape, y_train.shape, len(set(y_train)))

        print('Train mlp network and save model...')
        time1 = datetime.datetime.now()
        model, history = train_mlp(num_features, num_classes, x_train, y_train_nn, x_test, y_test_nn,
                                   batch_size, epoch, patience, path_model_out)
        time2 = datetime.datetime.now()

        print('Save results when training mlp network...')
        log_notes = 'Notes: {}, {}, {}, {}\n'.format(name_data, name_model, name_scale, get_current_time())
        log_shapes = 'Shapes: training {}, '.format(x_train.shape)
        log_shapes += 'test {}\n'.format(x_test.shape)
        log_times = 'Times: training {} in {} seconds\n'.format(name_model, (time2 - time1).seconds)
        get_results(log_notes, log_shapes, log_times, history, path_result_out, path_figure_out)

        # release ram and clean tf static graph
        print('Clear memory...')
        del model
        del x_train, y_train, x_test, y_test
        gc.collect()
