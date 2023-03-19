#!/usr/bin/env python
# encoding: utf-8
# @time: 2023/1/20 8:36
# @author: Bai (๑•̀ㅂ•́)و✧
# @file: 4_siamese_training.py
# @software: PyCharm

"""
    drebin/androzoo * part, all features, rf
        - drebin_part (1512, 223)/(195, 223), features all 221, family 105
        - androzoo_part (1472, 223)/(198, 223), features all 221, family 108
        - ['_layer_1', '_layer_2', '_layer_3', '_net_cnn', '_net_lstm']

    loss function:
        - csdn 论文阅读，https://blog.csdn.net/zhaoyin214/article/details/94396243
        - 知乎论文阅读，https://zhuanlan.zhihu.com/p/390653624
        - 原文，http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    path in: 'F:/12.soj_few_shot/4_siamese/*/feature'
             'F:/12.soj_few_shot/4_siamese/*/train'
             'F:/12.soj_few_shot/4_siamese/*/test'
    path out: 'F:/12.soj_few_shot/4_siamese/*/result'

"""

import os
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

from keras import backend as K
from itertools import product, combinations


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


# get pairs
def get_pairs(x, digit_indices):
    """
    Positive and negative pair creation.
    :param x: train data, numpy
    :param digit_indices: family label, list containing array(s)
    :return:
    """
    pairs, labels = [], []  # pairs of sample, label of pairs
    counts = [0, 0]  # count the amount of pos pairs and neg pairs, counts[0] = positive, counts[1] = negative

    # 按 len(family) 升序排列，m1 共有 33 个 family，修改完 code 即可 comment 该行，不再需要 sort
    # digit_indices.sort(key=lambda digit: len(digit))  # 2，2，3，3，……，100
    print(' - App number in each family: ', [len(item) for item in digit_indices])

    # 遍历所有 family，digit_indices 中存储的是各 family 中 sample 的序号（train 中的序号）
    family_number = len(digit_indices)  # 33, m1 250
    for family in range(family_number):
        # print('***************** positive pair ******************')
        # create positive pairs, 无重复组合 + 自身组合, C(len(family), 2) + len(family)
        positive_pair_id = [list(item) for item in combinations(digit_indices[family], 2)]  # 无重复组合
        for item in digit_indices[family]:  # 自身组合
            positive_pair_id.append([item, item])
        # update pairs and labels
        for item in positive_pair_id:
            # print(item[0], ' | ', item[1])
            pairs.append([x[item[0]], x[item[1]]])
            labels.append(1)
            counts[1] = counts[1] + 1
        # print('family and number: ', family, len(digit_indices[family].tolist()), ' | sample: ', digit_indices[family])
        # print('update pos: {}'.format(counts))

        # print('***************** negative pair ******************')
        # create negative pairs, family 与 family + 1: 组成 nagative pairs, len(family) * len(family + 1)
        if family + 1 < family_number:  # 末尾 family 不组成 negative pair
            negative_pair_id = []
            for item in digit_indices[family + 1:]:  # 遍历 family + 1:
                for pair in list(product(digit_indices[family], item)):  # family 与 family + 1: 组成 nagative pairs
                    negative_pair_id.append(list(pair))
            # update pairs and labels
            for item in negative_pair_id:
                # print(item[0], ' | ', item[1])
                pairs.append([x[item[0]], x[item[1]]])
                labels.append(0)
                counts[0] = counts[0] + 1
                # print('family and number: ', family, len(digit_indices[family].tolist()), ' | family_pos: ',
                #       sum([len(family.tolist()) for family in digit_indices[family + 1:]]), ' | sample: ',
                #       digit_indices[family], ' | family_pos sample: ', digit_indices[family + 1:])
                # print('update neg: {}'.format(counts))

    pairs, labels = np.array(pairs), np.array(labels)
    # print(' - Shape of pairs: {}'.format(pairs.shape))
    # print(' - Shape of labels: {}'.format(labels.shape))
    # print(' - Labels 0 (neg): {}, labels 1 (pos): {}'.format(counts[0], counts[1]))
    #
    return pairs, labels


###############################################################################################


# siamese network
def train_siamese(input_shape, x_train, y_train, x_test, y_test, batch_sizes, epochs, patience, path_model, flag):
    """
    siamese network, distance, contrastive loss
    :param input_shape:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param batch_sizes:
    :param epochs:
    :param patience:
    :param path_model:
    :param flag: ['_layer_1', '_layer_2', '_layer_3', '_net_cnn', '_net_lstm']
    :return:
    """

    # for distance
    def euclidean_distance(vects):
        """
        distance of two networks
        :param vects:
        :return:
        """
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    # for distance
    def eucl_dist_output_shape(shapes):
        """
        distance of two networks
        :param shapes:
        :return:
        """
        shape1, shape2 = shapes
        return (shape1[0], 1)

    # for compiling
    def contrastive_loss(y_true, y_pred):
        """
        Contrastive loss from Hadsell-et-al.'06.
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        :param y_true:
        :param y_pred:
        :return:
        """
        margin = 1
        sqaure_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

    # for compiling
    def accuracy(y_true, y_pred):
        """
        Operation on tensor.
        Compute classification accuracy with a fixed threshold on distances.
        :param y_true:
        :param y_pred:
        :return:
        """
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

    if flag == '_layer_3':
        print('in mlp, three layer')
        # network, three layers
        input_a = Input(shape=input_shape, name='input_a')  # re-use the same instance `base_network` and weight
        input_b = Input(shape=input_shape, name='input_b')
        dense1_a = Dense(1024, activation='relu', name='dense1_a')(input_a)
        dense2_a = Dense(512, activation='relu', name='dense2_a')(dense1_a)
        dense3_a = Dense(256, activation='relu', name='dense3_a')(dense2_a)
        dense1_b = Dense(1024, activation='relu', name='dense1_b')(input_b)
        dense2_b = Dense(512, activation='relu', name='dense2_b')(dense1_b)
        dense3_b = Dense(256, activation='relu', name='dense3_b')(dense2_b)
        # calculate distance of dense3_a and dense3_b
        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([dense3_a, dense3_b])
        model = Model(inputs=[input_a, input_b], outputs=distance)  # create model and optimize distance
        # model.summary()
        # train
        model.compile(loss=contrastive_loss, optimizer=Adam(lr=0.0005), metrics=[accuracy])
        history = model.fit(
            [x_train[:, 0], x_train[:, 1]], y_train,
            batch_size=batch_sizes, epochs=epochs, verbose=2,
            validation_data=([x_test[:, 0], x_test[:, 1]], y_test),
            callbacks=[EarlyStopping(monitor='val_loss', patience=patience, verbose=2),
                       ModelCheckpoint(filepath=path_model, monitor='val_loss', save_best_only='True', verbose=2)])
        #
        return model, history
    elif flag == '_layer_2':
        print('in mlp, two layer')
        # network, two layers
        input_a = Input(shape=input_shape, name='input_a')  # re-use the same instance `base_network` and weight
        input_b = Input(shape=input_shape, name='input_b')
        dense1_a = Dense(1024, activation='relu', name='dense1_a')(input_a)
        dense2_a = Dense(512, activation='relu', name='dense2_a')(dense1_a)
        dense1_b = Dense(1024, activation='relu', name='dense1_b')(input_b)
        dense2_b = Dense(512, activation='relu', name='dense2_b')(dense1_b)
        # calculate distance of dense2_a and dense2_b
        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([dense2_a, dense2_b])
        model = Model(inputs=[input_a, input_b], outputs=distance)  # create model and optimize distance
        # model.summary()
        # train
        model.compile(loss=contrastive_loss, optimizer=Adam(lr=0.0005), metrics=[accuracy])
        history = model.fit(
            [x_train[:, 0], x_train[:, 1]], y_train,
            batch_size=batch_sizes, epochs=epochs, verbose=2,
            validation_data=([x_test[:, 0], x_test[:, 1]], y_test),
            callbacks=[EarlyStopping(monitor='val_loss', patience=patience, verbose=2),
                       ModelCheckpoint(filepath=path_model, monitor='val_loss', save_best_only='True', verbose=2)])
        #
        return model, history
    elif flag == '_layer_1':
        print('in mlp, one layer')
        # network, one layer
        input_a = Input(shape=input_shape, name='input_a')  # re-use the same instance `base_network` and weight
        input_b = Input(shape=input_shape, name='input_b')
        dense1_a = Dense(1024, activation='relu', name='dense1_a')(input_a)
        dense1_b = Dense(1024, activation='relu', name='dense1_b')(input_b)
        # calculate distance of dense1_a and dense1_b
        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([dense1_a, dense1_b])
        model = Model(inputs=[input_a, input_b], outputs=distance)  # create model and optimize distance
        # model.summary()
        # train
        model.compile(loss=contrastive_loss, optimizer=Adam(lr=0.0005), metrics=[accuracy])
        history = model.fit(
            [x_train[:, 0], x_train[:, 1]], y_train,
            batch_size=batch_sizes, epochs=epochs, verbose=2,
            validation_data=([x_test[:, 0], x_test[:, 1]], y_test),
            callbacks=[EarlyStopping(monitor='val_loss', patience=patience, verbose=2),
                       ModelCheckpoint(filepath=path_model, monitor='val_loss', save_best_only='True', verbose=2)])
        #
        return model, history
    elif flag == '_net_cnn':
        print('in cnn, one layer')
        # network, cnn
        input_a = Input(shape=input_shape, name='input_a')  # re-use the same instance `base_network` and weight
        input_b = Input(shape=input_shape, name='input_b')
        embed_a = Embedding(input_dim=input_shape[0], output_dim=1, input_length=input_shape[0], name='embed_a')(
            input_a)
        cnn_a = Conv1D(filters=32, kernel_size=1, activation='relu', name='cnn_a')(embed_a)
        flatten_a = Flatten(name='flatten_a')(cnn_a)
        dense_a = Dense(256, activation='relu', name='dense_a')(flatten_a)
        embed_b = Embedding(input_dim=input_shape[0], output_dim=1, input_length=input_shape[0], name='embed_b')(
            input_b)
        cnn_b = Conv1D(filters=32, kernel_size=1, activation='relu', name='cnn_b')(embed_b)
        flatten_b = Flatten(name='flatten_b')(cnn_b)
        dense_b = Dense(256, activation='relu', name='dense_b')(flatten_b)
        # calculate distance of dense_a and dense_b
        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([dense_a, dense_b])
        model = Model(inputs=[input_a, input_b], outputs=distance)  # create model and optimize distance
        # model.summary()
        # train
        model.compile(loss=contrastive_loss, optimizer=Adam(lr=0.0005), metrics=[accuracy])
        history = model.fit(
            [x_train[:, 0], x_train[:, 1]], y_train,
            batch_size=batch_sizes, epochs=epochs, verbose=1,
            validation_data=([x_test[:, 0], x_test[:, 1]], y_test),
            callbacks=[EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
                       ModelCheckpoint(filepath=path_model, monitor='val_loss', save_best_only='True', verbose=2)])
        #
        return model, history
    elif flag == '_net_lstm':
        print('in lstm, one layer')
        batch_sizes, patience = 1024, 1
        # network, lstm
        input_a = Input(shape=input_shape, name='input_a')  # re-use the same instance `base_network` and weight
        input_b = Input(shape=input_shape, name='input_b')
        embed_a = Embedding(input_dim=input_shape[0], output_dim=1, input_length=input_shape[0], name='embed_a')(
            input_a)
        lstm_a = LSTM(units=32, activation='relu', name='lstm_a')(embed_a)
        dense_a = Dense(256, activation='relu', name='dense_a')(lstm_a)
        embed_b = Embedding(input_dim=input_shape[0], output_dim=1, input_length=input_shape[0], name='embed_b')(
            input_b)
        lstm_b = LSTM(units=32, activation='relu', name='lstm_b')(embed_b)
        dense_b = Dense(256, activation='relu', name='dense_b')(lstm_b)
        # calculate distance of dense_a and dense_b
        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([dense_a, dense_b])
        model = Model(inputs=[input_a, input_b], outputs=distance)  # create model and optimize distance
        # model.summary()
        # train
        model.compile(loss=contrastive_loss, optimizer=Adam(lr=0.0005), metrics=[accuracy])
        history = model.fit(
            [x_train[:, 0], x_train[:, 1]], y_train,
            batch_size=batch_sizes, epochs=epochs, verbose=1,
            validation_data=([x_test[:, 0], x_test[:, 1]], y_test),
            callbacks=[EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
                       ModelCheckpoint(filepath=path_model, monitor='val_loss', save_best_only='True', verbose=2)])
        #
        return model, history


# for metrics
def get_accuracy(y_true, y_pred):
    """
    Operation on tensor.
    Compute classification accuracy with a fixed threshold on distances.
    :param y_true:
    :param y_pred:
    :return:
    """
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


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
    axs[1].plot(history.history["accuracy"], label="train_acc")
    axs[1].plot(history.history["val_accuracy"], label="val_acc")
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(range(len(history.history["accuracy"])))
    axs[1].set_xticklabels(range(1, len(history.history["accuracy"]) + 1), rotation=90)
    axs[1].legend()
    #
    plt.tight_layout()
    plt.savefig(path_figure_out, dpi=1200)
    # plt.show()


if __name__ == '__main__':
    ############################## drebin, part, all feature, rf ##############################
    # drebin_part (1512, 223)/(195, 223), features all 221
    path_in = 'F:/12.soj_few_shot/4_siamese/drebin_part/'
    path_feature_in = os.path.join(path_in, 'feature/5_drebin_MI.csv')
    paths_train_in = [os.path.join(path_in, 'train/train_part_{}.csv'.format(_)) for _ in range(10)]
    paths_test_in = [os.path.join(path_in, 'test/test_part_{}.csv'.format(_)) for _ in range(10)]
    paths_model_out = [os.path.join(path_in, 'result/part_model_{}.h5'.format(_)) for _ in range(10)]
    paths_result_out = [os.path.join(path_in, 'result/part_result_{}.txt'.format(_)) for _ in range(10)]
    paths_figure_out = [os.path.join(path_in, 'result/part_figure_{}.pdf'.format(_)) for _ in range(10)]

    # para
    # name_data, name_model, name_scale = ['drebin', 'androzoo'], ['siamese'], ['all', 'part']  # log
    # k_max, k_skip = [221, 146, 133], 10  # each 10 feature training a model
    # num_classes, num_features = [105, 108], [(221,), (146,), (133,)]  # drebin/androzoo, all/hyperlink/similarity
    name_data, name_model, name_scale = 'drebin', 'siamese', 'part'
    k_max, k_skip = 221, 10  # each 10 feature training a model
    num_classes, num_features = 105, (221,)  # drebin, all
    # epoch, batch_size, patience = 2, 128, 1  # 100, 64, 3
    epoch, batch_size, patience = 100, 128, 3  # 100, 128, 3
    # metrics
    metrics_label = ['accuracy', 'precision', 'recall', 'f1score', 'matthews correlation coefficient']
    name_label = ['_layer_1', '_layer_2', '_layer_3', '_net_cnn', '_net_lstm']
    ############################## drebin, part, all feature, rf ##############################

    ############################## androzoo, part, all feature, rf ##############################
    # # androzoo_part (1472, 223)/(198, 223), features all 221
    # path_in = 'F:/12.soj_few_shot/4_siamese/androzoo_part/'
    # path_feature_in = os.path.join(path_in, 'feature/5_androzoo_MI.csv')
    # paths_train_in = [os.path.join(path_in, 'train/train_part_{}.csv'.format(_)) for _ in range(10)]
    # paths_test_in = [os.path.join(path_in, 'test/test_part_{}.csv'.format(_)) for _ in range(10)]
    # paths_model_out = [os.path.join(path_in, 'result/part_model_{}.h5'.format(_)) for _ in range(10)]
    # paths_result_out = [os.path.join(path_in, 'result/part_result_{}.txt'.format(_)) for _ in range(10)]
    # paths_figure_out = [os.path.join(path_in, 'result/part_figure_{}.pdf'.format(_)) for _ in range(10)]
    #
    # # para
    # # name_data, name_model, name_scale = ['drebin', 'androzoo'], ['siamese'], ['all', 'part']  # log
    # # k_max, k_skip = [221, 146, 133], 10  # each 10 feature training a model
    # # num_classes, num_features = [105, 108], [(221,), (146,), (133,)]  # drebin/androzoo, all/hyperlink/similarity
    # name_data, name_model, name_scale = 'androzoo', 'siamese', 'part'
    # k_max, k_skip = 221, 10  # each 10 feature training a model
    # num_classes, num_features = 108, (221,)  # androzoo, all
    # # epoch, batch_size, patience = 2, 128, 1  # 100, 64, 3
    # epoch, batch_size, patience = 100, 128, 3  # 100, 128, 3
    # # metrics
    # metrics_label = ['accuracy', 'precision', 'recall', 'f1score', 'matthews correlation coefficient']
    # name_label = ['_layer_1', '_layer_2', '_layer_3', '_net_cnn', '_net_lstm']
    ############################## androzoo, part, all feature, rf ##############################

    ############################## train, save ##############################
    for index, (path_train_in, path_test_in, path_model_out, path_result_out, path_figure_out) in enumerate(
            zip(paths_train_in, paths_test_in, paths_model_out, paths_result_out, paths_figure_out)):

        print('************************ Fold {}, {} ************************'.format(index, get_current_time()))
        path_model_out_, path_result_out_, path_figure_out_ = path_model_out, path_result_out, path_figure_out

        if index < 5:
            continue

        for i in name_label:  # 1/2/3 layer, cnn, lstm
            # new paths
            print('Generate new path and new feature number about k layers or net...')
            path_model_out = path_model_out_.replace('.h5', '{}.h5'.format(i))
            path_result_out = path_result_out_.replace('.txt', '{}.txt'.format(i))
            path_figure_out = path_figure_out_.replace('.pdf', '{}.pdf'.format(i))
            print(path_model_out, path_result_out, path_figure_out)

            print('Choose {} features by mutual_info...'.format(k_max))
            feature_set = get_feature_set(path_in=path_feature_in, k=k_max)
            train, test, x_train, y_train, x_test, y_test = get_train_test(path_train_in, path_test_in, feature_set)
            y_train_nn, y_test_nn = to_categorical(y_train), to_categorical(y_test)
            # y_train_, y_test_ = np.argmax(y_train_nn, axis=-1), np.argmax(y_test_nn, axis=-1)  # one hot to number
            print(x_train.shape)

            print('Get pairs...')
            digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]  # a list including n arrays
            tr_pairs, tr_y = get_pairs(x_train, digit_indices)  #
            digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
            te_pairs, te_y = get_pairs(x_test, digit_indices)  #

            print('Train siamese network and save model...')
            time1 = datetime.datetime.now()
            model, history = train_siamese(num_features, tr_pairs, tr_y, te_pairs, te_y, batch_size, epoch, patience,
                                           path_model_out, flag=i)
            time2 = datetime.datetime.now()

            # print('Save results when training siamese network...')
            log_notes = 'Notes: {}, {}, {}, {}\n'.format(name_data, name_model, name_scale, get_current_time())
            log_shapes = 'Shapes: training (source) {}, training (pairs) {}, '.format(x_train.shape, tr_pairs.shape)
            log_shapes += 'test (source) {}, test (pairs) {}\n'.format(x_test.shape, te_pairs.shape)
            log_times = 'Times: training {} in {} seconds\n'.format(name_model, (time2 - time1).seconds)
            get_results(log_notes, log_shapes, log_times, history, path_result_out, path_figure_out)

            # release ram and clean tf static graph
            print('Clear memory...')
            # clear_memory(model, tr_pairs, tr_y, te_pairs, te_y)
            del model
            del tr_pairs, tr_y, te_pairs, te_y
