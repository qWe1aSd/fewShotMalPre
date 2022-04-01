#!/usr/bin/env python
# encoding: utf-8
# @time: 2022/2/22 17:28
# @author: Bai (๑•̀ㅂ•́)و✧
# @file: 5_siamese_training.py
# @software: PyCharm

"""
    The real siamese network

    loss function:
        - 原文，http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    About RQ2
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Lambda
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
    feature_list = pd.read_csv(path_in)['name'].tolist()
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
def train_siamese(input_shape, x_train, y_train, x_test, y_test, batch_sizes, epochs, patience, path_model):
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

    # network
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
    """
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to                     
        ==================================================================================================
        input_a (InputLayer)            (None, 333)          0                                            
        __________________________________________________________________________________________________
        input_b (InputLayer)            (None, 333)          0                                            
        __________________________________________________________________________________________________
        dense1_a (Dense)                (None, 1024)         342016      input_a[0][0]                    
        __________________________________________________________________________________________________
        dense1_b (Dense)                (None, 1024)         342016      input_b[0][0]                    
        __________________________________________________________________________________________________
        dense2_a (Dense)                (None, 512)          524800      dense1_a[0][0]                   
        __________________________________________________________________________________________________
        dense2_b (Dense)                (None, 512)          524800      dense1_b[0][0]                   
        __________________________________________________________________________________________________
        dense3_a (Dense)                (None, 256)          131328      dense2_a[0][0]                   
        __________________________________________________________________________________________________
        dense3_b (Dense)                (None, 256)          131328      dense2_b[0][0]                   
        __________________________________________________________________________________________________
        lambda_1 (Lambda)               (None, 1)            0           dense3_a[0][0]                   
                                                                         dense3_b[0][0]                   
        ==================================================================================================
        Total params: 1,996,288
        Trainable params: 1,996,288
        Non-trainable params: 0
        __________________________________________________________________________________________________
    """

    # train
    model.compile(loss=contrastive_loss, optimizer=Adam(lr=0.0005), metrics=[accuracy])
    history = model.fit(
        [x_train[:, 0], x_train[:, 1]], y_train,
        batch_size=batch_sizes, epochs=epochs, verbose=2,
        validation_data=([x_test[:, 0], x_test[:, 1]], y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience, verbose=2),
                   ModelCheckpoint(filepath=path_model, monitor='val_loss', save_best_only='True', verbose=2)])

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
    path_in = 'F:/8.temp/13.few_shot/sia_part/'
    path_feature_in = os.path.join(path_in, 'feature/mutual_info_classif.csv')
    paths_train_in = [os.path.join(path_in, 'train/train_sia_part_{}.csv'.format(_)) for _ in range(10)]
    paths_test_in = [os.path.join(path_in, 'test/test_sia_part_{}.csv'.format(_)) for _ in range(10)]
    paths_model_out = [os.path.join(path_in, 'result/sia_part_model_{}.h5'.format(_)) for _ in range(10)]
    paths_result_out = [os.path.join(path_in, 'result/sia_part_result_{}.txt'.format(_)) for _ in range(10)]
    paths_figure_out = [os.path.join(path_in, 'result/sia_part_figure_{}.pdf'.format(_)) for _ in range(10)]

    name_data, name_model, name_scale = 'drebin', 'siamese', 'part'
    k_max, k_skip = 333, 10  # feature of drebin, each 10 feature training a model
    num_classes, num_features = 131, (333,)  # for drebin
    # epoch, batch_size, patience = 1, 128, 1  # 100, 128, 5
    epoch, batch_size, patience = 100, 128, 5  # 100, 128, 5

    for index, (path_train_in, path_test_in, path_model_out, path_result_out, path_figure_out) in enumerate(
            zip(paths_train_in, paths_test_in, paths_model_out, paths_result_out, paths_figure_out)):
        print('************************ Fold {}, {} ************************'.format(index, get_current_time()))

        print('Choose {} features by mutual_info...'.format(k_max))
        feature_set = get_feature_set(path_in=path_feature_in, k=k_max)
        train, test, x_train, y_train, x_test, y_test = get_train_test(path_train_in, path_test_in, feature_set)
        y_train_nn, y_test_nn = to_categorical(y_train), to_categorical(y_test)  # (5139, 131) (221, 131)
        # y_train_, y_test_ = np.argmax(y_train_nn, axis=-1), np.argmax(y_test_nn, axis=-1)  # one hot to number
        print(x_train.shape)

        print('Get pairs...')
        digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]  # a list including 131 arrays
        tr_pairs, tr_y = get_pairs(x_train, digit_indices)  # (171991, 2, 333) (171991,), (1551441, 2, 333)
        digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
        te_pairs, te_y = get_pairs(x_test, digit_indices)  # (24531, 2, 333) (24531,), (24531, 2, 333)

		print('Train siamese network and save model...')
		time1 = datetime.datetime.now()
		model, history = train_siamese(num_features, tr_pairs, tr_y, te_pairs, te_y, batch_size, epoch, patience,
									   path_model_out)
		time2 = datetime.datetime.now()

		print('Save results when training siamese network...')
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


