#!/usr/bin/env python
# encoding: utf-8
# @time: 2023/1/31 22:54
# @author: Bai (๑•̀ㅂ•́)و✧
# @file: 13_avpass.py
# @software: PyCharm

"""
    avpass，drebin/androzoo * all/part，all feature，rf
        - drebin_part (1512, 223)/(195, 223), features all 221
        - androzoo_part (1472, 223)/(198, 223), features all 221

    path in: 'F:/12.soj_few_shot/6_avpass/*/feature'
             'F:/12.soj_few_shot/6_avpass/*/train'
             'F:/12.soj_few_shot/6_avpass/*/test'
             'F:/12.soj_few_shot/6_avpass/*_input_2.csv' --- avpass 需要补充的 app
    path out: 'F:/12.soj_few_shot/6_avpass/*/result'

"""

import os
import time
import random
import pandas as pd

from collections import Counter
from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE

from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use cpu


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
def get_train_test(path_train_in, path_test_in, select_feature, path_supply, flag):
    """
    切掉 app name 后，前 n-2 列为 features，第 n-1 列为 label
    在前 n-2 列中选择 select_feature 列
    :param path_train_in:
    :param path_test_in:
    :param select_feature:
    :param path_supply:
    :param flag:
    :return:
    """
    data_train = pd.read_csv(path_train_in, encoding='utf-8')
    data_test = pd.read_csv(path_test_in, encoding='utf-8')
    # supplement from avpass apps
    data_supply = pd.read_csv(path_supply, encoding='utf-8')
    data_train = pd.concat([data_train, data_supply], axis=0)

    #
    data_x_train, data_y_train = data_train.loc[:, select_feature].values, data_train.iloc[:, -2].values
    data_x_test, data_y_test = data_test.loc[:, select_feature].values, data_test.iloc[:, -2].values

    return data_train, data_test, data_x_train, data_y_train, data_x_test, data_y_test


###############################################################################################


# get features and classifiers
def get_classifier(path_out, fold):
    """
    get classifiers
        - classifiers, classifiers_name
    :param path_out:
    :param fold:
    :return:
    """
    # model
    svm = SVC(C=10, decision_function_shape='ovo', gamma=0.1, kernel='rbf', probability=True)
    dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=157, splitter='best')
    rf = RandomForestClassifier(random_state=14, criterion='gini', max_depth=200, n_estimators=200)
    knn = KNeighborsClassifier(algorithm='brute', n_neighbors=2, weights='distance')
    #
    classifiers = [rf]
    classifiers_name = ['rf']
    classifiers_path = [path_out + '{}_{}.pkl'.format(_, fold) for _ in classifiers_name]

    return classifiers, classifiers_name, classifiers_path


# fit ml model
def general_ml_model(x_train, y_train, x_test, y_test, ml_model, path_ml_model):
    """
    fit general ml model
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param ml_model:
    :param path_ml_model:
    :return:
    """
    # input
    ml_x_train, ml_y_train = x_train, y_train
    ml_x_test, ml_y_test = x_test, y_test

    # train and test
    ml_model.fit(ml_x_train, ml_y_train)
    ml_y_pre = ml_model.predict(ml_x_test)
    ml_y_pre_proba = ml_model.predict_proba(ml_x_test)

    return ml_y_pre, ml_y_pre_proba


# get results [acc, pre, re, f1, mcc]
def get_results(y_te, y_pr, y_pr_proba, clf_name, path_result_out):
    """
    get results [acc, pre, re, f1, mcc]
    :param y_te: real y
    :param y_pr: predicted y
    :param y_pr_proba: predicted probability y
    :param clf_name:
    :param path_result_out:
    :return:
    """
    # cm = confusion_matrix(y_te, y_pr, labels=np.unique(y_te))
    cr = classification_report(y_te, y_pr, target_names=['family_' + str(i) for i in list(set(y_te))], digits=4)

    accuracy = accuracy_score(y_te, y_pr)
    precision = precision_score(y_te, y_pr, average='weighted')
    recall = recall_score(y_te, y_pr, average='weighted')
    f1score = f1_score(y_te, y_pr, average='weighted')
    mcc = matthews_corrcoef(y_te, y_pr)

    with open(path_result_out, 'a', encoding='utf-8') as f:
        f.write('Classifier: {}, {}\n'.format(clf_name, get_current_time()))
        # f.write('confusion matrix:\n')
        # for i in range(cm.shape[0]):
        #     f.write('\t{}\n'.format(cm.tolist()[i]))
        f.write('Classification report:\n{}'.format(cr))
        f.write('{}, {}, {}, {}, {}:\n{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(
            metrics_label[0], metrics_label[1], metrics_label[2], metrics_label[3], metrics_label[4],
            accuracy, precision, recall, f1score, mcc
        ))


# get false negative samples
def get_false_negative(y_te, y_pr, path_result_out):
    """
    分类器分错的为易混淆 app，这里集包括被错分为 benign app 的 malware，也包括被错分为 malware 的 benign app
    :param y_te:
    :param y_pr:
    :param path_result_out: to save to the index (in the source csv) of false negative samples
    :return:
    """
    fn_index = []
    real_label, false_label = [], []
    for index, (a, b) in enumerate(zip(y_te, y_pr)):
        if a != b:
            fn_index.append(index)
            real_label.append(a)
            false_label.append(b)
    with open(path_result_out, 'a', encoding='utf-8') as f:
        f.write('Index of false negative samples: {}\n'.format(str(fn_index)))
        f.write('Real family: {}\n'.format(str(real_label)))
        f.write('False (labeled) family: {}\n'.format(str(false_label)))
        f.write('----------------------------------------------------------------------\n')


if __name__ == '__main__':
    # # drebin_part (1512, 223)/(195, 223), features all 221
    # path_in = 'F:/12.soj_few_shot/6_avpass/drebin_part/'
    # path_feature_in = os.path.join(path_in, 'feature/5_drebin_MI.csv')
    # path_supply_in = 'F:/12.soj_few_shot/6_avpass/drebin_input_2.csv'
    # paths_train_in = [os.path.join(path_in, 'train/train_part_{}.csv'.format(_)) for _ in range(10)]
    # paths_test_in = [os.path.join(path_in, 'test/test_part_{}.csv'.format(_)) for _ in range(10)]
    # path_out = os.path.join(path_in, 'result/part_')
    # paths_result_out = [os.path.join(path_in, 'result/part_result_{}.txt'.format(_)) for _ in range(10)]
    # # para
    # # name_data, name_model, name_scale = ['drebin', 'androzoo'], ['siamese'], ['all', 'part']  # log
    # # k_max, k_skip = [221, 146, 133], 10  # each 10 feature training a model
    # # num_classes, num_features = [105, 108], [(221,), (146,), (133,)]  # drebin/androzoo, all/hyperlink/similarity
    # name_data, name_model, name_scale = 'drebin', 'siamese', 'all/part'
    # k_max, k_skip = 221, 10  # each 10 feature training a model
    # num_classes, num_features = 105, (221,)  # drebin, all
    # # epoch, batch_size, patience = 2, 128, 1  # 100, 32, 5
    # epoch, batch_size, patience = 100, 32, 5  # 100, 32, 5
    # # metrics
    # metrics_label = ['accuracy', 'precision', 'recall', 'f1score', 'matthews correlation coefficient']
    # data_flag = 'drebin'

    # androzoo_part (1472, 223)/(198, 223), features all 221
    path_in = 'F:/12.soj_few_shot/6_avpass/androzoo_part/'
    path_feature_in = os.path.join(path_in, 'feature/5_androzoo_MI.csv')
    path_supply_in = 'F:/12.soj_few_shot/6_avpass/androzoo_input_2.csv'
    paths_train_in = [os.path.join(path_in, 'train/train_part_{}.csv'.format(_)) for _ in range(10)]
    paths_test_in = [os.path.join(path_in, 'test/test_part_{}.csv'.format(_)) for _ in range(10)]
    path_out = os.path.join(path_in, 'result/part_')
    paths_result_out = [os.path.join(path_in, 'result/part_result_{}.txt'.format(_)) for _ in range(10)]
    # para
    # name_data, name_model, name_scale = ['drebin', 'androzoo'], ['siamese'], ['all', 'part']  # log
    # k_max, k_skip = [221, 146, 133], 10  # each 10 feature training a model
    # num_classes, num_features = [105, 108], [(221,), (146,), (133,)]  # drebin/androzoo, all/hyperlink/similarity
    name_data, name_model, name_scale = 'androzoo', 'siamese', 'all/part'
    k_max, k_skip = 221, 10  # each 10 feature training a model
    num_classes, num_features = 108, (221,)  # drebin, all
    # epoch, batch_size, patience = 2, 128, 1  # 100, 32, 5
    epoch, batch_size, patience = 100, 32, 5  # 100, 32, 5
    # metrics
    metrics_label = ['accuracy', 'precision', 'recall', 'f1score', 'matthews correlation coefficient']
    data_flag = 'androzoo'

    for index, (path_train_in, path_test_in, path_result_out) in enumerate(
            zip(paths_train_in, paths_test_in, paths_result_out)):
        print(path_train_in, path_test_in, path_result_out)

        print('Choose {} features by mutual_info...'.format(k_max))
        feature_set = get_feature_set(path_in=path_feature_in, k=k_max)
        train, test, x_train, y_train, x_test, y_test = get_train_test(path_train_in, path_test_in,
                                                                       feature_set, path_supply_in, flag=data_flag)
        y_train_nn, y_test_nn = to_categorical(y_train), to_categorical(y_test)  #
        # y_train_, y_test_ = np.argmax(y_train_nn, axis=-1), np.argmax(y_test_nn, axis=-1)  # one hot to number
        print(x_train.shape, y_train.shape)

        # clfs, clfs_name, clfs_path = get_classifier(path_out, index)
        # for index, (clf, clf_name, clf_path) in enumerate(zip(clfs, clfs_name, clfs_path)):
        #     print('----------------- {} | {} -----------------'.format(clf_name, get_current_time()))
        #     y_pre, y_pre_proba = general_ml_model(x_train, y_train, x_test, y_test, clf, clf_path)
        #     get_results(y_test, y_pre, y_pre_proba, clf_name, path_result_out)
        #     get_false_negative(y_test, y_pre, path_result_out)
