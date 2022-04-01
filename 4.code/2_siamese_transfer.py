#!/usr/bin/env python
# encoding: utf-8
# @time: 2022/2/22 17:28
# @author: Bai (๑•̀ㅂ•́)و✧
# @file: 6_siamese_transfer.py
# @software: PyCharm

"""
    The real siamese network

    About RQ2
"""

import os
import time
import datetime
import pandas as pd

from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import *

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
    rf = RandomForestClassifier(random_state=14, criterion='gini', max_depth=20, n_estimators=150)
    knn = KNeighborsClassifier(algorithm='brute', n_neighbors=2, weights='distance')
    #
    classifiers = [svm, dtree, rf, knn]
    classifiers_name = ['svm', 'dtree', 'rf', 'knn']
    classifiers_path = [path_out + '{}_{}.pkl'.format(_, fold) for _ in classifiers_name]

    return classifiers, classifiers_name, classifiers_path


# transfer sia model and fit ml model
def transfer_ml_model(x_train, y_train, x_test, y_test, sia_model, ml_model, path_ml_model):
    """
    transfer sia model and fit ml model
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param sia_model:
    :param ml_model:
    :param path_ml_model:
    :return:
    """

    # for load model
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

    source_model = load_model(sia_model, custom_objects={'contrastive_loss': contrastive_loss})  # reload
    feature_extractor = Model(inputs=source_model.get_layer('input_a').input,
                              outputs=source_model.get_layer('dense3_a').output)
    # input
    ml_x_train, ml_y_train = feature_extractor.predict(x_train, verbose=2), y_train
    ml_x_test, ml_y_test = feature_extractor.predict(x_test, verbose=2), y_test
    ml_x_train, ml_x_test = np.nan_to_num(ml_x_train), np.nan_to_num(ml_x_test)  # nan to 0

    # train and test
    ml_model.fit(ml_x_train, ml_y_train)
    ml_y_pre = ml_model.predict(ml_x_test)
    ml_y_pre_proba = ml_model.predict_proba(ml_x_test)

    # save model
    # joblib.dump(ml_model, path_ml_model)
    del ml_model

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
def get_false_nagative(y_te, y_pr, path_result_out):
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
    path_in = 'F:/8.temp/13.few_shot/sia_part/'
    path_feature_in = os.path.join(path_in, 'feature/mutual_info_classif.csv')
    paths_train_in = [os.path.join(path_in, 'train/train_sia_part_{}.csv'.format(_)) for _ in range(10)]
    paths_test_in = [os.path.join(path_in, 'test/test_sia_part_{}.csv'.format(_)) for _ in range(10)]
    path_out = os.path.join(path_in, 'result/sia_part_')
    paths_model_in = [os.path.join(path_in, 'result/sia_part_model_{}.h5'.format(_)) for _ in range(10)]
    paths_result_out = [os.path.join(path_in, 'result/sia_part_result_{}.txt'.format(_)) for _ in range(10)]

    # para
    name_data, name_model, name_scale = 'drebin', 'siamese', 'part'
    k_max, k_skip = 333, 10  # feature of drebin, each 10 feature training a model
    num_classes, num_features = 131, (333,)  # for drebin
    # epoch, batch_size, patience = 1, 128, 1  # 100, 32, 5
    epoch, batch_size, patience = 100, 128, 5  # 100, 32, 5
    # metrics
    metrics_label = ['accuracy', 'precision', 'recall', 'f1score', 'matthews correlation coefficient']

    for index, (path_train_in, path_test_in, path_model_in, path_result_out) in enumerate(
            zip(paths_train_in, paths_test_in, paths_model_in, paths_result_out)):

        print('************************ Fold {}, {} ************************'.format(index, get_current_time()))

        print('Choose {} features by mutual_info...'.format(k_max))
        feature_set = get_feature_set(path_in=path_feature_in, k=k_max)
        train, test, x_train, y_train, x_test, y_test = get_train_test(path_train_in, path_test_in, feature_set)
        y_train_nn, y_test_nn = to_categorical(y_train), to_categorical(y_test)  # (5139, 131) (221, 131)
        # y_train_, y_test_ = np.argmax(y_train_nn, axis=-1), np.argmax(y_test_nn, axis=-1)  # one hot to number
        print(x_train.shape, y_train.shape)

        print('Save results by transferred siamese network...')
        clfs, clfs_name, clfs_path = get_classifier(path_out, index)
        for index, (clf, clf_name, clf_path) in enumerate(zip(clfs, clfs_name, clfs_path)):
            print('----------------- {} | {} -----------------'.format(clf_name, get_current_time()))
            y_pre, y_pre_proba = transfer_ml_model(x_train, y_train, x_test, y_test, path_model_in, clf, clf_path)
            get_results(y_test, y_pre, y_pre_proba, clf_name, path_result_out)
            get_false_nagative(y_test, y_pre, path_result_out)
