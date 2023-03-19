#!/usr/bin/env python
# encoding: utf-8
# @time: 2023/2/26 23:48
# @author: Bai (๑•̀ㅂ•́)و✧
# @file: 26_prototype.py
# @software: PyCharm

"""
    drebin/androzoo * all/part，all/hyperlink/similarity feature，rf
        - drebin_part (1512, 223)/(195, 223), features all 221
        - drebin_part_hyperlink (1512, 223)/(195, 223), features hyperlink 146
        - drebin_part_similarity (1512, 223)/(195, 223), features similarity 133
        - androzoo_part (1472, 223)/(198, 223), features all 221
        - androzoo_part_hyperlink (1472, 223)/(198, 223), features hyperlink 146
        - androzoo_part_similarity (1472, 223)/(198, 223), features similarity 133

    参考代码和 2017 NIPS 论文：https://proceedings.neurips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf
    第一篇 CSDN：https://blog.csdn.net/hei653779919/article/details/106595614 --- 代码中 model 并未看到支持集
    第二篇 CSDN：https://blog.csdn.net/u014403221/article/details/126045466
    第三篇 CSDN：https://blog.csdn.net/qlkaicx/article/details/127739759 --- 查询集/支持集/训练验证测试集

    path in: 'F:/12.soj_few_shot/7_meta/*/feature'
             'F:/12.soj_few_shot/7_meta/*/train'
             'F:/12.soj_few_shot/7_meta/*/test'
    path out: 'F:/12.soj_few_shot/7_meta/*/result'

"""

import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter

from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef


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


# get support and query
def get_support_query(x_train, y_train, x_test, y_test):
    """
    train 1:1 划分为 support/query
    test 全部作 query
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    # 切分 train 为 support 和 query，先取 index
    support_index = []
    query_index = []
    for i in range(len(set(y_train))):
        total = np.where(y_train == i)[0]
        if len(total) % 2 == 0:
            support_ = total[:len(total) // 2]
            query_ = total[len(total) // 2:]
            support_index.extend(list(support_))
            query_index.extend(list(query_))
        else:
            support_ = total[:len(total) // 2 + 1]
            query_ = total[len(total) // 2 + 1:]
            support_index.extend(list(support_))
            query_index.extend(list(query_))
    x_train_support, y_train_support = x_train[support_index], y_train[support_index]
    x_train_query, y_train_query = x_train[query_index], y_train[query_index]

    x_train_support, y_train_support = torch.Tensor(x_train_support), torch.LongTensor(y_train_support)
    x_train_query, y_train_query = torch.Tensor(x_train_query), torch.LongTensor(y_train_query)
    x_test, y_test = torch.Tensor(x_test), torch.LongTensor(y_test)

    return x_train_support, y_train_support, x_train_query, y_train_query, x_test, y_test


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


# prototype learning
class Prototype(nn.Module):
    """
    需要修改计算 embedding 和的一行代码
    """

    def __init__(self, input_dim, hidden_dim, num_class):
        super(Prototype, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        # 线性层进行编码
        self.linear = nn.Linear(input_dim, hidden_dim)  # from input_dim to hidden_dim, 线性变换

    def embedding(self, features):
        result = self.linear(features)  # (*, dim) --> (*, embed), one hot to float
        return result

    def forward(self, support_input, support_label, query_input):
        # support
        support_embedding = self.embedding(support_input)  # (*, dim) --> (*, embed)
        support_size = support_embedding.shape[0]  # 支持集样本个数，第一维
        class_meta_dict = {}
        for i in range(0, self.num_class):  # class 0,1,2,*
            class_index = support_label == i  # boolean, 索引为 i 的 class index
            class_meta_dict[i] = torch.sum(support_embedding[class_index, :], dim=0) / sum(class_index)
        class_meta_information = torch.zeros(size=[len(class_meta_dict), support_embedding.shape[1]])
        for key, item in class_meta_dict.items():
            class_meta_information[key, :] = class_meta_dict[key]

        # query
        query_embedding = self.embedding(query_input)  # (*, dim) --> (*, embed)
        query_size = query_embedding.shape[0]  # 测试集样本个数，第一维
        result = torch.zeros(size=[query_size, self.num_class])  # [样本数量, 各类概率]
        for i in range(0, query_size):
            temp_value = query_embedding[i].repeat(self.num_class, 1)  # 按类 copy，每一个测试样本转换为 num_class*4
            cosine_value = torch.cosine_similarity(class_meta_information, temp_value, dim=1)
            result[i] = cosine_value
        result = F.log_softmax(result, dim=1)  # 计算每个测试样本属于不同类概率的 softmax

        return class_meta_information, result

    def predict(self, meta_information, test_input):
        # test
        test_embedding = self.embedding(test_input)  # (*, dim) --> (*, embed)
        test_size = test_embedding.shape[0]  # 测试集样本个数，第一维
        result = torch.zeros(size=[test_size, self.num_class])  # [样本数量, 各类概率]
        for i in range(0, test_size):
            temp_value = test_embedding[i].repeat(self.num_class, 1)  # 按类 copy，每一个测试样本转换为 num_class*4
            cosine_value = torch.cosine_similarity(meta_information, temp_value, dim=1)
            result[i] = cosine_value
        result = F.log_softmax(result, dim=1)  # 计算每个测试样本属于不同类概率的 softmax

        return result


# fit meta learning model
def general_ml_model(support_input, support_label, query_input, query_label, x_test, y_test, paras, paras_index):
    """
    训练集 1:1 分为支持集查询集，测试集仅测试用，tensor
    迭代 epoch 次，取 acc 最大一次的预测结果
    :param support_input: tensor
    :param support_label: tensor
    :param query_input: tensor
    :param query_label: tensor
    :param x_test: tensor
    :param y_test: tensor
    :param paras: {'hidden_dim': [*], 'lr': [*], *}
    :param paras: index
    :return:
    """
    # 初始化
    model = Prototype(support_input.numpy().shape[1], paras['hidden_dim'][paras_index], len(set(support_label.numpy())))
    optimizer = torch.optim.Adam(model.parameters(), paras['lr'][paras_index])
    #
    best_val_acc, best_epoch = 0., 0.
    ml_pre, ml_pre_proba = [], []

    for epoch_ in range(paras['epoch'][paras_index]):
        # train 中取支持集和查询集，query_label 对应 query_input
        optimizer.zero_grad()
        meta_info, output = model.forward(support_input, support_label, query_input)
        loss = F.nll_loss(output, query_label)
        loss.backward()
        optimizer.step()

        # test
        with torch.no_grad():  # 不计算梯度，不改变模型参数，只输出测试集结果
            predict_result = model.predict(meta_info, x_test)  # 预测概率
            _, predict_label = torch.max(predict_result.data, 1)  # 预测标签
            ml_y_pre = predict_label.numpy()
            ml_y_pre_proba = predict_result.data.numpy()

            correct = ml_y_pre == y_test.numpy()
            acc = int(correct.sum()) / int(len(correct))
            if acc > best_val_acc:
                print(f"Improved. Epoch: {epoch_}, test_acc: {acc:.4f}. {get_current_time()}")
                best_val_acc, best_epoch = acc, epoch_
                ml_pre, ml_pre_proba = ml_y_pre, ml_y_pre_proba

    return ml_pre, ml_pre_proba


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
            model_metrics[0], model_metrics[1], model_metrics[2], model_metrics[3], model_metrics[4],
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

    # drebin_part (1512, 223)/(195, 223), features all/hyperlink/similarity 221/146/133
    # androzoo_part (1472, 223)/(198, 223), features all/hyperlink/similarity 221/146/133
    path_in = 'F:/12.soj_few_shot/7_meta/1_prototype/'
    model_paras = {
        # path
        'path_in': [
            os.path.join(path_in, 'drebin_part'),
            os.path.join(path_in, 'drebin_part_hyperlink'),
            os.path.join(path_in, 'drebin_part_similarity'),
            os.path.join(path_in, 'androzoo_part'),
            os.path.join(path_in, 'androzoo_part_hyperlink'),
            os.path.join(path_in, 'androzoo_part_similarity')
        ],
        'path_feature': [
            os.path.join(path_in, 'drebin_part/feature/5_drebin_MI.csv'),
            os.path.join(path_in, 'drebin_part_hyperlink/feature/6_drebin_MI_hyperlink.csv'),
            os.path.join(path_in, 'drebin_part_similarity/feature/7_drebin_MI_similarity.csv'),
            os.path.join(path_in, 'androzoo_part/feature/5_drebin_MI.csv'),
            os.path.join(path_in, 'androzoo_part_hyperlink/feature/6_drebin_MI_hyperlink.csv'),
            os.path.join(path_in, 'androzoo_part_similarity/feature/7_drebin_MI_similarity.csv')
        ],
        # family
        'k_max': [221, 146, 133, 221, 146, 133],
        'num_classes': [105, 105, 105, 108, 108, 108],
        'num_features': [221, 146, 133, 221, 146, 133],
        # hyper
        'hidden_dim': [50, 50, 50, 50, 50, 50],
        'lr': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        'epoch': [100, 100, 100, 100, 100, 100]
    }
    model_metrics = ['accuracy', 'precision', 'recall', 'f1score', 'matthews correlation coefficient']

    #
    for idx in range(len(model_paras['path_in'])):

        path_temp = model_paras['path_in'][idx]
        paths_train_in = [os.path.join(path_temp, 'train/train_part_{}.csv'.format(_)) for _ in range(10)]
        paths_test_in = [os.path.join(path_temp, 'test/test_part_{}.csv'.format(_)) for _ in range(10)]
        path_out = os.path.join(path_temp, 'result/part_')
        paths_model_in = [os.path.join(path_temp, 'result/part_model_{}.h5'.format(_)) for _ in range(10)]
        paths_result_out = [os.path.join(path_temp, 'result/part_result_{}.txt'.format(_)) for _ in range(10)]

        for index, (path_train_in, path_test_in, path_model_in, path_result_out) in enumerate(
                zip(paths_train_in, paths_test_in, paths_model_in, paths_result_out)):

            print('************************ Fold {}, {} ************************'.format(index, get_current_time()))
            print('Choose {} features by mutual_info...'.format(model_paras['k_max'][idx]))
            feature_set = get_feature_set(path_in=model_paras['path_feature'][idx], k=model_paras['k_max'][idx])
            train, test, x_train, y_train, x_test, y_test = get_train_test(path_train_in, path_test_in, feature_set)
            x_train_s_ten, y_train_s_ten, x_train_q_ten, y_train_q_ten, x_test_ten, y_test_ten = get_support_query(
                x_train, y_train, x_test, y_test
            )
            print(x_train.shape, x_test.shape, x_train_s_ten.shape, x_train_q_ten.shape, x_test_ten.shape)

            print('Save results by meta learning...')
            clfs, clfs_name, clfs_path = get_classifier(path_out, index)
            for _, (clf, clf_name, clf_path) in enumerate(zip(clfs, clfs_name, clfs_path)):
                print('----------------- {} | {} -----------------'.format(clf_name, get_current_time()))
                y_pre, y_pre_proba = general_ml_model(x_train_s_ten, y_train_s_ten, x_train_q_ten, y_train_q_ten,
                                                      x_test_ten, y_test_ten, model_paras, idx)
                get_results(y_test, y_pre, y_pre_proba, clf_name, path_result_out)
                get_false_negative(y_test, y_pre, path_result_out)
