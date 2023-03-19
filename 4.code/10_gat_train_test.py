import csv as cv
import numpy as np
import torch
import pickle as pkl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from GNN_model import GAT
import os
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

metrics_label = ['accuracy', 'precision', 'recall', 'f1score', 'matthews correlation coefficient']


def get_current_time():
    """
    :return: current time
    """
    current_time = '{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    return current_time


def get_results(y_te, y_pr, clf_name, path_result_out, n_class):
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
    cr = classification_report(y_te, y_pr, labels=range(n_class),
                               target_names=['family_' + str(i) for i in list(set(y_te))], digits=4)

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


def create_data_list(data):
    data_list = []
    for line in data:
        x = torch.tensor(line['node'])
        edge_index = torch.tensor(line['edge'], dtype=torch.int64)
        y = torch.tensor(int(line['label']))
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list


base_dir = ['androzoo', 'drebin']
# base_dir = ['drebin']
scale = ['part']
kind = ['hyperlink', 'similarity']

for d in base_dir:
    for s in scale:
        for ki in kind:
            for i in range(0, 9):
                test_data_path = 'data/bai3/' + d + '_' + s + '_' + ki + '/' + 'test/' + 'test_{}_{}.pkl'.format(
                    s,
                    i)
                train_data_path = 'data/bai3/' + d + '_' + s + '_' + ki + '/' + 'train/' + 'train_{}_{}.pkl'.format(
                    s, i)
                with open(test_data_path, 'rb') as f:
                    test_data = pkl.load(f)
                    f.close()
                with open(train_data_path, 'rb') as f:
                    train_data = pkl.load(f)
                    f.close()

                train_set = create_data_list(train_data)
                test_set = create_data_list(test_data)

                train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
                test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
                num_class = 0
                if d == 'androzoo':
                    num_class = 108
                else:
                    num_class = 105
                model = GAT(50, 50, num_class)
                # model.cuda()

                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                criterion = torch.nn.CrossEntropyLoss()

                best_val_acc = 0.0
                best_epoch = 0
                output_file = 'output/GAT/{}_{}/{}/{}_{}_{}.txt'.format(d, ki, s, d, s, i)

                pre = []

                for epoch in range(15000):
                    model.train()
                    for data in train_loader:
                        # data.cuda()
                        optimizer.zero_grad()
                        print(data.x.shape, data.edge_index.shape)
                        out = model(data.x, data.edge_index, data.batch)
                        loss = criterion(out, data.y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        optimizer.step()

                    model.eval()
                    with torch.no_grad():  # 不计算梯度，不改变模型参数，只输出测试集结果
                        data = next(iter(test_loader))
                        # data.cuda()
                        out = model(data.x, data.edge_index, data.batch)
                        preds = out.argmax(dim=1)
                        correct = preds == data.y
                        acc = int(correct.sum()) / int(correct.size(0))

                        print(f"Epoch: {epoch}, test_acc: {acc:.4f}")

                        if acc > best_val_acc:
                            print("Val improved")
                            best_val_acc = acc
                            best_epoch = epoch

                            pre = preds

                            # torch.save(checkpoints, os.path.join(output_dir, f"best_model"))
                        # if epoch - best_epoch > 100:
                        #     break

                # 统计
                test_data = next(iter(test_loader))
                # get_results(test_data.y.tolist(), pre.tolist(), 'GAT', output_file, num_class)
                # get_false_nagative(test_data.y.tolist(), pre.tolist(), output_file)
                print(111)
