import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from DHDataset import DHData
import pandas as pd

import numpy as np
import math as math
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from prep_data.getData import dealwithdata3, dealwithdata4, dealwithdata5, dealwithdata6

from tqdm import tqdm
import time
import random

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, length, inp_size):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(inp_size, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, 4)
        out = out.view(out.size(0), -1)
        return out


class DoubleHeadNNModule(nn.Module):

    def __init__(self, inp1_size=100, len1=96, inp2_size=128, len2=340, num_classes=2):
        super(DoubleHeadNNModule, self).__init__()

        f = lambda x: math.floor(x / 32)

        # head1
        self.res1 = ResNet(ResidualBlock, len1, inp1_size)
        # head2
        self.res2 = ResNet(ResidualBlock, len2, inp2_size)

        self.fc = nn.Linear(512 * (f(len1) + f(len2)), num_classes)

    def forward(self, x1, x2):
        # head1
        x1 = self.res1(x1)
        # head2
        x2 = self.res2(x2)

        x_merge = torch.cat(
            (x1, x2), dim=1
        )
        x_merge = x_merge.view(x_merge.size(0), -1)
        out = self.fc(x_merge)
        logits = F.log_softmax(out, dim=1)

        return torch.exp(logits)


def dhnet(input_size=None, length=None):
    """construct a cnn model
    """
    model = DoubleHeadNNModule()
    return model


class DHNet(nn.Module):

    def __init__(self):

        super(DHNet, self).__init__()
        self.dhnet = dhnet()

    def forward(self, x1, x2):
        # x.size() = [batch_size, time_size, input_size]
        x1 = x1.transpose(1, 2).contiguous()
        x2 = x2.transpose(1, 2).contiguous()
        return self.dhnet(x1, x2)


if __name__ == '__main__':
    rna_embed_fn = './prep_data/7MerRna100.json.npy'
    protein_embed_fn = './prep_data/5MerProtein128_rbp.npy'
    dict_pth = os.path.join('.', 'board_dhnet_whh_noseen', 'dhnet.pth')
    net = DHNet()
    net.load_state_dict(torch.load(dict_pth))

    # linear rna数据集 ########################################################
    # fns = os.listdir('./clip')
    # fns.sort(key=lambda x: int(x.split('_')[0]))
    # # print(fns)
    # proteins = [(fn.split('_')[0], fn.split('_')[2].upper(), fn) for fn in fns]
    # # print(proteins)
    ##########################################################################

    # circular rna数据集 ######################################################
    proteins = os.listdir('./RBP37')
    random.Random(100).shuffle(proteins)
    print(proteins)
    protein_list = [(protein, protein) for protein in proteins]
    ##########################################################################

    # 统计每个独立的test data ################################################## 
    y_true_all, y_pred_all = [], []
    for idx, (protein, fn) in enumerate(protein_list):
        if idx < math.floor(len(protein_list) * 0.8):   continue
        test_X, test_y = dealwithdata6(protein, fn)
        if test_X == 'error':
            print()
            print("第{}个文件对应的蛋白{}不存在".format(idx, protein))
            continue

        testset = DHData(rna_embed_fn, protein_embed_fn, test_X, test_y)
        testloader = DataLoader(testset, batch_size=128, shuffle=False)
        pred_all, label_all = [], []
        for i, data in enumerate(testloader):
            rnas, proteins, labels = data['inp1'], data['inp2'], data['label']
            outputs = torch.argmax(net(rnas, proteins), dim=1)
            preds = list(np.array(outputs))
            labels = list(np.array(labels))
            pred_all.extend(preds)
            label_all.extend(labels)
            #print("preds: {}".format(preds))
            #print("labels: {}".format(labels))

        y_true = np.array(label_all)
        y_pred = np.array(pred_all)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print("第{}个文件对应的蛋白{}指标如下:".format(idx, protein))
        print("accuracy: {}, precision: {}, recall: {}, f1: {}".format(accuracy, precision, recall, f1))

        y_true_all.extend(label_all)
        y_pred_all.extend(pred_all)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    accuracy, precision, recall, f1 = accuracy_score(y_true_all, y_pred_all), precision_score(y_true_all, y_pred_all), recall_score(y_true_all, y_pred_all), f1_score(y_true_all, y_pred_all)
    print("全部测集集指标如下:")
    print("accuracy: {}, precision: {}, recall: {}, f1: {}".format(accuracy, precision, recall, f1))
    
    # 所有test data集合 #######################################################
    train_X, test_X, train_y, test_y = dealwithdata4()
    testset = DHData(rna_embed_fn, protein_embed_fn, train_X, train_y)
    testloader = DataLoader(testset, batch_size=512, shuffle=False)

    pred_all, label_all = [], []
    for i, data in enumerate(testloader):
        rnas, proteins, labels = data['inp1'], data['inp2'], data['label']
        outputs = torch.argmax(net(rnas, proteins), dim=1)
        preds = list(np.array(outputs))
        labels = list(np.array(labels))
        pred_all.extend(preds)
        label_all.extend(labels)
        #print("preds: {}".format(preds))
        #print("labels: {}".format(labels))

    y_true = np.array(label_all)
    y_pred = np.array(pred_all)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("全部测集集指标如下:")
    print("accuracy: {}, precision: {}, recall: {}, f1: {}".format(accuracy, precision, recall, f1))
    ###########################################################################