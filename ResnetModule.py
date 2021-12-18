import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from CRIPDataset import CRIPData
from prep_data.getData import dealwithdata2
import pandas as pd
import math
from ResnetDataset import ResnetData

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


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
    def __init__(self, ResidualBlock, inp_size, length, num_classes=2):
        super(ResNet, self).__init__()

        f = lambda x: math.floor(x / 32)
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(inp_size, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512*f(length), num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
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
        out = self.fc(out)
        logits = F.log_softmax(out, dim=1)
        return logits

class CnnNet(nn.Module):
    
    def __init__(self, inp1_size=100, len1=95, num_classes=2):
        super(CnnNet, self).__init__()
        # self.conv1 = nn.Conv1d(input_size, 8, kernel_size=7, stride=2, padding=3, bias=False)

        # head1
        self.conv1_1 = torch.nn.Conv1d(inp1_size, 32, 5, padding=2)
        self.conv1_2 = torch.nn.Conv1d(32, 32, 5, padding=2)
        self.conv1_3 = torch.nn.Conv1d(32, 32, 5, padding=2)
        self.maxpool1 = nn.MaxPool1d(len1)

        self.fc = nn.Linear(32, num_classes)

    def forward(self, x1):

        # head1
        x1 = self.conv1_1(x1)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.maxpool1(x1)

        x1 = x1.view(x1.size(0), -1)
        out = self.fc(x1)
        logits = F.log_softmax(out, dim=1)

        return logits

class ResnetNet(pl.LightningModule):

    def __init__(self, protein, save_fn, inp_size=100, len=95):
        super(ResnetNet, self).__init__()

        train_X, test_X, train_y, test_y = dealwithdata2(protein)
        self.net = CnnNet(inp_size, len)
        self.trainset = ResnetData('./prep_data/7MerRna100.json.npy', train_X, train_y)
        self.valset = ResnetData('./prep_data/7MerRna100.json.npy', test_X, test_y)
        self.learning_rate = 0.001
        self.f1_max = 0
        self.save_fn = save_fn

    def forward(self, x):
        # x.size() = [batch_size, time_size, input_size]
        x = x.transpose(1, 2).contiguous()
        return self.net(x)

    def loss(self, y_hat, y):
        return F.nll_loss(y_hat, y)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]

    def training_step(self, data_batch, batch_nb):
        x, y = data_batch['feature'].cuda(), data_batch['label'].cuda()

        y_hat = self.forward(x)
        val_loss = self.loss(y_hat, y)
        val_acc = (torch.argmax(y_hat, dim=1) == y).sum(dim=0) / (len(y) * 1.0)

        return {
            'loss': val_loss
        }

    def validation_step(self, data_batch, batch_nb):
        x, y = data_batch['feature'].cuda(), data_batch['label'].cuda()
        y_hat = self.forward(x)
        y_pred = y_hat.argmax(dim=1)
        val_loss = self.loss(y_hat, y)

        val_acc = (y_pred == y).sum(dim=0).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        return {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'y_true': y,
            'y_pred': y_pred,
            'y_hat': torch.exp(y_hat[:, 1])
        }

    def validation_end(self, outputs):
        val_loss_mean = 0
        val_acc_mean = 0
        y_true = []
        y_pred = []
        y_hat = []

        for output in outputs:
            val_loss_mean += output['val_loss']
            val_acc_mean += output['val_acc']
            y_true.extend(list(np.array(output['y_true'].cpu())))
            y_pred.extend(list(np.array(output['y_pred'].cpu())))
            y_hat.extend(list(np.array(output['y_hat'].cpu())))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_hat)

        if self.f1_max < f1:
            self.f1_max = f1
            self._save(y_true, y_pred, y_hat)

        return {
            'val_loss': val_loss_mean.item(),
            'val_acc': val_acc_mean.item(),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
        }

    def _save(self, y_true, y_pred, y_hat):

        df = pd.DataFrame(data={
            'y_true': y_true,
            'y_pred': y_pred,
            'y_hat': y_hat
        })
        df.to_csv(self.save_fn, index=False)

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(self.trainset, batch_size=128, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=256, shuffle=True)