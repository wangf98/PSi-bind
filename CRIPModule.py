import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from CRIPDataset import CRIPData
from prep_data.getData import dealwithdata
import pandas as pd

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class CRIPModule(nn.Module):

    def __init__(self, input_size=21, length=99, num_classes=2):
        super(CRIPModule, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_size, 102, 7)
        self.avgpool = nn.AvgPool1d(kernel_size=5, stride=5)
        self.drop1 = nn.Dropout(p=0.5)
        self.lstm = nn.LSTM(input_size=102, hidden_size=120, batch_first=True,
                            bidirectional=True)
        self.drop2 = nn.Dropout(p=0.25)
        self.fc = nn.Linear(18 * 2 * 120, num_classes)

    def forward(self, x):
        # input: [N, C, L]
        batch_size, _, _ = x.shape
        x = self.conv1(x)
        x = self.avgpool(x)
        x = self.drop1(x)
        x = torch.transpose(x, 1, 2)  # [N, L, C]
        output, (_, _) = self.lstm(x)
        output = output.contiguous().view(batch_size, -1)
        output = self.drop2(output)
        output = self.fc(output)
        logits = F.log_softmax(output, dim=1)
        return logits


def cripnet():
    model = CRIPModule()
    return model


class CripNet(pl.LightningModule):

    def __init__(self, protein, save_fn):
        super(CripNet, self).__init__()
        train_X, test_X, train_y, test_y = dealwithdata(protein)

        self.cripnet = cripnet()
        self.trainset = CRIPData(train_X, train_y)
        self.valset = CRIPData(test_X, test_y)
        self.learning_rate = 0.001
        self.f1_max = 0
        self.save_fn = save_fn

    def forward(self, x):
        # x.size() = [batch_size, time_size, input_size]
        x = x.transpose(1, 2).contiguous()
        return self.cripnet(x)

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