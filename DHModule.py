from argparse import ArgumentParser

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from DHDataset import DHData
#import pandas as pd

import numpy as np
import math as math
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from prep_data.getData import dealwithdata3, dealwithdata4, dealwithdata6, dealwithdata5, dealwithdata7, dealwithdata8, dealwithdata9

from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
from pytorch_lightning.callbacks import EarlyStopping
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
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)

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
        return out


class DoubleHeadNNModule(nn.Module):

    def __init__(self, inp1_size=100, len1=96, inp2_size=128, len2=340, num_classes=2):
        super(DoubleHeadNNModule, self).__init__()

        f = lambda x: math.floor(x/32)

        # head1
        self.res1 = ResNet(ResidualBlock, len1, inp1_size)
        # head2
        self.res2 = ResNet(ResidualBlock, len2, inp2_size)

        self.fc = nn.Linear(512*(f(len1)+f(len2)), num_classes)

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

        return logits


def dhnet(input_size=None, length=None):
    """construct a cnn model
    """
    model = DoubleHeadNNModule()
    return model


class DHNet(pl.LightningModule):

    def __init__(self, backbone, lr=0.001):

        super().__init__()

        #train_X, test_X, train_y, test_y = dealwithdata4()
        self.dhnet = backbone
        #self.trainset = DHData(rna_embed_fn, protein_embed_fn, train_X, train_y)
        #self.valset = DHData(rna_embed_fn, protein_embed_fn, test_X, test_y)
        self.learning_rate = lr
        #self.f1_max = 0
        self.accuracy = pl.metrics.Accuracy()
        #self.precision = pl.metrics.Precision()
        #self.recall = pl.metrics.Recall()
        #self.f1 = pl.metrics.F1()

        #self.save_fn = save_fn
        #self.dict_pth = dict_pth

    def forward(self, x1, x2):
        # x.size() = [batch_size, time_size, input_size]
        x1 = x1.transpose(1, 2).contiguous()
        x2 = x2.transpose(1, 2).contiguous()
        return self.dhnet(x1, x2)

    def loss(self, y_hat, y):
        return F.nll_loss(y_hat, y)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]

    def training_step(self, data_batch, batch_idx):
        x1, x2, y = data_batch['inp1'], data_batch['inp2'], data_batch['label']
        y_hat = self.forward(x1, x2)
        loss = self.loss(y_hat, y)
        #val_acc = (torch.argmax(y_hat, dim=1) == y).sum(dim=0) / (len(y) * 1.0)
        preds = torch.argmax(y_hat, dim=1)

        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc_step', self.accuracy(preds, y))
        return loss
    
    def training_epoch_end(self, outputs):
        self.log('train_acc_epoch', self.accuracy.compute())
        self.accuracy.reset()

    def validation_step(self, data_batch, batch_nb):
        #x1, x2, y = data_batch['inp1'].cuda(), data_batch['inp2'].cuda(), data_batch['label'].cuda()
        x1, x2, y = data_batch['inp1'], data_batch['inp2'], data_batch['label']
        y_hat = self.forward(x1, x2)
        preds = torch.argmax(y_hat, dim=1)

        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc_step', self.accuracy(preds, y))

        #val_acc = (y_pred == y).sum(dim=0).item() / (len(y) * 1.0)
        #val_acc = torch.tensor(val_acc)

        #return {
        #    'val_loss': val_loss,
        #    'val_acc': val_acc,
        #    'y_true': y,
        #    'y_pred': y_pred,
        #    'y_hat': torch.exp(y_hat[:, 1])
        #}

    def validation_epoch_end(self, outputs):
        self.log('val_acc_epoch', self.accuracy.compute())
        self.accuracy.reset()
    #    val_loss_mean = 0
    #    val_acc_mean = 0
    #    y_true = []
    #    y_pred = []
    #    y_hat = []

    #    for output in outputs:
    #        val_loss_mean += output['val_loss']
    #        val_acc_mean += output['val_acc']
    #        y_true.extend(list(np.array(output['y_true'].cpu())))
    #        y_pred.extend(list(np.array(output['y_pred'].cpu())))
    #        y_hat.extend(list(np.array(output['y_hat'].cpu())))

    #    y_true = np.array(y_true)
    #    y_pred = np.array(y_pred)

    #    val_loss_mean /= len(outputs)
    #    val_acc_mean /= len(outputs)

    #    precision = precision_score(y_true, y_pred)
    #    recall = recall_score(y_true, y_pred)
    #    f1 = f1_score(y_true, y_pred)
    #    auc = roc_auc_score(y_true, y_hat)

    #    if self.f1_max < f1:
    #        self.f1_max = f1
    #        self._save(y_true, y_pred, y_hat)
            #torch.save(self.state_dict(), self.dict_pth)

    #    return {
    #        'val_loss': val_loss_mean.item(),
    #        'val_acc': val_acc_mean.item(),
    #        'precision': precision,
    #        'recall': recall,
    #        'f1': f1,
    #        'auc': auc
    #    }

    # def _save(self, y_true, y_pred, y_hat):

    #     df = pd.DataFrame(data={
    #         'y_true': y_true,
    #         'y_pred': y_pred,
    #         'y_hat': y_hat
    #     })
    #     df.to_csv(self.save_fn, index=False)

    def test_step(self, batch, batch_idx):
        x1, x2, y = batch['inp1'], batch['inp2'], batch['label']
        y_hat = self.forward(x1, x2)
        preds = torch.argmax(y_hat, dim=1)

        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc_step', self.accuracy(preds, y))
    
    def test_epoch_end(self, outputs):
        self.log('test_acc_epoch', self.accuracy.compute())
        self.accuracy.reset()
    
    #@pl.data_loader
    #def tng_dataloader(self):
    #    return DataLoader(self.trainset, batch_size=128, shuffle=True)

    #@pl.data_loader
    #def val_dataloader(self):
    #    return DataLoader(self.valset, batch_size=256, shuffle=True)

    #@pl.data_loader
    #def test_dataloader(self):
    #    return DataLoader(self.valset, batch_size=256, shuffle=True)

#if __name__ == '__main__':
#    train_X, _, train_y, _ = dealwithdata3()
#    rna_embed_fn='./prep_data/7MerRna100.json.npy'
#    protein_embed_fn='./prep_data/5MerProtein128_rbp.npy'
#    trainset = DHData(rna_embed_fn, protein_embed_fn, train_X, train_y)

#    for i, data in enumerate(trainset):
#        inp2 = data['inp2']
#        print(inp2.size())
#        print(inp2)
#        break

def main():
    pl.seed_everything(1234)

   # parser = ArgumentParser()
   # parser.add_argument('--batch_size', default=128, type=int)

    # train_size = int(np.floor(len(train)*(11/12.0)))
    # val_size = len(train)-train_size
    # train, val = random_split(train, [train_size, val_size])
    train, val, test = dealwithdata9()

    rna_embed_fn='./prep_data/7MerRna100.json.npy'
    protein_embed_fn='./prep_data/5MerProtein128_rbp.npy'
    trainset = DHData(rna_embed_fn, protein_embed_fn, train)
    valset = DHData(rna_embed_fn, protein_embed_fn, val)
    testset = DHData(rna_embed_fn, protein_embed_fn, test)

    train_loader = DataLoader(trainset, batch_size=128)
    val_loader = DataLoader(valset, batch_size=256)
    test_loader = DataLoader(testset, batch_size=256)

    model = DHNet(dhnet())

    es = EarlyStopping(monitor='val_loss')
    trainer = pl.Trainer(
        callbacks=[es],
        max_epochs=20,
        gradient_clip_val=1.,
        default_root_dir='./prep_data/add_neg_version/noseen_version/',
        gpus = 1
        )
    trainer.fit(model, train_loader, val_loader)

    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':

    main()
    # pl.seed_everything(1234)
    # # model = DHNet.load_from_checkpoint('./prep_data/lightning_logs/version_1/checkpoints/epoch=19-step=90419.ckpt', backbone=dhnet())
    # model = DHNet.load_from_checkpoint('./prep_data/add_neg_version/noseen_version/lightning_logs/version_0/checkpoints/epoch=19-step=135619.ckpt', backbone=dhnet())
    # model.eval()
    # rna_embed_fn = './prep_data/7MerRna100.json.npy'
    # protein_embed_fn = './prep_data/5MerProtein128_rbp.npy'

    # # # circular rna数据集 ######################################################
    # # #proteins = os.listdir('./RBP37')
    # # #random.Random(100).shuffle(proteins)
    # # #protein_list = [(protein, protein) for protein in proteins]
    # # ##########################################################################
    # # # circular rna数据集 ######################################################
    # # # fns = os.listdir('./clip')
    # # # protein_list = [(fn.split('_')[2].upper(), fn) for fn in fns]
    # # ##########################################################################
    # # # graphprot rna数据集 ########################################################
    # fns = os.listdir('./GraphProt_CLIP_sequences')
    # protein_list = [(fn, fn) for fn in fns]
    # # ##########################################################################

    # # # 统计每个独立的test data ################################################## 
    # y_true_all, y_pred_all = [], []
    # for idx, (protein, fn) in enumerate(protein_list):
    #     #if idx < math.floor(len(protein_list) * 0.8):   continue
    #     test = dealwithdata8(protein, fn)
    #     if test == 'error':
    #         print()
    #         print("第{}个文件对应的蛋白{}不存在".format(idx, protein))
    #         continue
        
    #     print("开始处理文件:{}".format(fn))
    #     testset = DHData(rna_embed_fn, protein_embed_fn, test)
    #     testloader = DataLoader(testset, batch_size=128, shuffle=False)
    #     pred_all, label_all = [], []
    #     for i, data in enumerate(testloader):
    #         rnas, proteins, labels = data['inp1'], data['inp2'], data['label']
    #         outputs = torch.argmax(model(rnas, proteins), dim=1)
    #         preds = list(np.array(outputs))
    #         labels = list(np.array(labels))
    #         pred_all.extend(preds)
    #         label_all.extend(labels)
    #         #print("preds: {}".format(preds))
    #         #print("labels: {}".format(labels))

    #     y_true_all.extend(label_all)
    #     y_pred_all.extend(pred_all)
    #     y_true = np.array(label_all)
    #     y_pred = np.array(pred_all)

    #     accuracy = accuracy_score(y_true, y_pred)
    #     precision = precision_score(y_true, y_pred)
    #     recall = recall_score(y_true, y_pred)
    #     f1 = f1_score(y_true, y_pred)

    #     print("第{}个文件对应的蛋白{}指标如下:".format(idx, protein))
    #     print("accuracy: {}, precision: {}, recall: {}, f1: {}".format(accuracy, precision, recall, f1))

    # y_true_all = np.array(y_true_all)
    # y_pred_all = np.array(y_pred_all)
    # accuracy, precision, recall, f1 = accuracy_score(y_true_all, y_pred_all), precision_score(y_true_all, y_pred_all), recall_score(y_true_all, y_pred_all), f1_score(y_true_all, y_pred_all)
    # print("全部测集集指标如下:")
    # print("accuracy: {}, precision: {}, recall: {}, f1: {}".format(accuracy, precision, recall, f1))
    # print("全部测集集指标如下:")
    # print("accuracy: {}, precision: {}, recall: {}, f1: {}".format(accuracy, precision, recall, f1))
