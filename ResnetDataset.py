from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import os
from prep_data.getData import dealwithdata2


class ResnetData(Dataset):

    def __init__(self, rna_embed_fn, feature, label):

        self.rna_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(
                np.load(rna_embed_fn)
            )
        )

        self.feature = torch.from_numpy(feature)
        self.label = torch.from_numpy(label)

    def __len__(self):

        return len(self.feature)

    def __getitem__(self, item):

        feature = self.feature[item]
        feature = self.rna_embedding(feature)
        label = self.label[item]

        return {
            'feature': feature,
            'label': label
        }


if __name__ == '__main__':

    train_X, _, train_y, _ = dealwithdata2('AGO1')
    trainset = ResnetData('./prep_data/7MerRna100.json.npy', train_X, train_y)
    tng_dataloader = DataLoader(trainset, batch_size=32)
    for i, batch in enumerate(tng_dataloader):
        feature = batch['feature']
        print(feature.size())