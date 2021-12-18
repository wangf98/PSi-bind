from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
#import pandas as pd
import numpy as np
import json
import os
from prep_data.getData import dealwithdata3


class DHData(Dataset):

    def __init__(self, rna_embed_fn, protein_embed_fn, dataset):

        self.rna_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(
                np.load(rna_embed_fn)
            )
        )
        self.protein_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(
                np.load(protein_embed_fn)
            )
        )
        self.dataset = dataset

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, item):

        # feature = self.feature[item]
        # feature = self.rna_embedding(feature)
        inp1, inp2, label = self.dataset[item]
        inp1, inp2, label = torch.from_numpy(np.array(inp1)), torch.from_numpy(np.array(inp2)), torch.tensor(label)
        inp1 = self.rna_embedding(inp1)
        inp2 = self.protein_embedding(inp2)

        return {
            'inp1': inp1,
            'inp2': inp2,
            'label': label
        }


if __name__ == '__main__':

    train_X, _, train_y, _ = dealwithdata3('AGO1')
    trainset = DHData('./prep_data/7MerRna100.json.npy', './prep_data/5MerProtein128_rbp.npy', train_X, train_y)
    tng_dataloader = DataLoader(trainset, batch_size=32)
    for i, batch in enumerate(tng_dataloader):
        feature = batch['inp2']
        print(feature.size())