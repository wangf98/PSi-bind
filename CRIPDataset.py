from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from prep_data.getData import dealwithdata


class CRIPData(Dataset):

    def __init__(self, feature, label, rna_embed_fn):
        super().__init__()
        self.rna_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(
                np.load(rna_embed_fn)
            )
        )
        self.feature = torch.from_numpy(feature).type(torch.FloatTensor)
        self.label = torch.from_numpy(label)

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, item):
        feature = self.feature[item]
        label = self.label[item]

        return {
            'feature': feature,
            'label': label
        }


if __name__ == '__main__':
    # print(getEmbedding())
    # data = np.load('../prep_data/npz_file_crip/AGO1_data.npz')
    # trainset = CRIPData(data['train_X'], data['train_y'])
    # tng_dataloader = DataLoader(trainset, batch_size=32)
    # for i, batch in enumerate(tng_dataloader):
    #     feature = batch['feature']
    #     print(feature.size())
    #     break

    train_X, _, train_y, _ = dealwithdata('AGO1')
    trainset = CRIPData(train_X, train_y)
    tng_dataloader = DataLoader(trainset, batch_size=32)
    for i, batch in enumerate(tng_dataloader):
        feature = batch['feature']
        print(feature.size())
