from os import path
import sys
import pickle as pkl
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import random

class DummyDataset(Dataset):
    def __init__(self, datasize= 8*8*1024, data_dim = 128 ):
        # self.data_tensor = torch.nn.rand(datasize, data_dim, dtype=torch.float32)
        self.dataset = [[random.random() for _ in range(data_dim)] for i in range(datasize)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, self.dataset[idx])


def dummy_collate_fn(batch):
    idx = []
    input = []
    label = []
    for data in batch:
        idx.append(data[0])
        input.append(data[1])
        label_of_input = int(int(sum(data[1])) // 30)
        label.append(label_of_input)
    idx = torch.tensor(idx, dtype=torch.int)
    input_tensor = torch.tensor(input, dtype=torch.float32)
    label_tensor = torch.tensor(label, dtype=torch.int)
    output_dict = dict()
    output_dict["idx"] = idx
    output_dict["input"] = input_tensor
    output_dict["label"] = label_tensor
    return output_dict

def get_dataloader(batch_size, shuffle=True):
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dummy_collate_fn, shuffle=shuffle)
    return dataloader


def test():
    dataloader = get_dataloader(20)
    iter_dataloader = iter(dataloader)
    print(len(dataloader))
    for _ in range(len(dataloader)):
        data = next(iter_dataloader)
        print(data["idx"])

    print("----------------------")
    iter_dataloader = iter(dataloader)
    for _ in range(len(dataloader)):
        data = next(iter_dataloader)
        print(data["idx"])

if __name__ == "__main__":
    test()