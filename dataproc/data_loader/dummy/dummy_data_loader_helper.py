import torch
import torch.utils.data

from dataproc.data_loader.ABC_data_loader_helper import ABCDataLoaderHelper
from dataproc.data_loader.dummy.dummy_dataset import DummyDataset

class DummyDataLoaderHelper(ABCDataLoaderHelper):
    def collate_fn(self, batch):
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

    def get_dataloader(self, batch_size, shuffle=True, path=None):
        dataset = DummyDataset(path)
        dataloader = self.get_dataloader_with_dataset(dataset, batch_size, shuffle, path)
        return dataloader


    def get_dist_dataloader(self, batch_size, path, num_replicas, rank):
        dataset = DummyDataset(path)
        dataloader = self.get_dist_dataloader_with_dataset(dataset, batch_size, path, num_replicas, rank)
        return dataloader