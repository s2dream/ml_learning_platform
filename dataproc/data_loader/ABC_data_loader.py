from os import path
import sys
import pickle as pkl

import torch.utils.data
from torch.utils.data import Dataset, DataLoader


class PickleDataset(Dataset):
    def __init__(self, bin_file_path, idx_file_path):
        if not path.exists(bin_file_path):
            print("bin_file_path not exists. check -> "+str(bin_file_path))
            sys.exist(1)
        if not path.exists(idx_file_path):
            print("idx_file_path not exsits. check -> "+str(idx_file_path))
        self.bin_data_file = open(bin_file_path,"rb")
        self.idx_list = []
        with open(idx_file_path, "rb") as fp:
            try:
                data_index = pkl.load(fp)
                self.idx_list.append(data_index)
            except:
                pass
        print("{0} of data in bin file".format(len(self.idx_list)))

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        if idx >= len(self.idx_list):
            return None
        position = self.idx_list[idx]
        self.bin_data_file.seek(position)
        data = pkl.load(self.bin_data_file)
        return data

def get_dataloader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_distributed_dataloader(dataset, num_replicas, rank, batch_size, shuffle=True):
    def customized_collated_fn(): # 참조: https://velog.io/@jiyoung/torch-collatefn%EC%97%90-arguments-%EC%B6%94%EA%B0%80%ED%95%98%EA%B8%B0
        pass

    # sampler and batch sampler are mutually exclusive
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle )
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=customized_collated_fn, shuffle=shuffle, sampler=sampler)
    return dataloader

