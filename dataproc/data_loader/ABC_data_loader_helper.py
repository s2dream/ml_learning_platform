from abc import *

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import BatchSampler

class ABCDataLoaderHelper:
    @abstractmethod
    def collate_fn(self, batch):
        pass

    @abstractmethod
    def get_dataloader(self,  batch_size, shuffle=True, path=None):
        pass

    @abstractmethod
    def get_dist_dataloader(self, batch_size, path, num_replicas, rank):
        pass

    def get_dataloader_with_dataset(self, dataset, batch_size, shuffle=True, path=None):
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)
        return dataloader

    def get_dist_dataloader_with_dataset(self, dataset, batch_size, path, num_replicas, rank):
        dist_sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank)
        sampler = BatchSampler(dist_sampler, batch_size=batch_size, drop_last=True)
        dataloader = DataLoader(dataset, collate_fn=self.collate_fn, batch_sampler=sampler)
        return dataloader