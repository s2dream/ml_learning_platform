from torch.utils.data import Dataset
import random

class DummyDataset(Dataset):
    def __init__(self, path=None, datasize= 8*8*8*1024, data_dim = 128 ):
        self.dataset = [[random.random() for _ in range(data_dim)] for i in range(datasize)]
        if path is not None:
            pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, self.dataset[idx])