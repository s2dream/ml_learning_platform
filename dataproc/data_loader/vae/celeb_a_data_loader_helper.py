from typing import *
import os
from torch.utils.data import Dataset
import random
from dataproc.data_loader.vae.celeb_a_dataset import CelebADataset


RANDOM_SEED = 1
FILE_SEPERATOR = "\t"

class CelebADataLoaderHelper(Dataset):
    def __init__(self,
                 data_dir_path: str,
                 train_batch_size: int = 8,
                 val_batch_size: int = 8,
                 patch_size: Union[int, Sequence[int]] = (256, 256),
                 num_workers: int = 0,
                 pin_memory: bool = False, ):
        self.data_dir_path = data_dir_path
        self.data_eval_partition_path = os.path.join(data_dir_path, "list_eval_partition")
        self.data_attr_celeba_path = os.path.join(data_dir_path, "list_attr_celeba.txt")
        self.data_bbox_celeba_path = os.path.join(data_dir_path, "list_bbox_celeba.txt")
        self.identity_celeba_path = os.path.join(data_dir_path, "identity_CelebA.txt")
        self.list_landmarks_align_celeba_path = os.path.join(data_dir_path, "list_landmarks_align_celeba.txt")
        self.list_landmarks_celeba_path = os.path.join(data_dir_path, "list_landmarks_celeba.txt")

    def load_bbox_celeba(self, data_bbox_celeba_path):
        lines = []
        with open(data_bbox_celeba_path, "r") as fp:
            for line in fp:
                line = line.strip()
                lines.append(line)
        # total_size = int(lines[0])
        columns = lines[2].split(FILE_SEPERATOR)
        ret_dict = dict()
        for line in lines[2:]:
            data = line.split(FILE_SEPERATOR)
            data_key = data[0]
            data = [int(x) for x in data[1:]]
            data_dict = dict()
            for i, key in enumerate(columns[1:]):
                data_dict[key] = data[i]
            ret_dict[data_key]=data_dict
        return ret_dict

    def load_identity_celeba(self, identity_celeba_path):
        name_id_dict = dict()
        id_name_dict = dict()
        with open(identity_celeba_path, "r") as fp:
            for line in fp:
                line = line.strip()
                file_name, id = line.split(FILE_SEPERATOR)
                name_id_dict[file_name] = id
                id_name_dict[id] = file_name
        return name_id_dict, id_name_dict

    def load_landmarks_align_celeba(self, list_landmarks_align_celeba_path):
        lines = []
        with open(list_landmarks_align_celeba_path, "r") as fp:
            for line in fp:
                line = line.strip()
                lines.append(line)

        # total_size = int(lines[0])
        columns = lines[1].split(FILE_SEPERATOR)
        ret_dict = dict()
        for i in range(2, len(lines)):
            line = lines[i]
            data = line.split(FILE_SEPERATOR)
            data_dict = {}
            data_key = data[0]
            for j, key in enumerate(columns):
                data_dict[key] = data[j + 1]
            ret_dict[data_key] = data_dict
        return ret_dict

    def load_landmarks_celeab(self, list_landmarks_celeba_path):
        lines = []
        with open(list_landmarks_celeba_path, "r") as fp:
            for line in fp:
                line = line.strip()
                lines.append(line)

        # total_size = int(lines[0])
        columns = lines[1].split(FILE_SEPERATOR)
        ret_dict = dict()
        for i in range(2, len(lines)):
            line = lines[i]
            data = line.split(FILE_SEPERATOR)
            data_dict = {}
            data_key = data[0]
            for j, key in enumerate(columns):
                data_dict[key] = data[j + 1]
            ret_dict[data_key] = data_dict
        return ret_dict

    def load_attr_celeba(self, data_attr_celeba_path):
        lines = []
        with open(data_attr_celeba_path, "r") as fp:
            for line in fp:
                line = line.strip()
                lines.append(line)

        # total_size = int(lines[0])
        columns = lines[1].split(FILE_SEPERATOR)
        ret_dict = dict()
        for i in range(2, len(lines)):
            line = lines[i]
            data = line.split(FILE_SEPERATOR)
            data_dict = {}
            data_key = data[0]
            for j, key in enumerate(columns):
                data_dict[key] = data[j+1]
            ret_dict[data_key] = data_dict
        return ret_dict

    def load_data_eval_test_partition_path(self, data_eval_partition_path):
        list_train_data =[]
        list_valid_data = []
        list_test_data = []
        data_pool_dict = dict()
        data_pool_dict["0"] = list_train_data
        data_pool_dict["1"] = list_valid_data
        data_pool_dict["2"] = list_test_data
        with open(data_eval_partition_path, "r") as fp:
            for line in fp:
                line = line.strip()
                file_name, category = line.split(FILE_SEPERATOR)
                if category in data_pool_dict:
                    data_pool_dict[category].append(os.path.join(self.data_dir_path, file_name))
        ret = {
            "train" : data_pool_dict[0],
            "valid" : data_pool_dict[1],
            "test" : data_pool_dict[2]
        }
        return ret


    def get_dataloader(self, batch_size, shuffle=True, path=None):
        dataset = CelebADataset(path)
        dataloader = self.get_dataloader_with_dataset(dataset, batch_size, shuffle, path)
        return dataloader


    def get_dist_dataloader(self, batch_size, path, num_replicas, rank):
        dataset = CelebADataset(path)
        dataloader = self.get_dist_dataloader_with_dataset(dataset, batch_size, path, num_replicas, rank)
        return dataloader
