import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os

from configuration.configuration import Configuration
from utility.parse_arguments import ParseArgs
from model.model_enum import ModelE

from task.dummy_tasks.dummy_train_task import DummyTrainTask
from task.dummy_tasks.dummy_test_task import DummyInferenceTask

CPU = "cpu"
GPU = "cuda"


#
# def get_model_class(type_str:str, alias_to_class):
#     if type_str in alias_to_class:
#         return alias_to_class[type_str]
#     else:
#         return None

def get_task(type_str:str, model_name:str, configruation:Configuration=None, device='cpu', dist=False, rank=0, num_replica=1):
    if type_str.lower() == "train":
        if model_name == "dummy":
            return DummyTrainTask(device, config=configruation, dist=dist, rank=rank, num_replica=num_replica)
    else:
        if model_name == "dummy":
            return DummyInferenceTask(device)
    return None

class Main:
    def __init__(self):
        self.args = ParseArgs.parse_arguments()
        self.model_name = self.args.model ## model name
        self.task_mode = self.args.mode ## training, test

        print(str(torch.cuda.device(0)))
        print(str(torch.cuda.device_count()))
        if torch.cuda.is_available():
            world_size = torch.cuda.device_count()
            if self.args.dist and world_size > 1:
                self.set_distributed_gpus()
            else:
                self.set_single_gpu()
        else:
            # print(str(torch.cuda.get_device_name(0)))
            self.set_cpu_environment()

    def set_single_gpu(self):
        self.num_of_devices = 1
        self.device_type = GPU

    def setup_dist_environ(self, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '5001'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def set_distributed_gpus(self):
        self.num_of_devices = torch.cuda.device_count()
        self.device_type = GPU

    def set_cpu_environment(self):
        self.device_type = CPU

    def launch_dist(self, rank, world_size):
        self.setup_dist_environ(rank, world_size)
        # model_type = get_model_type(self.model_name)
        task = get_task(self.task_mode, self.model_name, None, device=rank)
        # task.setup_model(device=rank, dist=True)
        task.start_task()
        self.cleanup_dist()

    def cleanup_dist(self):
        dist.destroy_process_group()

    def launch_single(self, device):
        task = get_task(self.task_mode, self.model_name, None, device=device)
        # task.setup_model(device=device)
        task.start_task()

    def launch(self):
        if torch.cuda.is_available():
            world_size = self.num_of_devices
            if self.args.dist and world_size > 1:
                mp.spawn(self.launch_dist,
                         args=(world_size,),
                         nprocs=world_size,
                         join=True)
                return
            else:
                device_name = GPU
        else:
            device_name = CPU
        self.launch_single(torch.device(device_name))


if __name__ == '__main__':
    main = Main()
    main.launch()