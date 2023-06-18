import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os

from task.task_factory import TaskFactory
from utility.parse_arguments import ParseArgs

from log_module.ml_logger import MLLogger

CPU = "cpu"
GPU = "cuda"
logger = MLLogger.get_logger()

class Main:
    def __init__(self):
        self.args = ParseArgs.parse_arguments()
        self.task_name = self.args.task
        logger.info(str(torch.cuda.device(0)))
        logger.info(str(torch.cuda.device_count()))
        if torch.cuda.is_available():
            world_size = torch.cuda.device_count()
            if self.args.dist and world_size > 1:
                self.set_distributed_gpus()
            else:
                self.set_single_gpu()
        else:
            # print(str(torch.cuda.get_device_name(0)))
            self.set_cpu_environment()

    def get_task(self, task_name, device, dist=False, num_replica=1, rank=0):
        return TaskFactory.create_task(task_name, device=device, dist=dist, num_replica=num_replica, rank=rank, args=self.args)

    def set_single_gpu(self):
        self.num_of_devices = 1
        self.device_type = GPU

    def setup_dist_environ(self, rank, world_size):
        torch.cuda.set_device(rank)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50001'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def set_distributed_gpus(self):
        self.num_of_devices = torch.cuda.device_count()
        self.device_type = GPU

    def set_cpu_environment(self):
        self.device_type = CPU

    def launch_dist(self, rank, world_size):
        self.setup_dist_environ(rank, world_size)
        task = self.get_task(self.task_name, device=rank, dist=dist, num_replica=world_size, rank=rank)
        task.start_task()
        self.cleanup_dist()

    def cleanup_dist(self):
        dist.destroy_process_group()

    def launch_single(self, device):
        task = self.get_task(self.task_name, device=device)
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
