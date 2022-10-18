from task.ABC_task import ABCTask
from abc import *
from torch.utils.tensorboard import SummaryWriter
import torch


class TrainTask(ABCTask):

    def __init__(self, device, config, dist=False, num_replica=1, rank=0):
        super().__init__(device, config)
        self.dist=dist
        self.num_replica = num_replica
        self.rank = rank
        self.writer = None

    @abstractmethod
    def set_summary_writer(self):
        print("set_summary_writer")

    @abstractmethod
    def get_num_epochs(self):
        print("get_num_epochs")

    @abstractmethod
    def get_num_iterations(self):
        print("get_num_iterations")

    @abstractmethod
    def job_before_epochs_loops(self, params_dict):
        print("job_before_epochs")

    @abstractmethod
    def job_after_epochs_loops(self, params_dict):
        print("job_after_epochs")

    @abstractmethod
    def job_before_iterations(self, params_dict):
        print("job_before_iterations")

    @abstractmethod
    def job_after_iterations(self, params_dict):
        print("job_after_iterations")

    @abstractmethod
    def job_for_each_iteration(self, params_dict, cur_iter_in_an_epoch, cur_epoch):
        print("job_for_each_iteration")

    def start_train(self):
        num_epoch = self.get_num_epochs()
        if num_epoch is None:
            raise Exception('define the number of epochs for learning')
        params_dict = dict()
        params_dict = self.job_before_epochs_loops(params_dict)
        for epoch in range(num_epoch):
            num_iterations = self.get_num_iterations()
            params_dict = self.job_before_iterations(params_dict)
            for cur_iter, iteration in enumerate(range(num_iterations)):
                params_dict = self.job_for_each_iteration(params_dict, cur_iter, epoch)
            params_dict = self.job_after_iterations(params_dict)
        self.job_after_epochs_loops(params_dict)
        return params_dict

    def start_task(self):
        self.start_train()

    def save_total_ckpt(self, model, opt, cur_epoch, cur_iter, save_path):
        params = dict()
        params["model"] = model.state_dict()
        params["optimizer"] = opt.state_dict()
        params["epoch"] = cur_epoch
        params["iteration"] = cur_iter
        torch.save(params, save_path)

    def load_total_ckpt(self, model, opt, load_path):
        params = torch.load(load_path)
        model.load_state_dict(params["model"])
        opt.load_state_dict(params["optimizer"])
        epoch = params["epoch"]
        iteration = params
        return epoch, iteration

    def save_model_ckpt(self, model, save_path):
        torch.save(model.state_dict(), save_path)

    def save_optimizer_ckpt(self, opt, save_path):
        torch.save(opt.state_dict(), save_path)

    def load_model_ckpt(self, model, load_path):
        loaded_params = torch.load(load_path)
        model.load_state_dict(loaded_params)

    def load_optimizer_ckpt(self, opt, load_path):
        loaded_params = torch.load(load_path)
        opt.load_state_dict(loaded_params)

    def summary_add_scalar(self,tag,value,global_step):
        if self.writer is not None:
            self.writer.add_scalar(tag=tag, value=value, global_step=global_step)