from task.ABC_task import ABCTask
from abc import *


class TrainTask(ABCTask):

    def __init__(self, config):
        super().__init__(config)

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
