from task.ABC_task_graph_model import TaskGraphModel
from abc import *


class TrainGraphModel(TaskGraphModel):

    def __init__(self):
        self.config = None

    def setup(self):
        pass

    def setup_model(self):
        pass

    @abstractmethod
    def get_num_epochs(self):
        pass

    @abstractmethod
    def get_num_iterations(self):
        pass

    @abstractmethod
    def job_before_epochs(self, params_dict):
        pass

    @abstractmethod
    def job_after_epochs(self, params_dict):
        pass

    @abstractmethod
    def job_before_iterations(self, params_dict):
        pass

    @abstractmethod
    def job_after_iterations(self, params_dict):
        pass


    @abstractmethod
    def job_for_each_iteration(self, params_dict):
        pass

    def start_train(self):
        num_epoch = self.get_num_epochs()
        if num_epoch is None:
            raise Exception('define the number of epochs for learning')
        params_dict = dict()
        params_dict = self.job_before_epochs(params_dict)
        for epoch in range(num_epoch):
            num_iterations = self.get_num_iterations()
            params_dict = self.job_before_iterations(params_dict)
            for iteration in range(num_iterations):
                params_dict = self.job_for_each_iteration(params_dict)
            params_dict = self.job_after_iterations(params_dict)
        self.job_after_epochs(params_dict)

    def start_task(self):
        self.start_train()
