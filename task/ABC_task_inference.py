from task.ABC_task import ABCTask
from abc import *


class InferenceTask(ABCTask):




    def __init__(self,config):
        super().__init__(config)
        print("init InferenceTask")


    @abstractmethod
    def get_num_iterations(self):
        print("get_num_iterations")

    @abstractmethod
    def job_before_iterations(self, params_dict):
        print("job_after_epochs")

    @abstractmethod
    def job_after_iterations(self, params_dict):
        print("job_after_iterations")

    @abstractmethod
    def job_for_each_iteration(self, params_dict):
        print("job_for_each_iteration")


    def start_inference(self):
        params_dict = dict()
        num_iterations = self.get_num_iterations()
        params_dict = self.job_before_iterations(params_dict)
        for iteration in range(num_iterations):
            params_dict = self.job_for_each_iteration(params_dict)
        params_dict = self.job_after_iterations(params_dict)
        return params_dict

    def start_task(self):
        self.start_inference()
    