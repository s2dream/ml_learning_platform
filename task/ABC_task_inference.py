from task.ABC_task import ABCTask
from abc import *
from log_module.ml_logger import MLLogger
logger = MLLogger.get_logger()

class ABCTestTask(ABCTask):
    def __init__(self,device, config):
        super().__init__(device, config)
        logger.info("init InferenceTask")

    @abstractmethod
    def get_num_iterations(self):
        logger.info("get_num_iterations")

    @abstractmethod
    def job_before_iterations(self, params_dict):
        logger.info("job_after_epochs")

    @abstractmethod
    def job_after_iterations(self, params_dict):
        logger.info("job_after_iterations")

    @abstractmethod
    def job_for_each_iteration(self, params_dict):
        logger.info("job_for_each_iteration")

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
