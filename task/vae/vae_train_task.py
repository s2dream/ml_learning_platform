from task.ABC_task_train import ABCTrainTask
from log_module.ml_logger import MLLogger

logger = MLLogger.get_logger()

class VAETrainTask(ABCTrainTask):

    def __init__(self):
        pass

    def set_summary_writer(self):
        logger.info("set_summary_writer")

    def get_num_epochs(self):
        logger.info("get_num_epochs")

    def get_num_iterations(self):
        logger.info("get_num_iterations")

    def job_before_epochs_loops(self, params_dict):
        logger.info("job_before_epochs")

    def job_after_epochs_loops(self, params_dict):
        logger.info("job_after_epochs")

    def job_before_iterations(self, params_dict):
        logger.info("job_before_iterations")

    def job_after_iterations(self, params_dict):
        logger.info("job_after_iterations")

    def job_for_each_iteration(self, params_dict, cur_iter_in_an_epoch, cur_epoch):
        logger.info("job_for_each_iteration")
