from task.ABC_task_inference import ABCTestTask
from configuration.dummy.config_dummy import DummyConfigurationTrain as Configuration
from log_module.ml_logger import MLLogger
logger = MLLogger.get_logger()

class DummyInferenceTask(ABCTestTask):

    def __init__(self, device, config=None):
        if config == None:
            config = Configuration()
        super().__init__(device, config)
        logger.info("init DummyInferenceTask")

    def get_num_iterations(self):
        logger.info("get_num_iterations")
        return 100

    def job_before_iterations(self, params_dict):
        logger.info("job_after_epochs")

    def job_after_iterations(self, params_dict):
        logger.info("job_after_iterations")

    def job_for_each_iteration(self, params_dict):
        logger.info("job_for_each_iteration")

if __name__ == "__main__":
    a = DummyInferenceTask(None)