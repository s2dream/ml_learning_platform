from task.ABC_task_inference import ABCTestTask
from configuration.dummy.config_dummy import DummyConfigurationTrain as Configuration


class DummyInferenceTask(ABCTestTask):

    def __init__(self, device, config=None):
        if config == None:
            config = Configuration()
        super().__init__(device, config)
        print("init DummyInferenceTask")

    def get_num_iterations(self):
        print("get_num_iterations")
        return 100

    def job_before_iterations(self, params_dict):
        print("job_after_epochs")

    def job_after_iterations(self, params_dict):
        print("job_after_iterations")

    def job_for_each_iteration(self, params_dict):
        print("job_for_each_iteration")

if __name__ == "__main__":
    a = DummyInferenceTask(None)