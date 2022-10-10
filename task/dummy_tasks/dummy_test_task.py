from task.ABC_task_inference import InferenceTask


class DummyInferenceTask(InferenceTask):



    def __init__(self,config):
        super().__init__(config)
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