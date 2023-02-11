from task.ABC_task_train import TrainTask


class VAETrainTask(TrainTask):

    def __init__(self):
        pass

    def set_summary_writer(self):
        print("set_summary_writer")

    def get_num_epochs(self):
        print("get_num_epochs")

    def get_num_iterations(self):
        print("get_num_iterations")

    def job_before_epochs_loops(self, params_dict):
        print("job_before_epochs")

    def job_after_epochs_loops(self, params_dict):
        print("job_after_epochs")

    def job_before_iterations(self, params_dict):
        print("job_before_iterations")

    def job_after_iterations(self, params_dict):
        print("job_after_iterations")

    def job_for_each_iteration(self, params_dict, cur_iter_in_an_epoch, cur_epoch):
        print("job_for_each_iteration")
