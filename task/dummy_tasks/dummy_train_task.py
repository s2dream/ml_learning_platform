from task.ABC_task_train import TrainTask
from configuration.config_dummy import ConfigurationTrain as Configuration
from dataproc.data_loader.dummy.dummy_data_loader import get_dataloader
from model.dummy_model import DummyModel
import torch
import time


class DummyTrainTask(TrainTask):

    def __init__(self, config=None):
        config = Configuration()
        super().__init__(config)

    def get_num_epochs(self):
        print("get_num_epochs")
        num_epoch = self.config.get_val("num_epoch")
        return num_epoch

    def get_num_iterations(self):
        print("get_num_iterations")
        return len(self.dataset_loader)

    def job_before_epochs_loops(self, params_dict):
        '''
         - if dist. gpu environment, then initialize the settings for distributed gpu processing
         - dataloader initializing
         - model loading
         - optimizer loading
        :param params_dict:
        :return:
        '''
        print("job_before_epochs")
        batch_size = self.config.get_val("batch_size")
        self.dataset_loader = get_dataloader(batch_size=batch_size)
        self.model = DummyModel()
        self.set_adam_optimizer(self.model)

    def set_adam_optimizer(self, model):
        learning_rate = self.config.get_val("lr")
        adam_beta_1 = self.config.get_val("adam_beta_1")
        adam_beta_2 = self.config.get_val("adam_beta_2")
        epsilon = self.config.get_val("epsilon")
        weight_decay = self.config.get_val("weight_decay")
        self.optimizer = torch.optim.AdamW(model.parameters(),
                                           lr=learning_rate,
                                           betas=(adam_beta_1, adam_beta_2),
                                           eps=epsilon,
                                           weight_decay=weight_decay)

    def job_after_epochs_loops(self, params_dict):
        print("job_after_epochs")

    def job_before_iterations(self, params_dict):
        print("job_before_iterations")
        self.iter_dataloader = iter(self.dataset_loader)

    def job_for_each_iteration(self, params_dict, cur_iter, cur_epoch):
        start_time = time.time()
        print("job_for_each_iteration:{0}".format(cur_iter))
        data = next(self.iter_dataloader)
        idx = data["idx"]
        input = data["input"]
        label = data["label"]

        logit = self.model(input)
        loss = self.compute_loss(logit, label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # total_iter = self.get_num_iterations()
        if cur_iter>0 and cur_iter % 1000 == 0:
            total_iter = self.get_num_iterations()
            acc = self.compute_acc
            elapsed_time = time.time() - start_time
            print("[epoch:{0},iter:{1}/{2}] loss:{3}, accuracy:{4}, elapsed_time(iter):{5}".format(cur_epoch,
                                                                                                   cur_iter,
                                                                                                   total_iter,
                                                                                                   loss,
                                                                                                   acc,
                                                                                                   elapsed_time))
            
    def compute_loss(self, logit, label):
        return 0.0

    def compute_acc(self, logit, label):
        return 0.0

    def job_after_iterations(self, params_dict):
        print("job_after_iterations")


if __name__ == "__main__":
    d = DummyTrainTask()
    d.start_train()
