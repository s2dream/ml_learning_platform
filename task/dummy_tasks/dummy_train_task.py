from task.ABC_task_train import TrainTask
from configuration.config_dummy import ConfigurationTrain as Configuration
from dataproc.data_loader.dummy.dummy_data_loader import get_dataloader
from dataproc.data_loader.dummy.dummy_data_loader import get_dist_dataloader
from model.dummy_model import DummyModel2 as DummyModel
from torch.utils.tensorboard import SummaryWriter
import torch
import time


class DummyTrainTask(TrainTask):
    def __init__(self, device, config=None, dist=False, num_replica=1, rank=0):
        if config == None:
            config = Configuration()
        super().__init__(device, config, dist, num_replica, rank)
        self.cross_entropy_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    def set_summary_writer(self):
        path = self.config.get_val("summary_writer_path")
        self.writer = SummaryWriter(path)

    def get_num_epochs(self):
        num_epoch = self.config.get_val("num_epoch")
        return num_epoch

    def get_num_iterations(self):
        num_replica = 1
        if self.dist:
            num_replica = self.num_replica
        return len(self.dataset_loader) // num_replica

    def job_before_epochs_loops(self, params_dict):
        '''
         - if dist. gpu environment, then initialize the settings for distributed gpu processing
         - dataloader initializing
         - model loading
         - optimizer loading
        :param params_dict:
        :return:
        '''
        # print("job_before_epochs")
        batch_size = self.config.get_val("batch_size")
        if not self.dist:
            self.dataset_loader = get_dataloader(batch_size=batch_size)
        else:
            self.dataset_loader = get_dist_dataloader(batch_size, path=None, num_replicas=self.num_replica, rank=self.rank)
        self.model = DummyModel()
        self.model.to(self.device)
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

        num_epoch = self.get_num_epochs()
        num_iter_for_each_epoch = self.get_num_iterations()
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                                max_lr=learning_rate,
                                                                pct_start=0.3,
                                                                steps_per_epoch=num_iter_for_each_epoch,
                                                                epochs=num_epoch,
                                                                anneal_strategy='linear')

    def job_after_epochs_loops(self, params_dict):
        pass

    def job_before_iterations(self, params_dict):
        # print("job_before_iterations")
        self.iter_dataloader = iter(self.dataset_loader)


    def job_for_each_iteration(self, params_dict, cur_iter_in_an_epoch, cur_epoch):
        start_time = time.time()
        # print("job_for_each_iteration:{0}".format(cur_iter))
        data = next(self.iter_dataloader)
        idx = data["idx"]
        input = data["input"]
        label = data["label"]


        self.optimizer.zero_grad()
        logit = self.model(input.to(self.device))
        loss = self.compute_loss(logit, label)

        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        if cur_iter_in_an_epoch > 0 and cur_iter_in_an_epoch % 100 == 0:
            total_iter = self.get_num_iterations()
            acc = self.compute_acc(logit, label)
            elapsed_time = time.time() - start_time
            cur_lr = self.lr_scheduler.get_lr()
            cur_lr = cur_lr[0]
            if not self.dist or (self.dist and self.rank==0):
                print("[epoch:{0},iter:{1}/{2}] loss:{3}, accuracy:{4}, lr:{5}, elapsed_time(iter):{6}".format(cur_epoch,
                                                                                                           cur_iter_in_an_epoch,
                                                                                                           total_iter,
                                                                                                           loss,
                                                                                                           acc,
                                                                                                           cur_lr,
                                                                                                           elapsed_time))
    def compute_loss(self, logit, label):
        return self.cross_entropy_loss_fn(logit, label.to(torch.long))

    def compute_acc(self, logit, label):
        logit_arg_max = torch.argmax(logit, dim=1)
        total_size = logit.size(dim=0)
        eq_result = torch.eq(logit_arg_max, label).to(torch.int32)
        num_correct = torch.sum(eq_result).item()
        acc = float(num_correct) / float(total_size)
        return acc

    def job_after_iterations(self, params_dict):
        pass


if __name__ == "__main__":
    d = DummyTrainTask()
    d.start_train()
