from configuration.configuration_factory import ConfigurationFactory
from model.model_factory import ModelFactory
from task.ABC_task_train import ABCTrainTask
from dataproc.data_loader.dummy.dummy_data_loader_helper import DummyDataLoaderHelper
from torch.utils.tensorboard import SummaryWriter
import torch
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from log_module.ml_logger import MLLogger
logger = MLLogger.get_logger()

wandb.init(project="dummy_project", entity="s2dream")


class DummyTrainTask(ABCTrainTask):
    def __init__(self, device, dist=False, num_replica=1, rank=0, args=None):
        task_name = args.task
        config = ConfigurationFactory.create_configuration(task_name)
        super().__init__(device, config, dist, num_replica, rank, args)
        self.cross_entropy_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.set_wandb()

    def set_wandb(self):
        batch_size = self.config.get_val("batch_size")
        learning_rate = self.config.get_val("lr")
        epochs = self.config.get_val("num_epoch")
        wandb.config = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size
        }

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
        self.start_time = time.time()
        dummy_dataloader_helper = DummyDataLoaderHelper()
        batch_size = self.config.get_val("batch_size")
        if not self.dist:
            self.dataset_loader = dummy_dataloader_helper.get_dataloader(batch_size=batch_size)
        else:
            self.dataset_loader = dummy_dataloader_helper.get_dist_dataloader(batch_size, path=None, num_replicas=self.num_replica, rank=self.rank)
        # self.model = DummyModel()
        self.model = ModelFactory.create_model(self.model_name, self.config)
        self.model.to(self.device)
        if self.dist:
            self.model = DDP(self.model, device_ids=[self.rank])
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

    def set_dataloader_epoch(self):
        self.dataset_loader.set_epoch()

    def job_after_epochs_loops(self, params_dict):
        self.end_time = time.time()
        logger.info("total elapsed time:{0}".format(self.end_time-self.start_time))
        self.save_total_ckpt()

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

        #wandb
        wandb.log({"loss": loss,
                   "accuracy": self.compute_acc(logit, label)})

        if cur_iter_in_an_epoch > 0 and cur_iter_in_an_epoch % 100 == 0:
            total_iter = self.get_num_iterations()
            acc = self.compute_acc(logit, label)
            elapsed_time = time.time() - start_time
            cur_lr = self.lr_scheduler.get_lr()
            cur_lr = cur_lr[0]
            if not self.dist or (self.dist and self.rank==0):
                logger.info("[epoch:{0},iter:{1}/{2}] loss:{3}, accuracy:{4}, lr:{5}, elapsed_time(iter):{6}".format(cur_epoch,
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
