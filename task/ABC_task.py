from abc import *


class ABCTask:
    def __init__(self, device=None, config=None, args=None):
        self.set_config(config)
        self.set_device(device)
        self.set_args(args)
        self.set_model_name(self.args.model)
        self.set_dataloader_name(self.args.dataloader)

    def get_model_name(self):
        return self.model_name

    def get_dataloader_name(self):
        return self.dataloader_name

    def set_model_name(self, model_name):
        self.model_name = model_name

    def set_dataloader_name(self, dataloader_name):
        self.dataloader_name = dataloader_name

    def set_args(self, args):
        self.args = args

    def set_device(self, device):
        self.device = device

    def set_config(self, config):
        self.config = config

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def start_task(self):
        pass
