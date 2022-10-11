from abc import *

class ABCTask:
    def __init__(self, device=None, config=None):
        self.set_config(config)
        self.set_device(device)

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