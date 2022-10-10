from abc import *

class ABCTask:
    def __init__(self, config):
        self.set_config(config)
        # print("init ABC_Task")


    @abstractmethod
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