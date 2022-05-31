from abc import *


class TaskGraphModel(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def start_task(self):
        pass


