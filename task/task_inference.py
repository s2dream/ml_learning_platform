from task.ABC_task_graph_model import TaskGraphModel
from abc import *


class InferenceGraphModel(TaskGraphModel):

    def __init__(self):
        pass

    def setup(self):
        pass

    def setup_model(self):
        pass

    def start_inference(self):
        pass

    def start_task(self):
        self.start_inference()
    