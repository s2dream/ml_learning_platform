from configuration.config_GCN import ConfigurationGCN
from model.model_factory import ModelFactory
from model.model_enum import ModelE
from task.task_inference import InferenceGraphModel


class InferenceGCN(InferenceGraphModel):
    def __init__(self, configuration=None):
        if configuration == None:
            self.config = ConfigurationGCN()
        self.setup()

    def setup(self):
        pass

    def setup_model(self):
        self.model = ModelFactory.create_model(ModelE.GCN, self.config)

    def start_train(self):
        pass






