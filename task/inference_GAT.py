from configuration.config_GAT import ConfigurationGAT
from model.model_factory import ModelFactory
from model.model_enum import ModelE
from task.task_inference import InferenceGraphModel

class InferenceGAT(InferenceGraphModel):
    def __init__(self, configuration=None):
        super().__init__()
        if configuration == None:
            self.config = ConfigurationGAT()
        self.setup()

    def setup(self):
        pass

    def setup_model(self):
        self.model = ModelFactory.create_model(ModelE.GAT, self.config)

    def start_inference(self):
        ## todo
        pass






