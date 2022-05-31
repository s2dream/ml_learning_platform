from configuration.config_Graphomer import ConfigurationGraphomer
from model.model_factory import ModelFactory
from model.model_enum import ModelE
from task.task_inference import InferenceGraphModel


class InferenceGraphomer(InferenceGraphModel):
    def __init__(self, configuration=None):
        if configuration == None:
            self.config = ConfigurationGraphomer()
        self.setup()

    def setup(self):
        pass

    def setup_model(self):
        self.model = ModelFactory.create_model(ModelE.GRAPHOMER, self.config)

    def start_Inference(self):
        pass






