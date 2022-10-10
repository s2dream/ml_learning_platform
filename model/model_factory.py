from model.model_enum import ModelE
from configuration.configuration import Configuration
from model.dummy_model import DummyModel


class ModelFactory:


    @classmethod
    def create_model(cls, model_enum:ModelE, configuration:Configuration):
        pass
        if model_enum == ModelE.DUMMY_MODEL:
            return DummyModel(configuration)
        # if model_enum == ModelE.GCN:
        #     return GCN(configuration)
        # elif model_enum == ModelE.GAT:
        #     return GAT(configuration)
        # elif model_enum == ModelE.GRAPHOMER:
        #     return Graphomer(configuration)







