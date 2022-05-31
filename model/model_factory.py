from model.model_enum import ModelE
from model.model_GCN import GCN
from model.model_GAT import GAT
from model.model_Graphomer import Graphomer
from configuration.configuration import Configuration



class ModelFactory:
    @classmethod
    def create_model(cls, model_enum:ModelE, configuration:Configuration):
        if model_enum == ModelE.GCN:
            return GCN(configuration)
        elif model_enum == ModelE.GAT:
            return GAT(configuration)
        elif model_enum == ModelE.GRAPHOMER:
            return Graphomer(configuration)
