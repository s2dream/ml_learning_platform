from configuration.ABC_configuration import ABCConfiguration
from utility.load_model_description import LoadArgumentDescriptionMethod
from utility.module_loader import str_to_class_module


class ModelFactory:
    @classmethod
    def create_model(cls, model_class_name, configuration:ABCConfiguration):
        arguments_desc_json = LoadArgumentDescriptionMethod.load_argument_description()
        dict_config = arguments_desc_json["model"]
        if not model_class_name in dict_config:
            raise Exception('There is no model option such as {0}'.format(model_class_name))
        module_str = dict_config[model_class_name]["module_name"]
        class_str = dict_config[model_class_name]["class_name"]
        class_module = str_to_class_module(module_str, class_str)
        return class_module(configuration)


