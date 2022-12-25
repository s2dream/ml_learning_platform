from utility.load_model_description import LoadArgumentDescriptionMethod
from utility.module_loader import str_to_class_module
class ConfigurationFactory:
    @classmethod
    def create_configuration(cls, str_type):
        arguments_desc_json = LoadArgumentDescriptionMethod.load_argument_description()
        dict_config = arguments_desc_json["configuration"]
        if not str_type in dict_config:
            raise Exception('There is no configuration option such as {0}'.format(str_type))
        module_str = dict_config[str_type]["module_name"]
        class_str = dict_config[str_type]["class_name"]
        class_module = str_to_class_module(module_str, class_str)
        return class_module()
