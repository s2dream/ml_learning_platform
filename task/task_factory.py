from configuration.ABC_configuration import ABCConfiguration
from utility.load_model_description import LoadArgumentDescriptionMethod
from utility.module_loader import str_to_class_module
from log_module.ml_logger import MLLogger
logger = MLLogger.get_logger()

class TaskFactory:
    @classmethod
    def create_task(cls, str_type, device, dist=False, num_replica=1, rank=0, args=None):
        arguments_desc_json = LoadArgumentDescriptionMethod.load_argument_description()
        dict_config = arguments_desc_json["task"]
        if not str_type in dict_config:
            raise Exception('There is no task option such as {0}'.format(str_type))
        module_str = dict_config[str_type]["module_name"]
        class_str = dict_config[str_type]["class_name"]
        logger.info(module_str,"/", class_str)
        class_module = str_to_class_module(module_str, class_str)
        return class_module(device, dist=dist, num_replica=num_replica, rank=rank, args=args)
