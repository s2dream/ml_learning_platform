import os
import importlib
import json
from log_module.ml_logger import MLLogger

def str_to_class_module(module_name, class_name):
    """Return a class instance from a string reference"""
    class_ = None
    logger = MLLogger.get_logger()
    try:
        module_ = importlib.import_module(module_name)
        # print(module_)
        try:
            class_ = getattr(module_, class_name)
        except AttributeError as e1:
            logger.error("[Error in str_to_class(class)]")
            logger.error(e1)
    except ImportError as e2:
        logger.error("[Error in str_to_class(module)]")
        logger.error(e2)
    return class_ or None