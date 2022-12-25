import os
import importlib
import json

def str_to_class_module(module_name, class_name):
    """Return a class instance from a string reference"""
    class_ = None
    try:
        module_ = importlib.import_module(module_name)
        try:
            class_ = getattr(module_, class_name)
        except AttributeError as e1:
            print("[Error in str_to_class(class)]")
            print(e1)
    except ImportError as e2:
        print("[Error in str_to_class(module)]")
        print(e2)
    return class_ or None