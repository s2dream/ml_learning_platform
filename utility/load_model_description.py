import os
import importlib

def str_to_class(module_name, class_name):
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

def load_model_descriptions(path):
    if os.path.exists(path):
        alias_to_class= dict()
        with open(path, "r") as fp:
            for line in fp:
                line = line.strip()
                elems = line.split("\t")
                alias = elems[0].strip()
                modulename = elems[1].strip()
                classname = elems[2].strip()
                var_class = str_to_class(modulename, classname)
                if var_class is not None:
                    alias_to_class[alias] = var_class
        return alias_to_class
    else:
        return None



