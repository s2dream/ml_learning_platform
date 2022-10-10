import importlib

def str_to_class(module_name, class_name):
    """Return a class instance from a string reference"""
    class_ = None
    try:
        module_ = importlib.import_module(module_name)
        try:
            class_ = getattr(module_, class_name)
        except AttributeError as e:
            print(e)
    except ImportError:
        print('Module does not exist')
    return class_ or None


module_name = "model.dummy_model"
class_name = "DummyModel"

result = str_to_class(module_name, class_name)

print(type(result))
print(type(result()))





