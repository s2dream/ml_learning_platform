
class ABCConfiguration:
    def __init__(self):
        pass

    def get_val(self, key):
        if key in self.conf_dict:
            return self.conf_dict[key]

        return None

    def set_val(self, key, val):
        self.conf_dict[key] = val

