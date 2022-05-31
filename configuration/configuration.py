class Configuration:

    def __init__(self):
        self.dict =     dict()

    def get_value(self, key):
        return self.dict[key] if key in self.dict else None

    def set_value(self, key, value):
        self.dict[key] = value

