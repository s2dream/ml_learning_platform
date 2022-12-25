import json

class LoadArgumentDescriptionMethod:
    arg_desc_json = None
    path = "./argument_description.json"
    @classmethod
    def load_argument_description(cls):
        if cls.arg_desc_json is None:
            with open(cls.path, "r", encoding="utf-8") as fp:
                output_json = json.load(fp)
                cls.arg_desc_json = output_json
        return cls.arg_desc_json








