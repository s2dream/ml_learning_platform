

class ConfigurationTrain:
    def __init__(self):
        self.conf_dict = dict()
        self.conf_dict["num_epoch"] = 30
        self.conf_dict["batch_size"] = 128
        self.conf_dict["train_data_dir_path"] = ""
        self.conf_dict["test_data_dir_path"] = ""

        self.conf_dict["lr"] = 0.00001
        self.conf_dict["adam_beta_1"] = 0.9
        self.conf_dict["adam_beta_2"] = 0.999
        self.conf_dict["epsilon"] = 1e-8
        self.conf_dict["weight_decay"] = 1e-2


    def get_val(self, key):
        if key in self.conf_dict:
            return self.conf_dict[key]

        return None

    def set_val(self, key, val):
        self.conf_dict[key] = val
