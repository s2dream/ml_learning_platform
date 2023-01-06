from configuration.ABC_configuration import ABCConfiguration

class DummyConfigurationTrain(ABCConfiguration):
    def __init__(self):
        self.conf_dict = dict()
        self.conf_dict["num_epoch"] = 50
        self.conf_dict["batch_size"] = 128
        self.conf_dict["train_data_dir_path"] = [""]
        self.conf_dict["test_data_dir_path"] = [""]

        self.conf_dict["lr"] = 0.0001
        self.conf_dict["adam_beta_1"] = 0.9
        self.conf_dict["adam_beta_2"] = 0.999
        self.conf_dict["epsilon"] = 1e-8
        self.conf_dict["weight_decay"] = 1e-2

        self.conf_dict["summary_writer_path"] = "summary"



class DummyConfigurationTest(ABCConfiguration):
    def __init__(self):
        self.conf_dict = dict()
        self.conf_dict["num_epoch"] = 50
        self.conf_dict["batch_size"] = 128
        self.conf_dict["train_data_dir_path"] = [""]
        self.conf_dict["test_data_dir_path"] = [""]

        self.conf_dict["lr"] = 0.01
        self.conf_dict["adam_beta_1"] = 0.9
        self.conf_dict["adam_beta_2"] = 0.999
        self.conf_dict["epsilon"] = 1e-8
        self.conf_dict["weight_decay"] = 1e-2

        self.conf_dict["summary_writer_path"] = "summary"