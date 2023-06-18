from configuration.config_key_enum import ConfigKey as CK

class ConfigurationExample:
    def __init__(self):
        self.dict = dict()
        self.dict[CK.LEARNING_ITERATION] = 0
        self.dict[CK.LEARNING_OPTIMIZER] = None
        self.dict[CK.LEARNING_NUM_EPOCHS] = 0
        self.dict[CK.LEARNING_SCHEDULER] = None
        self.dict[CK.MODEL_NAME] = ""
        self.dict[CK.MODEL_LAYER] = 1
        self.dict[CK.MODEL_SETTING] = None
        self.dict[CK.DATA_VOCAB_SIZE] = 0
        self.dict[CK.DATA_SIZE] = 0
        self.dict[CK.DATA_TRAIN_DIR_PATHS] = [""]
        self.dict[CK.DATA_TEST_DIR_PATHS] = [""]

    def get_value(self, key):
        return self.dict[key] if key in self.dict else None

    def set_value(self, key, value):
        self.dict[key] = value
