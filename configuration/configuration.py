from configuration.config_enum import ConfigKey as CK

class Configuration:


    def __init__(self):
        self.dict = dict()
        self.dict[CK.LEARNING_ITERATION] = 0
        self.dict[CK.LEARNING_OPTIMIZER] = None
        self.dict[CK.LEARNING_TOTAL_EPOCHS] = 0
        self.dict[CK.LEARNING_SCHEDULER] = None
        self.dict[CK.MODEL_NAME] = ""
        self.dict[CK.MODEL_LAYER] = 1
        self.dict[CK.MODEL_SETTING] = None
        self.dict[CK.DATA_VOCAB_SIZE] = 0
        self.dict[CK.DATA_SIZE] = 0
        self.dict[CK.DATA_TRAIN_PATH] = ""
        self.dict[CK.DATA_TEST_PATH] = ""
        self.dict[CK.DATA_ENTITY_SIZE] = ""
        self.dict[CK.DATA_RELATION_SIZE] = ""

    def get_value(self, key):
        return self.dict[key] if key in self.dict else None

    def set_value(self, key, value):
        self.dict[key] = value