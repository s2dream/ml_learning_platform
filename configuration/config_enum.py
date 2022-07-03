from enum import Enum, auto

class ConfigKey(Enum):
    LEARNING_TOTAL_EPOCHS = auto()
    LEARNING_ITERATION = auto()
    LEARNING_OPTIMIZER = auto()
    LEARNING_SCHEDULER = auto()
    MODEL_NAME = auto()
    MODEL_SETTING = auto()
    MODEL_LAYER = auto()
    DATA_VOCAB_SIZE = auto()
    DATA_SIZE = auto()
    DATA_TRAIN_PATH = auto()
    DATA_TEST_PATH = auto()
    DATA_ENTITY_SIZE = auto()
    DATA_RELATION_SIZE = auto()
