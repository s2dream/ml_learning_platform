from enum import Enum, auto

class ConfigKey(Enum):
    LEARNING_NUM_EPOCHS = auto()
    LEARNING_ITERATION = auto()
    LEARNING_OPTIMIZER = auto()
    LEARNING_SCHEDULER = auto()
    LEARNING_MINI_BATCH_SIZE = auto()
    LEARNING_TOTAL_BATcH_SIZE = auto()
    LEARNING_LEARNING_RATE = auto()

    MODEL_NAME = auto()
    MODEL_SETTING = auto()
    MODEL_LAYER = auto()

    DATA_VOCAB_SIZE = auto()
    DATA_SIZE = auto()
    DATA_TRAIN_FILE_PATHS = auto()
    DATA_TRAIN_DIR_PATHS = auto()
    DATA_TEST_FILE_PATHS = auto()
    DATA_TEST_DIR_PATHS = auto()
    DATA_ENTITY_SIZE = auto()
    DATA_RELATION_SIZE = auto()

    OPT_ADAMW_BETA1 = auto()
    OPT_ADAMW_BETA2 = auto()
    OPT_ADAMW_EPSILON = auto()
    OPT_ADAMW_WEIGHT_DECAY = auto()

    LOG_SUMMARY_WRITER_PATH = auto()