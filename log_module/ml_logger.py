import logging
import os

class MLLogger:
    obj_logger = None

    @classmethod
    def get_logger(cls):
        if cls.obj_logger == None:
            cls.obj_logger = MLLogger()
        return cls.obj_logger

    def __init__(self, logger_level = logging.DEBUG):
        self.log_formatter = logging.Formatter("[%(asctime)s][%(levelname)s]: %(message)s")
        self.logger = logging.getLogger("ML_LOGGER")
        self.logger.setLevel(logger_level)

        # stdout handler
        stdout_handler = logging.StreamHandler()  # For stdout
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(self.log_formatter)
        self.logger.addHandler(stdout_handler)

        # file handler
        dir_path = os.path.dirname(os.path.realpath(__file__))
        parent_dir_path = os.path.dirname(dir_path)
        file_path = os.path.join(parent_dir_path, "ml_log.log")
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.log_formatter)
        self.logger.addHandler(file_handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)

logger = MLLogger.get_logger()
logger.info("this is the message for logging")