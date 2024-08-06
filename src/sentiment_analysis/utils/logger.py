import logging
import os
import datetime
# from config import config

LOGGER_NAME = "Sentiment-Analysis"
LEVEL = 'INFO'
FORMAT = "%(asctime)s: %(name)s: %(levelname)s: %(filename)s: %(funcName)s:- %(message)s"
LOG_FILE = "../logs/log_data/api/"
class SingletonType(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(object, metaclass=SingletonType):
    _logger = None

    def __init__(self):
        self._logger = logging.getLogger(LOGGER_NAME)
        self._logger.setLevel(LEVEL)
        formatter = logging.Formatter(FORMAT)
        now = datetime.datetime.now()
        dirname = os.getcwd()
        print("dir name{}".format(dirname))
        dirname = dirname.split('src')
        log_file_path = dirname[0] + LOG_FILE
        print(log_file_path)
        if not os.path.isdir(log_file_path):
            os.makedirs(log_file_path)
        fileHandler = logging.FileHandler(log_file_path + "/service" + now.strftime("%Y-%m-%d")+".log")

        streamHandler = logging.StreamHandler()

        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        self._logger.addHandler(fileHandler)
        self._logger.addHandler(streamHandler)

        print("Generate new instance")

    def get_logger(self):
        return self._logger


# if __name__=='__main__':
#     log = Logger.__call__().get_logger()
#     log.info("this logging")