import logging
import sys

APP_LOGGER_NAME = 'DnlPosTagger'
def setup_applevel_logger(logger_name = APP_LOGGER_NAME, file_name=None):
    """
    Set up a logger for the application
    :param logger_name: name of the logger
    :param file_name: file to log to
    :return: logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)
    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def get_logger(module_name):
    """
    get function for the logger
    :param module_name: module name
    :return: logger for the module
    """
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)