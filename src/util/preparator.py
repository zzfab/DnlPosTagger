import os
import sys

wdir = os.getcwd()
sys.path.append(wdir)

from src.util import logger

logger = logger.setup_applevel_logger(file_name = os.path.join(wdir,'logging/app_debug.log'))
logger.info('Run Preparator for POS Tagging')

