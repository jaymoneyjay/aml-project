# docs: https://github.com/Delgan/loguru
import sys
from loguru import logger

def logger_init(level='DEBUG'):
    if level not in ['TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL']:
        logger.warning('Loguru: Log level "{}" is invalid. The level has been set to "DEBUG".'.format(level))
        level = 'DEBUG'

    config = {
        "handlers": [
            {"sink": sys.stdout, 'format': '{time} {level} {message}', 'level': level},
            {"sink": sys.stderr, 'format': '{time} {level} {message}', 'level': 'ERROR'},
        ],
    }
    logger.configure(**config)
