# docs: https://github.com/Delgan/loguru
import sys
from loguru import logger

config = {
        "handlers": [
            {"sink": sys.stdout, 'format': '{time} {level} {message}'},
        ]
    }

def logger_init():
    logger.configure(**config)
