import sys
import logging
def IntegerType(value):
    return sys.maxsize if value == 'inf' else int(value)

def LoggerLevelType(value):
    choices = {'debug': logging.DEBUG, 'info': logging.INFO, 'warn': logging.WARN, 'error': logging.ERROR}
    result = choices.get(value, logging.ERROR)
    return result