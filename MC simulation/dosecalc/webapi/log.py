import sys
import logging
import warnings

modlogger_initialized = False
def get_module_logger(name, level=None):
    global modlogger_initialized

    if not modlogger_initialized:
        rootlogger = logging.getLogger('dosesim')
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(fmt="%(asctime)s %(threadName)s (%(levelname)s) | %(module)s:%(lineno)d: %(message)s"))
        rootlogger.addHandler(sh)
        modlogger_initialized = True

    # get named logger
    logger = logging.getLogger('dosesim.'+name)

    # reset module logging level
    if level is not None:
        if isinstance(level, str):
            level = logging._nameToLevel[level]
        logging.getLogger('dosesim').setLevel(level)

    return logger

def add_argument_loglevel(parser):
    parser.add_argument('--loglevel', '-L', type=str, default='INFO', choices=logging._nameToLevel.keys(), help='set the logging level')

def logtask(callable, message, taskdata):
    s = 'subbeam_id'
    if isinstance(taskdata, dict):
        id = taskdata.get(s, 'unknown id')
        addr = taskdata.get('host', 'unknown address')
    else:
        id =   getattr(taskdata, s, 'unknown id')
        addr = getattr(taskdata, 'host', 'unknown address')
    callable('{} for task {} from "{}"'.format(
        message, id, addr))

def depwarning(prefer_function):
    warnings.warn('This function will be deprecated in the future. Please use "{}" instead.'.format(prefer_function))
