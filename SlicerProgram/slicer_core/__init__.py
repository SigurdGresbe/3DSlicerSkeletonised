import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

logging.getLogger('slicer_core').setLevel(logging.DEBUG)
