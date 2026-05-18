import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_logging(log_file: Path):
    
    pkg_logger = logging.getLogger('slicer_core')
    pkg_logger.setLevel(logging.DEBUG)

    for handler in pkg_logger.handlers[:]:
        if isinstance(handler, (logging.FileHandler, logging.StreamHandler)):
            logger.debug(f"Removing old log handler: {handler.name}")
            pkg_logger.removeHandler(handler)
            
    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)-8s - %(name)-25s - %(message)s'
    )

    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        
        pkg_logger.addHandler(file_handler)
        logger.info(f"File logging configured. Log file: {log_file}")
    except Exception as e:
        logger.error(f"Failed to set up file logger: {e}")

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    
    pkg_logger.addHandler(console_handler)

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
