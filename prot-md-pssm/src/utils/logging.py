import logging
import time
from multiprocessing import Process, Lock, Queue, log_to_stderr, get_logger

def setup_logging(file_path: str = "./logs/", name: str = "log", console: bool = False):
    handlers = []
    
    if console:
        handlers.append(logging.StreamHandler())
    handlers.append(logging.FileHandler(f"{file_path}/{name}_{time.strftime('%Y%m%d-%H%M%S')}.log"))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )