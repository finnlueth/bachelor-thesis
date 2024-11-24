import logging
import time


def setup_logging(file_path: str = "./", name: str = "log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(f"{file_path}/{name}_{time.strftime('%Y%m%d-%H%M%S')}.log"), logging.StreamHandler()],
    )
