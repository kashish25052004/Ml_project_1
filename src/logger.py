import logging
import os
from datetime import datetime

Log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_folder = os.path.join(os.getcwd(),"logs")

os.makedirs(log_folder,exist_ok = True)

log_files_path = os.path.join(log_folder,Log_file)


logging.basicConfig(
    filename = log_files_path,
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
)


if __name__ == "__main__":
    logging.info("Logging has started")