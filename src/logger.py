import logging

import os
from datetime import datetime

# Creating a Log File
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Setting Path for LOG File
log_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Even if dir exist, append and add files
os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

logging.basicConfig(
    filename= LOG_FILE_PATH,
    format = "[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO, 

)
 