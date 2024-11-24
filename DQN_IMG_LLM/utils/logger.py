# utils/logger.py

import logging
import os
from config import SAVE_DIR

# Ensure the log directory exists
log_dir = os.path.join(SAVE_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configure the logger
logging.basicConfig(
    filename=os.path.join(log_dir, 'training.log'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Create a logger instance
logger = logging.getLogger(__name__)
