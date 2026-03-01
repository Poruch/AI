# src/config.py
import os

# Пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Параметры моделей
DEFAULT_BATCH_SIZE = 1
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 512

# Настройки UDP
UDP_IP = "127.0.0.1"
UDP_PORT = 12345