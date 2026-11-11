import sys
from loguru import logger
from src.config import LOG_DIR


#ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

#remove default Loguru handler
logger.remove()

#console logging
logger.add(
    sys.stderr,
    level="INFO",
    format=(  "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "{name}:{function}:{line} - {message}")
)

#file logging
logger.add(
    LOG_DIR / "rag-pipeline.log",
    level="DEBUG",
    rotation="10 MB",  # Rotate after 10 MB
    retention="7 days",  # Keep logs for 7 days
    compression="zip",  # Compress rotated logs
    encoding="utf-8",
    enqueue=True,  # Use a separate thread for logging
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} - {message}"
    )
)


#export configured logger
__all__=["logger"]
