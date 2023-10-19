import structlog
from src.core.config import config

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(),
        structlog.processors.KeyValueRenderer(),
    ]
)
logger = structlog.get_logger()

logger.info(
    "starting application",
    version=config.VERSION,
    env=config.ENV,
    host=config.HOST_IP,
    port=config.PORT,
)
