import dramatiq
from dramatiq.brokers.stub import StubBroker
from loguru import logger

from src.core.config import get_settings
from src.core.constants import ENV_PRODUCTION

# Global broker instance
broker = None


def setup_dramatiq():
    global broker
    
    if broker is not None:
        return broker
    
    settings = get_settings()
    
    try:
        # Try Redis first if available
        try:
            from dramatiq.brokers.redis import RedisBroker
            broker = RedisBroker(url=settings.redis_url)
            logger.info(f"Dramatiq broker configured with Redis: {settings.redis_host}:{settings.redis_port}")
        except ImportError:
            # Redis is required in production environment
            if settings.environment == ENV_PRODUCTION:
                error_msg = (
                    "Redis is not available but is required in production environment. "
                    "Please install redis-py package and ensure Redis server is running."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Fall back to in-memory broker for development only
            logger.warning("Redis not available, using StubBroker for development")
            broker = StubBroker()
            broker.emit_after("process_boot")
            
        dramatiq.set_broker(broker)
        return broker
        
    except Exception as e:
        # In production, fail fast if Redis setup fails
        if settings.environment == ENV_PRODUCTION:
            error_msg = f"Failed to configure Redis broker in production: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Use stub broker as fallback in development only
        logger.error(f"Failed to configure Dramatiq broker: {e}")
        broker = StubBroker()
        broker.emit_after("process_boot")
        dramatiq.set_broker(broker)
        logger.warning("Using StubBroker as fallback for development")
        return broker


def get_broker():
    global broker
    
    if broker is None:
        broker = setup_dramatiq()
    
    return broker