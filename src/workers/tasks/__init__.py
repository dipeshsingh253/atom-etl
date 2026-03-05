# Import all tasks to ensure they are registered with Dramatiq
# This file serves as the entry point for the worker process

# Document ingestion tasks
from .ingestion_tasks import *  # noqa: F403