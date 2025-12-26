"""
DeQoG Logger Module

Provides logging functionality for the DeQoG framework.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    from rich.logging import RichHandler
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class DeQoGLogger:
    """
    Custom logger for DeQoG framework with rich formatting support.
    """
    
    _loggers: dict = {}
    _initialized: bool = False
    _config: Optional[dict] = None
    
    @classmethod
    def setup(cls, config: Optional[dict] = None):
        """
        Setup the logging system.
        
        Args:
            config: Logging configuration dictionary
        """
        if cls._initialized and cls._config == config:
            return
        
        cls._config = config or {}
        
        # Get configuration values
        log_level = cls._config.get("level", "INFO")
        log_format = cls._config.get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        log_file = cls._config.get("file", "logs/deqog.log")
        console_output = cls._config.get("console", True)
        use_rich = cls._config.get("rich_traceback", True) and RICH_AVAILABLE
        
        # Create log directory if needed
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger("deqog")
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers = []
        
        # Add console handler
        if console_output:
            if use_rich:
                console_handler = RichHandler(
                    console=Console(stderr=True),
                    show_time=True,
                    show_path=True,
                    markup=True,
                    rich_tracebacks=True,
                )
                console_handler.setFormatter(logging.Formatter("%(message)s"))
            else:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(logging.Formatter(log_format))
            
            root_logger.addHandler(console_handler)
        
        # Add file handler
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(log_format))
            root_logger.addHandler(file_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str = "deqog") -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        if not cls._initialized:
            cls.setup()
        
        if name not in cls._loggers:
            logger = logging.getLogger(f"deqog.{name}" if name != "deqog" else name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]


def setup_logger(config: Optional[dict] = None):
    """
    Setup the DeQoG logging system.
    
    Args:
        config: Logging configuration dictionary
    """
    DeQoGLogger.setup(config)


def get_logger(name: str = "deqog") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return DeQoGLogger.get_logger(name)


# Create a default logger
logger = get_logger()

