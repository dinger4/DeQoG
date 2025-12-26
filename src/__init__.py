"""
DeQoG - Diversity-Driven Quality-Assured Code Generation

A framework for generating fault-tolerant N-version code using LLM-based
diversity enhancement and quality assurance mechanisms.
"""

__version__ = "1.0.0"
__author__ = "DeQoG Team"

from .core.pipeline import DeQoGPipeline
from .utils.config import Config

__all__ = [
    "DeQoGPipeline",
    "Config",
    "__version__",
]

