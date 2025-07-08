"""
AgentUp Image Processing Plugin

A comprehensive image processing plugin for the AgentUp framework that provides
image analysis, transformation, and manipulation capabilities.
"""

from .plugin import ImageProcessingPlugin
from .processor import ImageProcessor

__version__ = "1.0.0"
__all__ = ["ImageProcessingPlugin", "ImageProcessor"]
