"""67-Finder: Real-time gesture and number detection."""

__version__ = "0.1.0"
__author__ = "Bogdan11212"

from .inference import GestureNumberDetector
from .utils import load_model, save_results

__all__ = [
    "GestureNumberDetector",
    "load_model",
    "save_results",
]
