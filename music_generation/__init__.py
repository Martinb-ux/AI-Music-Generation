"""
Music Generation with LSTM

A deep learning project for generating original piano melodies using LSTM neural networks.
"""

__version__ = "1.0.0"
__author__ = "Rix"
__project__ = "CST435 Final Project"

# Core modules
from . import preprocessing
from . import models
from . import training
from . import evaluation

__all__ = ['preprocessing', 'models', 'training', 'evaluation']
