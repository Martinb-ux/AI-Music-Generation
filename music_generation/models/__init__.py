"""Models module for LSTM music generation."""

from .melody_lstm import MelodyLSTM, create_model
from .generator import MusicGenerator, create_generator

__all__ = ['MelodyLSTM', 'create_model', 'MusicGenerator', 'create_generator']
