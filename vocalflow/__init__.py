"""
VocalFlow - Universal Voice Dictation System
Cross-platform voice-to-text with adaptive learning
"""

__version__ = "1.0.0"
__author__ = "VocalFlow Team"
__license__ = "MIT"

from .main import (
    VocalFlowSystem,
    VoiceProfile,
    ProfileManager,
    AudioConfig,
    TranscriptionConfig,
    OutputConfig
)

__all__ = [
    "VocalFlowSystem",
    "VoiceProfile", 
    "ProfileManager",
    "AudioConfig",
    "TranscriptionConfig",
    "OutputConfig"
]
