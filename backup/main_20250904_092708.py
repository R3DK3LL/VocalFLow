#!/usr/bin/env python3
"""
VocalFlow - Universal Voice Dictation System with Enhanced Command Processing
Cross-platform voice-to-text with adaptive learning and linguistic command integration

Author: VocalFlow Team
License: MIT
Version: 1.1.0 (Enhanced Integration)
"""

import numpy as np
import sounddevice as sd
import subprocess
import platform
import sys
import os
import json
import time
import logging
import argparse
import signal
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
import re
import queue
import threading
from collections import defaultdict, deque

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper not installed. Run: pip install faster-whisper")
    sys.exit(1)

try:
    import pyperclip
except ImportError:
    pyperclip = None

# Enhanced Command Integration - Import linguistic components
try:
    from linguistic_bridge import LinguisticBridge
    from enhanced_agents import EnhancedVocalFlowAgent
    ENHANCED_COMMANDS_AVAILABLE = True
    print("Enhanced command processing available")
except ImportError as e:
    print(f"Enhanced commands not available: {e}")
    ENHANCED_COMMANDS_AVAILABLE = False

# Platform detection
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MAC = platform.system() == "Darwin"

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging with file and console handlers"""
    log_dir = Path.home() / ".vocalflow" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Setup logger
    logger = logging.getLogger('vocalflow')
    logger.setLevel(level)
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_dir / "vocalflow.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

@dataclass
class VoiceProfile:
    """Adaptive voice profile that learns from each user"""
    name: str = "default"
    background_noise: float = 0.005
    speech_threshold: float = 0.003
    avg_pause_length: float = 0.8
    avg_speech_volume: float = 0.1
    total_words: int = 0
    total_sessions: int = 0
    common_words: Dict[str, int] = None
    corrections: Dict[str, str] = None
    created: str = ""
    last_used: str = ""
    
    def __post_init__(self):
        if self.common_words is None:
            self.common_words = {}
        if self.corrections is None:
            self.corrections = {}
        if not self.created:
            self.created = time.strftime("%Y-%m-%d %H:%M:%S")

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    samplerate: int = 16000
    channels: int = 1
    dtype: str = 'float32'
    blocksize: int = 1024
    device_id: Optional[int] = None
    
@dataclass
class TranscriptionConfig:
    """Whisper transcription configuration"""
    model_size: str = "base"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 3
    vad_filter: bool = True
    temperature: float = 0.1
    
@dataclass
class OutputConfig:
    """Text output configuration"""
    method: str = "auto"  # auto, clipboard, xdotool, wintype
    typing_delay: int = 5
    auto_capitalize: bool = True
    smart_punctuation: bool = True

class ProfileManager:
    """Manages voice profiles with learning capabilities"""
    
    def __init__(self):
        self.profile_dir = Path.home() / ".vocalflow" / "profiles"
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('vocalflow.profiles')
        
    def create_profile(self, name: str) -> VoiceProfile:
        """Create a new voice profile"""
        profile = VoiceProfile(name=name)
        self.save_profile(profile)
        self.logger.info(f"Created new voice profile: {name}")
        return profile
        
    def load_profile(self, name: str) -> Optional[VoiceProfile]:
        """Load existing voice profile"""
        profile_file = self.profile_dir / f"{name}.json"
        if profile_file.exists():
            try:
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                    profile = VoiceProfile(**data)
                    self.logger.info(f"Loaded voice profile: {name}")
                    return profile
            except Exception as e:
                self.logger.error(f"Failed to load profile {name}: {e}")
        return None
        
    def save_profile(self, profile: VoiceProfile):
        """Save voice profile to disk"""
        profile.last_used = time.strftime("%Y-%m-%d %H:%M:%S")
        profile_file = self.profile_dir / f"{profile.name}.json"
        
        try:
            with open(profile_file, 'w') as f:
                json.dump(asdict(profile), f, indent=2)
            self.logger.debug(f"Saved profile: {profile.name}")
        except Exception as e:
            self.logger.error(f"Failed to save profile {profile.name}: {e}")
            
    def list_profiles(self) -> List[str]:
        """List all available profiles"""
        profiles = []
        for file in self.profile_dir.glob("*.json"):
            profiles.append(file.stem)
        return sorted(profiles)
        
    def get_or_create_profile(self, name: str) -> VoiceProfile:
        """Get existing profile or create new one"""
        profile = self.load_profile(name)
        if profile is None:
            profile = self.create_profile(name)
        return profile

class CrossPlatformOutput:
    """Universal text output for Windows, Linux, and Mac"""
    
    def __init__(self, config: OutputConfig):
        self.config = config
        self.logger = logging.getLogger('vocalflow.output')
        self.method = self._detect_best_method() if config.method == "auto" else config.method
        self.logger.info(f"Using output method: {self.method}")
        
    def _detect_best_method(self) -> str:
        """Auto-detect best output method for current platform"""
        if IS_WINDOWS:
            return "clipboard"
        elif IS_LINUX:
            # Try xdotool first
            try:
                result = subprocess.run(["which", "xdotool"], 
                                      check=True, capture_output=True, text=True)
                return "xdotool"
            except (subprocess.CalledProcessError, FileNotFoundError):
                return "clipboard"
        elif IS_MAC:
            return "clipboard"
        else:
            return "clipboard"
    
    def type_text(self, text: str):
        """Output text using best available method"""
        if not text.strip():
            return
            
        try:
            if self.method == "xdotool":
                self._xdotool_type(text)
            elif self.method == "clipboard":
                self._clipboard_type(text)
            elif self.method == "wintype":
                self._windows_type(text)
            else:
                self.logger.error(f"Unknown output method: {self.method}")
        except Exception as e:
            self.logger.error(f"Failed to output text '{text[:50]}...': {e}")
            
    def _xdotool_type(self, text: str):
        """Linux xdotool typing"""
        cmd = ["xdotool", "type", "--delay", str(self.config.typing_delay), text + " "]
        subprocess.run(cmd, check=True, capture_output=True)
        
    def _clipboard_type(self, text: str):
        """Cross-platform clipboard method"""
        try:
            if pyperclip:
                pyperclip.copy(text)
                print(f"📋 Copied: {text}")
            else:
                # System clipboard fallback
                if IS_WINDOWS:
                    subprocess.run(["clip"], input=text.encode('utf-8'), check=True)
                elif IS_MAC:
                    subprocess.run(["pbcopy"], input=text.encode('utf-8'), check=True)
                elif IS_LINUX:
                    try:
                        subprocess.run(["xclip", "-selection", "clipboard"], 
                                     input=text.encode('utf-8'), check=True)
                    except FileNotFoundError:
                        # Try xsel as fallback
                        subprocess.run(["xsel", "--clipboard", "--input"], 
                                     input=text.encode('utf-8'), check=True)
                print(f"📋 Copied: {text}")
        except Exception as e:
            self.logger.error(f"Clipboard operation failed: {e}")
            
    def _windows_type(self, text: str):
        """Windows direct typing (requires pyautogui)"""
        try:
            import pyautogui
            pyautogui.write(text + " ", interval=self.config.typing_delay/1000)
        except ImportError:
            self.logger.error("pyautogui not available for Windows typing")
            self._clipboard_type(text)  # Fallback

class AdaptiveLearner:
    """Learns and adapts to user's voice patterns"""
    
    def __init__(self, profile: VoiceProfile):
        self.profile = profile
        self.logger = logging.getLogger('vocalflow.learning')
        self.session_data = {
            'volumes': deque(maxlen=1000),
            'pause_lengths': deque(maxlen=100),
            'words': defaultdict(int),
            'corrections': [],
            'session_start': time.time()
        }
        
    def update_audio_stats(self, volume: float, pause_length: Optional[float] = None):
        """Update audio statistics for adaptive learning"""
        self.session_data['volumes'].append(volume)
        if pause_length is not None:
            self.session_data['pause_lengths'].append(pause_length)
            
    def learn_word(self, word: str):
        """Learn frequently used words"""
        word_clean = re.sub(r'[^\w]', '', word.lower())
        if len(word_clean) >= 2:
            self.session_data['words'][word_clean] += 1
            
    def learn_correction(self, original: str, corrected: str):
        """Learn user corrections for future application"""
        self.session_data['corrections'].append((original, corrected))
        self.logger.debug(f"Learned correction: '{original}' -> '{corrected}'")
        
    def update_profile(self):
        """Update profile with learned session data"""
        try:
            # Update audio characteristics
            if self.session_data['volumes']:
                volumes = list(self.session_data['volumes'])
                self.profile.avg_speech_volume = float(np.mean(volumes))
                self.profile.background_noise = float(np.percentile(volumes, 20))
                self.profile.speech_threshold = max(
                    self.profile.background_noise * 2.5, 0.003
                )
                
            # Update pause timing
            if self.session_data['pause_lengths']:
                pauses = list(self.session_data['pause_lengths'])
                self.profile.avg_pause_length = float(np.mean(pauses))
                
            # Update vocabulary
            for word, count in self.session_data['words'].items():
                self.profile.common_words[word] = (
                    self.profile.common_words.get(word, 0) + count
                )
                self.profile.total_words += count
                
            # Update corrections
            for original, corrected in self.session_data['corrections']:
                self.profile.corrections[original.lower()] = corrected
                
            self.profile.total_sessions += 1
            
            # Log session summary
            session_duration = time.time() - self.session_data['session_start']
            words_this_session = sum(self.session_data['words'].values())
            
            self.logger.info(
                f"Session complete: {session_duration:.1f}s, "
                f"{words_this_session} words, "
                f"{len(self.session_data['corrections'])} corrections learned"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update profile: {e}")

class SmartFilter:
    """Enhanced filtering with learning capabilities"""
    
    def __init__(self, learner: AdaptiveLearner):
        self.learner = learner
        self.logger = logging.getLogger('vocalflow.filter')
        
        # Common filler words to filter out
        self.FILLER_WORDS = {
            "uh", "um", "ah", "er", "hmm", "mm", "mhm",
            "you know", "like", "so", "well", "actually",
            "basically", "literally", "obviously", "definitely",
            "mm-hmm", "uh-huh", "yeah", "yep", "ok", "okay"
        }
        
    def is_good_text(self, text: str) -> bool:
        """Determine if text should be output"""
        if not text or len(text.strip()) < 2:
            return False
            
        text_clean = text.strip().lower()
        
        # Learn words for vocabulary building
        for word in text.split():
            self.learner.learn_word(word)
            
        # Filter filler words
        if text_clean in self.FILLER_WORDS:
            self.logger.debug(f"Filtered filler word: '{text_clean}'")
            return False
            
        # Filter if mostly non-alphabetic
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.6:
            self.logger.debug(f"Filtered low alpha ratio: '{text}' ({alpha_ratio:.2f})")
            return False
            
        # Filter repetitive text
        words = text_clean.split()
        if len(words) > 1 and len(set(words)) == 1:
            self.logger.debug(f"Filtered repetitive: '{text}'")
            return False
            
        return True
        
    def apply_corrections(self, text: str) -> str:
        """Apply learned corrections to text"""
        corrected_text = text
        
        for original, corrected in self.learner.profile.corrections.items():
            pattern = re.escape(original)
            corrected_text = re.sub(pattern, corrected, corrected_text, flags=re.IGNORECASE)
            
        if corrected_text != text:
            self.logger.debug(f"Applied correction: '{text}' -> '{corrected_text}'")
            
        return corrected_text

class AudioProcessor:
    """Handles audio capture and voice activity detection"""
    
    def __init__(self, config: AudioConfig, profile: VoiceProfile):
        self.config = config
        self.profile = profile
        self.logger = logging.getLogger('vocalflow.audio')
        
        self.audio_queue = queue.Queue(maxsize=100)
        self.is_running = False
        self.stream: Optional[sd.InputStream] = None
        
    def audio_callback(self, indata, frames, time, status):
        """Audio input callback"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
            
        try:
            if not self.audio_queue.full():
                self.audio_queue.put(indata.copy(), block=False)
            else:
                # Drop oldest frame if queue is full
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put(indata.copy(), block=False)
                except queue.Empty:
                    pass
        except Exception as e:
            self.logger.error(f"Audio callback error: {e}")
    
    def start_stream(self):
        """Start audio input stream"""
        try:
            # List available devices for debugging
            devices = sd.query_devices()
            self.logger.debug(f"Available audio devices: {devices}")
            
            self.stream = sd.InputStream(
                device=self.config.device_id,
                samplerate=self.config.samplerate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                callback=self.audio_callback,
                blocksize=self.config.blocksize,
                latency='low'
            )
            self.stream.start()
            self.is_running = True
            
            # Fixed device info logging
            if self.config.device_id is not None:
                device_info = sd.query_devices(self.config.device_id)
                if isinstance(device_info, dict) and 'name' in device_info:
                    device_name = device_info['name']
                else:
                    device_name = f"device {self.config.device_id}"
            else:
                device_name = "default device"
            
            self.logger.info(f"Audio stream started: {device_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to start audio stream: {e}")
            raise
    
    def stop_stream(self):
        """Stop audio input stream"""
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                self.logger.info("Audio stream stopped")
            except Exception as e:
                self.logger.error(f"Error stopping audio stream: {e}")
    
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get audio chunk from queue"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
    
    def calculate_rms(self, audio_data: np.ndarray) -> float:
        """Calculate RMS energy of audio data"""
        if len(audio_data) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_data.astype(np.float64) ** 2)))

class TranscriptionEngine:
    """Handles speech-to-text transcription with Whisper"""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = logging.getLogger('vocalflow.transcription')
        self.model: Optional[WhisperModel] = None
        
    def initialize_model(self):
        """Initialize Whisper model"""
        try:
            self.logger.info(f"Loading Whisper model: {self.config.model_size}")
            
            self.model = WhisperModel(
                self.config.model_size,
                compute_type=self.config.compute_type,
                device="auto"
            )
            
            # Warm up model with dummy audio
            dummy_audio = np.zeros(16000, dtype=np.float32)
            list(self.model.transcribe(dummy_audio, beam_size=1))
            
            self.logger.info("Whisper model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_data: np.ndarray) -> List[str]:
        """Transcribe audio data to text segments"""
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        try:
            start_time = time.time()
            
            segments, info = self.model.transcribe(
                audio_data.flatten(),
                language=self.config.language,
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
                temperature=self.config.temperature,
                condition_on_previous_text=False,
                no_speech_threshold=0.4
            )
            
            results = []
            for segment in segments:
                text = segment.text.strip()
                if text and len(text) >= 2:
                    results.append(text)
                    
            duration = time.time() - start_time
            self.logger.debug(
                f"Transcribed {len(audio_data)/16000:.1f}s audio in {duration:.2f}s, "
                f"got {len(results)} segments"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return []

class VocalFlowSystem:
    """Main VocalFlow dictation system with enhanced command processing"""
    
    def __init__(self, 
                 profile_name: str = "default",
                 audio_config: Optional[AudioConfig] = None,
                 transcription_config: Optional[TranscriptionConfig] = None,
                 output_config: Optional[OutputConfig] = None,
                 enable_commands: bool = True):
        
        # Initialize logging first
        self.logger = logging.getLogger('vocalflow.system')
        
        # Initialize profile management
        self.profile_manager = ProfileManager()
        self.profile = self.profile_manager.get_or_create_profile(profile_name)
        self.learner = AdaptiveLearner(self.profile)
        
        # Initialize configurations
        self.audio_config = audio_config or AudioConfig()
        self.transcription_config = transcription_config or TranscriptionConfig()
        self.output_config = output_config or OutputConfig()
        
        # Initialize components
        self.audio_processor = AudioProcessor(self.audio_config, self.profile)
        self.transcription_engine = TranscriptionEngine(self.transcription_config)
        self.output_handler = CrossPlatformOutput(self.output_config)
        self.filter = SmartFilter(self.learner)
        
        # ENHANCED: Initialize command processing components
        self.enable_commands = enable_commands and ENHANCED_COMMANDS_AVAILABLE
        self.linguistic_bridge = None
        self.command_agent = None
        
        if self.enable_commands:
            try:
                self.linguistic_bridge = LinguisticBridge()
                self.command_agent = EnhancedVocalFlowAgent()
                self.logger.info("Enhanced command processing enabled")
                print("🎯 Enhanced command processing: ENABLED")
                if hasattr(self.linguistic_bridge, 'get_status'):
                    status = self.linguistic_bridge.get_status()
                    self.logger.info(f"Linguistic status: {status}")
            except Exception as e:
                self.logger.warning(f"Could not enable enhanced commands: {e}")
                self.enable_commands = False
        else:
            self.logger.info("Enhanced command processing disabled")
        
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Shutdown signal received")
        self.stop()
        
    def initialize(self):
        """Initialize all system components"""
        self.logger.info("Initializing VocalFlow system...")
        
        try:
            # Initialize transcription engine
            self.transcription_engine.initialize_model()
            
            self.logger.info("VocalFlow system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise
        
    def start(self):
        """Start the dictation system"""
        try:
            self.initialize()
            
            # Start audio processing
            self.audio_processor.start_stream()
            
            # Start processing thread
            self.is_running = True
            self.processing_thread = threading.Thread(
                target=self._process_audio_stream, 
                daemon=True,
                name="AudioProcessor"
            )
            self.processing_thread.start()
            
            self.logger.info(f"VocalFlow started (Profile: {self.profile.name})")
            
        except Exception as e:
            self.logger.error(f"Failed to start VocalFlow: {e}")
            self.stop()
            raise
        
    def stop(self):
        """Stop dictation system and save learning data"""
        if not self.is_running:
            return
            
        self.logger.info("Stopping VocalFlow system...")
        
        self.is_running = False
        
        # Stop audio processing
        self.audio_processor.stop_stream()
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            
        # Save learned data
        try:
            self.learner.update_profile()
            self.profile_manager.save_profile(self.profile)
            self.logger.info("Learning data saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save learning data: {e}")
            
        self.logger.info("VocalFlow stopped")
        
    def _process_audio_stream(self):
        """Main audio processing loop with voice activity detection"""
        buffer = np.array([], dtype=np.float32).reshape(0, 1)
        has_speech = False
        silence_start = None
        last_process_time = time.time()
        
        self.logger.debug("Audio processing thread started")
        
        while self.is_running:
            try:
                # Get audio chunk
                chunk = self.audio_processor.get_audio_chunk()
                if chunk is None:
                    time.sleep(0.01)
                    continue
                
                # Add to buffer
                buffer = np.vstack([buffer, chunk])
                
                # Calculate audio characteristics
                rms = self.audio_processor.calculate_rms(chunk)
                current_time = time.time()
                
                # Update learning with audio stats
                self.learner.update_audio_stats(rms)
                
                # Voice activity detection using adaptive thresholds
                speech_threshold = max(
                    self.profile.speech_threshold,
                    self.profile.background_noise * 2.5
                )
                
                if rms > speech_threshold:
                    if not has_speech:
                        has_speech = True
                        self.logger.debug("Speech activity detected")
                    silence_start = None
                else:
                    # Track silence periods
                    if has_speech and silence_start is None:
                        silence_start = current_time
                        
                    # Check for end of speech
                    if (has_speech and silence_start and 
                        (current_time - silence_start) > self.profile.avg_pause_length):
                        
                        # Process accumulated buffer
                        if len(buffer) >= (self.audio_config.samplerate * 0.3):
                            pause_length = current_time - silence_start
                            self.learner.update_audio_stats(rms, pause_length)
                            self._process_audio_buffer(buffer)
                            
                        # Reset for next speech segment
                        buffer = np.array([], dtype=np.float32).reshape(0, 1)
                        has_speech = False
                        silence_start = None
                        last_process_time = current_time
                
                # Prevent buffer overflow
                max_buffer_length = self.audio_config.samplerate * 10  # 10 seconds max
                if len(buffer) >= max_buffer_length:
                    if has_speech:
                        self._process_audio_buffer(buffer)
                    buffer = np.array([], dtype=np.float32).reshape(0, 1)
                    has_speech = False
                    silence_start = None
                    
            except Exception as e:
                self.logger.error(f"Audio processing error: {e}")
                time.sleep(0.1)
                
        self.logger.debug("Audio processing thread stopped")
        
    def _process_audio_buffer(self, audio_buffer: np.ndarray):
        """Process accumulated audio buffer for transcription - WITH ENHANCED COMMAND INTEGRATION"""
        try:
            # Transcribe audio
            texts = self.transcription_engine.transcribe_audio(audio_buffer)
            
            # Process each transcribed segment
            for text in texts:
                # Apply smart filtering
                if self.filter.is_good_text(text):
                    # Apply learned corrections
                    corrected_text = self.filter.apply_corrections(text)
                    
                    # ENHANCED: Check for commands BEFORE formatting as dictation text
                    if self.enable_commands and self.linguistic_bridge and self.command_agent:
                        is_command, command_type, metadata = self.linguistic_bridge.process_command(corrected_text)
                        
                        if is_command:
                            # Process as command
                            self.logger.info(f"Command detected: '{command_type}' from text: '{corrected_text}'")
                            confidence = metadata.get('confidence', 0)
                            method = metadata.get('method', 'unknown')
                            print(f"🎯 Command: {command_type} (confidence: {confidence:.2f}, method: {method})")
                            
                            # Execute command using enhanced agent
                            try:
                                success = self.command_agent.process_text_enhanced(corrected_text)
                                if success:
                                    self.logger.info(f"Command '{command_type}' executed successfully")
                                else:
                                    self.logger.warning(f"Command '{command_type}' execution failed")
                            except Exception as cmd_error:
                                self.logger.error(f"Command execution error: {cmd_error}")
                            
                            # Don't process as dictation text if it was a command
                            continue
                    
                    # If not a command, process as regular dictation text
                    # Format text
                    formatted_text = self._format_text(corrected_text)
                    
                    # Output text
                    self.output_handler.type_text(formatted_text)
                    
        except Exception as e:
            self.logger.error(f"Buffer processing error: {e}")
            
    def _format_text(self, text: str) -> str:
        """Apply smart text formatting"""
        if not text:
            return text
            
        # Basic cleanup
        text = text.strip()
        
        # Fix common transcription issues
        text = re.sub(r'\bi\b', 'I', text)  # Fix lowercase "i"
        text = re.sub(r'\s+', ' ', text)    # Normalize whitespace
        
        # Smart capitalization
        if self.output_config.auto_capitalize and text:
            if text[0].islower():
                text = text[0].upper() + text[1:]
                
        # Smart punctuation
        if self.output_config.smart_punctuation:
            words = text.split()
            if len(words) >= 2 and not text.endswith(('.', '!', '?', ':', ';', ',')):
                text += '.'
                
        return text
    
    def get_command_status(self) -> Dict[str, Any]:
        """Get enhanced command processing status"""
        base_status = {
            'commands_enabled': self.enable_commands,
            'commands_available': ENHANCED_COMMANDS_AVAILABLE
        }
        
        if self.enable_commands and self.linguistic_bridge:
            if hasattr(self.linguistic_bridge, 'get_status'):
                base_status.update(self.linguistic_bridge.get_status())
            if self.command_agent and hasattr(self.command_agent, 'get_status'):
                base_status['agent_status'] = self.command_agent.get_status()
        
        return base_status

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="VocalFlow - Universal Voice Dictation System with Enhanced Commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m vocalflow.main --profile john
  python -m vocalflow.main --output clipboard --verbose
  python -m vocalflow.main --model small --language es
  python -m vocalflow.main --enable-commands  # Enable voice commands
  python -m vocalflow.main --disable-commands # Disable voice commands
        """
    )
    
    parser.add_argument(
        "--profile", "-p",
        default="default",
        help="Voice profile name (default: %(default)s)"
    )
    
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List all available voice profiles"
    )
    
    parser.add_argument(
        "--output", "-o",
        choices=["auto", "clipboard", "xdotool", "wintype"],
        default="auto",
        help="Text output method (default: %(default)s)"
    )
    
    parser.add_argument(
        "--model", "-m",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: %(default)s)"
    )
    
    parser.add_argument(
        "--language", "-l",
        default="en",
        help="Recognition language code (default: %(default)s)"
    )
    
    parser.add_argument(
        "--audio-device",
        type=int,
        help="Audio input device ID (use --list-devices to see options)"
    )
    
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices"
    )
    
    parser.add_argument(
        "--enable-commands",
        action="store_true",
        default=True,
        help="Enable enhanced voice command processing (default: enabled)"
    )
    
    parser.add_argument(
        "--disable-commands",
        action="store_true",
        help="Disable enhanced voice command processing"
    )
    
    parser.add_argument(
        "--command-status",
        action="store_true",
        help="Show command processing status and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser

def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Handle special commands
    if args.list_devices:
        print("Available audio input devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                status = " (default)" if i == sd.default.device[0] else ""
                print(f"  {i}: {device['name']}{status}")
        return 0
    
    if args.list_profiles:
        manager = ProfileManager()
        profiles = manager.list_profiles()
        if profiles:
            print("Available voice profiles:")
            for profile_name in profiles:
                profile = manager.load_profile(profile_name)
                if profile:
                    sessions = profile.total_sessions
                    words = profile.total_words
                    last_used = profile.last_used or "Never"
                    print(f"  {profile_name}: {sessions} sessions, {words} words, last used: {last_used}")
        else:
            print("No voice profiles found.")
        return 0
    
    # Handle command enable/disable flags
    enable_commands = args.enable_commands and not args.disable_commands
    
    if args.command_status:
        print("Enhanced Command Processing Status:")
        print(f"  Available: {ENHANCED_COMMANDS_AVAILABLE}")
        if ENHANCED_COMMANDS_AVAILABLE:
            try:
                bridge = LinguisticBridge()
                agent = EnhancedVocalFlowAgent()
                print(f"  Bridge Status: {bridge.get_status()}")
                print(f"  Agent Status: {agent.get_status()}")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print("  Reason: Enhanced components not found")
        return 0
    
    # Create configurations
    audio_config = AudioConfig()
    if args.audio_device is not None:
        audio_config.device_id = args.audio_device
    
    transcription_config = TranscriptionConfig()
    transcription_config.model_size = args.model
    transcription_config.language = args.language
    
    output_config = OutputConfig()
    output_config.method = args.output
    
    # Initialize system
    system = None
    try:
        logger.info("Starting VocalFlow Universal Voice Dictation System")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {platform.system()} {platform.release()}")
        
        system = VocalFlowSystem(
            profile_name=args.profile,
            audio_config=audio_config,
            transcription_config=transcription_config,
            output_config=output_config,
            enable_commands=enable_commands
        )
        
        system.start()
        
        # Print startup information
        print("\n" + "="*60)
        print("🎙️  VocalFlow - Universal Voice Dictation System")
        if enable_commands and ENHANCED_COMMANDS_AVAILABLE:
            print("🎯 Enhanced Command Processing: ENABLED")
        print("="*60)
        print(f"Profile: {args.profile}")
        print(f"Model: {args.model}")
        print(f"Language: {args.language}")
        print(f"Output: {args.output}")
        
        if enable_commands and ENHANCED_COMMANDS_AVAILABLE:
            command_status = system.get_command_status()
            print(f"Commands: {command_status.get('enhanced_enabled', False)}")
            print("-"*60)
            print("Available voice commands:")
            print("  • 'open browser' - Launch web browser")
            print("  • 'take screenshot' - Capture screen")
            print("  • 'volume up/down' - Control system volume")
            print("  • 'open terminal' - Launch terminal")
            print("  • Plus 227+ variations for natural speech!")
        
        print("-"*60)
        print("System is listening and learning...")
        print("Speak naturally - text will appear automatically")
        if enable_commands and ENHANCED_COMMANDS_AVAILABLE:
            print("Voice commands will be executed when detected")
        print("The system adapts to your voice patterns over time")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
            
    except Exception as e:
        logger.error(f"System error: {e}")
        if args.verbose:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
        
    finally:
        if system:
            system.stop()
            
        print("\nVocalFlow stopped. Thank you for using VocalFlow!")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
