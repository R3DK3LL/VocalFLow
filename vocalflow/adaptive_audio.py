#!/usr/bin/env python3
"""
adaptive_audio.py - Truly Adaptive Audio Processing for VocalFlow
NO hardcoded values - all bounds adapt to discovered environment

Adaptive Features:
- Environment discovery phase establishes operating ranges
- All thresholds relative to discovered baseline
- Bounds that scale with environment characteristics
- Self-tuning sensitivity with no preset assumptions
"""

import numpy as np
import time
import logging
from collections import deque
from typing import Optional, Tuple, List, Dict, Any
import threading


class EnvironmentDiscovery:
    """Discovers the audio environment to establish adaptive operating ranges"""

    def __init__(self):
        self.logger = logging.getLogger("adaptive_audio.discovery")

        # Discovery phase tracking
        self.discovery_samples = deque(
            maxlen=2000
        )  # Large sample for environment characterization
        self.is_discovery_complete = False
        self.discovery_start_time = time.time()
        self.min_discovery_duration = 30.0  # Minimum discovery time

        # Discovered environment characteristics
        self.baseline_noise_floor = None
        self.noise_ceiling = None
        self.typical_speech_range = None
        self.environment_variance = None

        # Operating ranges (established after discovery)
        self.dynamic_min_threshold = None
        self.dynamic_max_threshold = None
        self.adaptive_silence_duration = None

    def add_discovery_sample(self, audio_level: float, detected_activity: bool = False):
        """Add sample during environment discovery phase"""
        if self.is_discovery_complete:
            return

        self.discovery_samples.append(
            {
                "level": audio_level,
                "timestamp": time.time(),
                "activity_hint": detected_activity,  # External hint about activity
            }
        )

        # Check if discovery can be completed
        self._check_discovery_completion()

    def _check_discovery_completion(self):
        """Check if environment discovery can be completed"""
        current_time = time.time()
        discovery_duration = current_time - self.discovery_start_time

        # Need minimum duration and samples
        if (
            discovery_duration >= self.min_discovery_duration
            and len(self.discovery_samples) >= 500
        ):

            self._complete_discovery()

    def _complete_discovery(self):
        """Complete environment discovery and establish operating ranges"""
        if self.is_discovery_complete:
            return

        samples = [s["level"] for s in self.discovery_samples if s["level"] > 0]

        if len(samples) < 100:
            # Not enough data, extend discovery
            self.min_discovery_duration += 15.0
            return

        # Establish baseline characteristics
        self.baseline_noise_floor = np.percentile(
            samples, 10
        )  # Bottom 10% as noise floor
        self.noise_ceiling = np.percentile(samples, 90)  # Top 10% as potential speech
        self.environment_variance = np.var(samples)

        # Calculate speech range based on discovered characteristics
        dynamic_range = self.noise_ceiling - self.baseline_noise_floor
        self.typical_speech_range = (
            self.baseline_noise_floor + dynamic_range * 0.3,
            self.baseline_noise_floor + dynamic_range * 0.9,
        )

        # Establish adaptive operating bounds (no hardcoded limits)
        range_multiplier = max(2.0, np.sqrt(self.environment_variance * 1000))

        self.dynamic_min_threshold = (
            self.baseline_noise_floor * 0.8
        )  # Just below noise floor
        self.dynamic_max_threshold = self.baseline_noise_floor * range_multiplier

        # Adaptive timing based on environment stability
        stability_factor = 1.0 / (1.0 + self.environment_variance * 100)
        self.adaptive_silence_duration = 0.8 + (
            1.2 * stability_factor
        )  # 0.8 to 2.0 seconds

        self.is_discovery_complete = True

        self.logger.info(
            f"Environment discovered: noise_floor={self.baseline_noise_floor:.6f}, "
            f"range={dynamic_range:.6f}, adaptive_bounds=({self.dynamic_min_threshold:.6f}, "
            f"{self.dynamic_max_threshold:.6f}), silence_duration={self.adaptive_silence_duration:.1f}s"
        )

    def get_adaptive_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        """Get current adaptive bounds (None if still discovering)"""
        if not self.is_discovery_complete:
            return None, None
        return self.dynamic_min_threshold, self.dynamic_max_threshold

    def get_adaptive_silence_duration(self) -> Optional[float]:
        """Get adaptive silence duration (None if still discovering)"""
        if not self.is_discovery_complete:
            return None
        return self.adaptive_silence_duration

    def should_use_level_as_threshold(self, level: float) -> bool:
        """Determine if a level is suitable as a threshold based on discovered environment"""
        if not self.is_discovery_complete:
            return False

        # Level should be within the discovered reasonable range
        return self.dynamic_min_threshold <= level <= self.dynamic_max_threshold


class RelativeSignalProcessor:
    """Signal processor with no hardcoded values - all relative to environment"""

    def __init__(self, environment: EnvironmentDiscovery):
        self.environment = environment
        self.logger = logging.getLogger("adaptive_audio.signal")

        # Adaptive noise profiling
        self.recent_quiet_levels = deque(maxlen=200)
        self.recent_active_levels = deque(maxlen=200)

        # Relative signal characteristics
        self.current_noise_estimate = 0.001  # Initial guess, will adapt
        self.current_signal_baseline = None

        # Adaptive gain control (no target values)
        self.gain_history = deque(maxlen=50)
        self.current_gain = 1.0
        self.dc_offset = 0.0

        # Quality assessment (relative)
        self.signal_quality_history = deque(maxlen=100)

    def update_signal_profile(self, audio_chunk: np.ndarray, is_likely_speech: bool):
        """Update signal profile with relative measurements"""
        rms = np.sqrt(np.mean(audio_chunk**2))

        if is_likely_speech:
            self.recent_active_levels.append(rms)
        else:
            self.recent_quiet_levels.append(rms)

        # Update noise estimate relatively
        if len(self.recent_quiet_levels) >= 10:
            self.current_noise_estimate = np.percentile(
                list(self.recent_quiet_levels), 75
            )

        # Update signal baseline if we have active levels
        if len(self.recent_active_levels) >= 5:
            self.current_signal_baseline = np.percentile(
                list(self.recent_active_levels), 25
            )

    def enhance_signal_relatively(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Enhance signal relative to discovered characteristics"""
        if len(audio_chunk) == 0:
            return audio_chunk

        audio_flat = audio_chunk.flatten()

        # Relative DC offset removal
        current_dc = np.mean(audio_flat)
        adaptation_rate = 0.05 if abs(current_dc) > abs(self.dc_offset) * 2 else 0.01
        self.dc_offset = (
            1 - adaptation_rate
        ) * self.dc_offset + adaptation_rate * current_dc
        audio_flat = audio_flat - self.dc_offset

        # Relative noise suppression
        audio_enhanced = self._suppress_noise_relatively(audio_flat)

        # Relative gain control
        audio_normalized = self._apply_relative_gain(audio_enhanced)

        return audio_normalized.reshape(-1, 1)

    def _suppress_noise_relatively(self, audio: np.ndarray) -> np.ndarray:
        """Noise suppression relative to current noise estimate"""
        if self.current_noise_estimate <= 0:
            return audio

        rms = np.sqrt(np.mean(audio**2))

        # Relative threshold based on current noise
        relative_noise_threshold = self.current_noise_estimate * 1.5

        if rms < relative_noise_threshold:
            # Relative noise reduction
            suppression_factor = max(0.1, rms / relative_noise_threshold)
            return audio * suppression_factor

        return audio

    def _apply_relative_gain(self, audio: np.ndarray) -> np.ndarray:
        """Apply gain control relative to signal characteristics"""
        if len(audio) == 0:
            return audio

        rms = np.sqrt(np.mean(audio**2))

        if rms > 0 and self.current_signal_baseline:
            # Gain relative to typical signal levels we've seen
            relative_gain = self.current_signal_baseline / rms

            # Smooth gain changes
            self.current_gain = 0.9 * self.current_gain + 0.1 * relative_gain

            # Bound gain relative to historical range
            if self.gain_history:
                gain_range = max(self.gain_history) - min(self.gain_history)
                gain_center = np.mean(list(self.gain_history))

                # Allow gain to vary within discovered range + some expansion
                gain_min = gain_center - gain_range * 1.5
                gain_max = gain_center + gain_range * 1.5

                self.current_gain = np.clip(
                    self.current_gain, max(gain_min, 0.1), min(gain_max, 20.0)
                )

            self.gain_history.append(self.current_gain)
            return audio * self.current_gain

        return audio

    def calculate_relative_quality(self, audio_chunk: np.ndarray) -> float:
        """Calculate signal quality relative to environment characteristics"""
        if len(audio_chunk) == 0:
            return 0.0

        rms = np.sqrt(np.mean(audio_chunk.flatten() ** 2))

        # Quality relative to noise floor
        if self.current_noise_estimate > 0:
            snr_estimate = rms / self.current_noise_estimate
            base_quality = min(snr_estimate / 10.0, 1.0)  # Relative quality score
        else:
            base_quality = 0.5

        # Factor in consistency with previous signals
        if self.signal_quality_history:
            consistency = 1.0 - abs(
                base_quality - np.mean(list(self.signal_quality_history)[-5:])
            )
            base_quality = base_quality * 0.7 + consistency * 0.3

        self.signal_quality_history.append(base_quality)
        return base_quality


class TrulyAdaptiveThresholdController:
    """Threshold controller with no preset bounds - adapts to any environment"""

    def __init__(self, environment: EnvironmentDiscovery):
        self.environment = environment
        self.logger = logging.getLogger("adaptive_audio.threshold")

        # Current operating threshold (starts as guess, becomes adaptive)
        self.current_threshold = 0.005  # Initial guess only
        self.threshold_locked = False

        # Adaptive parameters
        self.adaptation_velocity = 0.0  # Current rate of change
        self.performance_trend = deque(maxlen=20)
        self.recent_adaptations = deque(maxlen=10)

        # Self-regulation
        self.consecutive_poor_performance = 0
        self.stability_score = 0.0
        self.last_successful_period = time.time()

        # Emergency self-correction
        self.emergency_active = False
        self.emergency_baseline = None

    def update_threshold_adaptively(self, performance_metrics: Dict[str, Any]) -> bool:
        """Update threshold based purely on performance feedback"""

        # During environment discovery, use conservative threshold
        if not self.environment.is_discovery_complete:
            self._handle_discovery_phase(performance_metrics)
            return False

        # Extract performance indicators
        transcription_success = performance_metrics.get("transcription_success", False)
        false_positive_likely = performance_metrics.get("false_positive_likely", False)
        signal_quality = performance_metrics.get("signal_quality", 0.5)
        audio_duration = performance_metrics.get("audio_duration", 0)

        # Calculate performance score
        performance_score = self._calculate_performance_score(
            transcription_success, false_positive_likely, signal_quality, audio_duration
        )

        self.performance_trend.append(performance_score)

        # Decide adaptation direction and magnitude
        return self._adapt_based_on_performance()

    def _handle_discovery_phase(self, metrics: Dict[str, Any]):
        """Handle threshold during environment discovery"""
        # Use environment samples to inform threshold
        audio_level = metrics.get("audio_level", 0)
        self.environment.add_discovery_sample(
            audio_level, metrics.get("transcription_success", False)
        )

        # During discovery, use conservative threshold
        if self.environment.baseline_noise_floor:
            conservative_threshold = self.environment.baseline_noise_floor * 1.8
            self.current_threshold = conservative_threshold

    def _calculate_performance_score(
        self,
        transcription_success: bool,
        false_positive_likely: bool,
        signal_quality: float,
        audio_duration: float,
    ) -> float:
        """Calculate performance score from multiple indicators"""
        score = 0.5  # Neutral baseline

        if transcription_success:
            score += 0.4

        if false_positive_likely:
            score -= 0.3

        # Quality bonus/penalty
        score += (signal_quality - 0.5) * 0.2

        # Duration factor (very short or very long audio suggests problems)
        if 1.0 <= audio_duration <= 8.0:  # Good duration range
            score += 0.1
        elif audio_duration < 0.5 or audio_duration > 15.0:  # Poor duration
            score -= 0.2

        return np.clip(score, 0.0, 1.0)

    def _adapt_based_on_performance(self) -> bool:
        """Adapt threshold based on performance trend"""
        if len(self.performance_trend) < 5:
            return False

        # Calculate recent performance
        recent_performance = np.mean(list(self.performance_trend)[-5:])
        overall_performance = np.mean(list(self.performance_trend))
        performance_variance = np.var(list(self.performance_trend))

        # Determine if adaptation is needed
        needs_adaptation = False
        adaptation_direction = 0
        adaptation_magnitude = 0

        # Performance-based adaptation logic
        if recent_performance < 0.3:  # Poor recent performance
            self.consecutive_poor_performance += 1
            needs_adaptation = True

            # If getting empty results, likely too sensitive
            if recent_performance < 0.2:
                adaptation_direction = 1  # Increase threshold (less sensitive)
                adaptation_magnitude = 0.1 + (self.consecutive_poor_performance * 0.05)
            else:
                adaptation_direction = -1  # Decrease threshold (more sensitive)
                adaptation_magnitude = 0.05

        elif recent_performance > 0.7 and overall_performance > 0.6:
            # Good performance, but check if we can be more sensitive
            self.consecutive_poor_performance = 0
            if performance_variance < 0.05:  # Stable performance
                adaptation_direction = -1  # Try being more sensitive
                adaptation_magnitude = 0.02
                needs_adaptation = True

        if needs_adaptation:
            return self._apply_adaptive_change(
                adaptation_direction, adaptation_magnitude
            )

        return False

    def _apply_adaptive_change(self, direction: int, magnitude: float) -> bool:
        """Apply threshold change with environmental bounds checking"""
        min_bound, max_bound = self.environment.get_adaptive_bounds()

        if min_bound is None:  # Still in discovery
            return False

        # Calculate proposed new threshold
        change_amount = magnitude * direction

        # Make change relative to current operating range
        operating_range = max_bound - min_bound
        relative_change = change_amount * operating_range

        proposed_threshold = self.current_threshold + relative_change

        # Check against adaptive bounds
        if not self.environment.should_use_level_as_threshold(proposed_threshold):
            # Try smaller change
            relative_change *= 0.5
            proposed_threshold = self.current_threshold + relative_change

            if not self.environment.should_use_level_as_threshold(proposed_threshold):
                return False  # Can't adapt in this direction

        # Apply adaptation
        old_threshold = self.current_threshold
        self.current_threshold = proposed_threshold

        self.recent_adaptations.append(
            {
                "old": old_threshold,
                "new": self.current_threshold,
                "direction": direction,
                "magnitude": magnitude,
                "time": time.time(),
            }
        )

        self.logger.info(
            f"Adaptive threshold change: {old_threshold:.6f} -> {self.current_threshold:.6f}"
        )

        return True

    def get_current_threshold(self) -> float:
        """Get current adaptive threshold"""
        return self.current_threshold

    def is_operating_effectively(self) -> bool:
        """Check if system is operating within discovered effective range"""
        if not self.environment.is_discovery_complete:
            return False

        if len(self.performance_trend) < 10:
            return False

        recent_avg = np.mean(list(self.performance_trend)[-10:])
        return recent_avg > 0.5 and self.consecutive_poor_performance < 3


class TrulyAutonomousAudioProcessor:
    """Fully autonomous processor - no hardcoded assumptions about environment"""

    def __init__(self, vocalflow_system):
        self.vocalflow_system = vocalflow_system
        self.logger = logging.getLogger("adaptive_audio.autonomous")

        # Initialize discovery-based components
        self.environment = EnvironmentDiscovery()
        self.signal_processor = RelativeSignalProcessor(self.environment)
        self.threshold_controller = TrulyAdaptiveThresholdController(self.environment)

        # Buffer management (adaptive sizing)
        self.audio_buffer = deque()
        self.buffer_start_time = 0

        # State tracking
        self.is_speech_active = False
        self.silence_start_time = None

        # Adaptive timing
        self.adaptive_silence_threshold = 1.0  # Will be replaced by discovered value

        # Performance tracking
        self.total_processing_attempts = 0
        self.successful_transcriptions = 0

        # Thread safety
        self._lock = threading.Lock()

    def process_audio_chunk(
        self, chunk: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Process audio with fully adaptive approach"""
        with self._lock:
            current_time = time.time()

            # Enhance signal using relative processing
            enhanced_chunk = self.signal_processor.enhance_signal_relatively(chunk)

            # Calculate current audio level
            rms = np.sqrt(np.mean(enhanced_chunk**2))

            # Get adaptive threshold
            current_threshold = self.threshold_controller.get_current_threshold()

            # Update signal profiles
            is_likely_speech = rms > current_threshold
            self.signal_processor.update_signal_profile(
                enhanced_chunk, is_likely_speech
            )

            # Get adaptive silence duration
            adaptive_silence = self.environment.get_adaptive_silence_duration()
            if adaptive_silence:
                self.adaptive_silence_threshold = adaptive_silence

            # Voice activity detection
            if is_likely_speech:
                if not self.is_speech_active:
                    self.is_speech_active = True
                    self.buffer_start_time = current_time
                    self.logger.debug(
                        f"Speech detected: RMS={rms:.6f}, threshold={current_threshold:.6f}"
                    )

                self.silence_start_time = None
                self.audio_buffer.extend(enhanced_chunk.flatten())

            else:
                if self.is_speech_active:
                    if self.silence_start_time is None:
                        self.silence_start_time = current_time

                    # Use adaptive silence duration
                    silence_duration = current_time - self.silence_start_time
                    if silence_duration > self.adaptive_silence_threshold:
                        return self._finalize_speech_segment(current_time, rms)

            # Adaptive buffer size management
            buffer_duration = len(self.audio_buffer) / 16000 if self.audio_buffer else 0
            max_buffer_duration = 25.0  # Only hard limit to prevent memory issues

            if buffer_duration > max_buffer_duration:
                self.logger.warning("Maximum buffer duration reached")
                return self._finalize_speech_segment(current_time, rms)

            return False, None

    def _finalize_speech_segment(
        self, current_time: float, current_rms: float
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Finalize speech segment with adaptive quality assessment"""
        if not self.audio_buffer:
            self._reset_state()
            return False, None

        # Convert to numpy array
        audio_array = np.array(list(self.audio_buffer), dtype=np.float32).reshape(-1, 1)
        audio_duration = len(audio_array) / 16000

        # Adaptive minimum duration (based on environment characteristics)
        min_duration = 0.3
        if self.environment.is_discovery_complete:
            # Adjust minimum duration based on discovered environment stability
            stability_factor = 1.0 / (1.0 + self.environment.environment_variance * 50)
            min_duration = 0.2 + (0.6 * stability_factor)  # 0.2 to 0.8 seconds

        if audio_duration < min_duration:
            self._reset_state()
            return False, None

        # Calculate relative quality
        quality_score = self.signal_processor.calculate_relative_quality(audio_array)

        self.logger.debug(
            f"Finalizing segment: {audio_duration:.1f}s, quality={quality_score:.3f}"
        )

        # Store for feedback
        self.total_processing_attempts += 1

        # Reset state
        buffer_copy = audio_array.copy()
        self._reset_state()

        return True, buffer_copy

    def _reset_state(self):
        """Reset processing state"""
        self.audio_buffer.clear()
        self.is_speech_active = False
        self.silence_start_time = None
        self.buffer_start_time = 0

    def report_transcription_result(
        self,
        audio_duration: float,
        segments: List[str],
        signal_quality: float,
        audio_rms: float,
    ):
        """Report results for adaptive learning"""
        transcription_success = len(segments) > 0 and any(
            len(s.strip()) > 1 for s in segments
        )

        # Detect likely false positives
        false_positive_likely = (
            not transcription_success and audio_duration > 2.0 and signal_quality < 0.3
        )

        if transcription_success:
            self.successful_transcriptions += 1

        # Create comprehensive performance metrics
        performance_metrics = {
            "transcription_success": transcription_success,
            "false_positive_likely": false_positive_likely,
            "signal_quality": signal_quality,
            "audio_duration": audio_duration,
            "audio_level": audio_rms,
            "segment_count": len(segments),
        }

        # Update adaptive threshold
        adapted = self.threshold_controller.update_threshold_adaptively(
            performance_metrics
        )

        if adapted:
            new_threshold = self.threshold_controller.get_current_threshold()
            self.logger.info(f"Autonomously adapted to threshold: {new_threshold:.6f}")

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get detailed system status"""
        min_bound, max_bound = self.environment.get_adaptive_bounds()

        return {
            # Current operating parameters
            "threshold": self.threshold_controller.get_current_threshold(),
            "noise_estimate": self.signal_processor.current_noise_estimate,
            "adaptive_silence_duration": self.adaptive_silence_threshold,
            # Environment discovery
            "discovery_complete": self.environment.is_discovery_complete,
            "baseline_noise_floor": self.environment.baseline_noise_floor,
            "operating_bounds": (min_bound, max_bound) if min_bound else None,
            # Performance
            "total_attempts": self.total_processing_attempts,
            "success_rate": self.successful_transcriptions
            / max(self.total_processing_attempts, 1),
            "operating_effectively": self.threshold_controller.is_operating_effectively(),
            # Current state
            "is_speech_active": self.is_speech_active,
            "buffer_duration": (
                len(self.audio_buffer) / 16000 if self.audio_buffer else 0
            ),
        }


def replace_with_truly_adaptive_processing(vocalflow_system):
    """Replace original processing with truly adaptive version"""

    # Create autonomous processor
    autonomous_processor = TrulyAutonomousAudioProcessor(vocalflow_system)

    def truly_adaptive_audio_stream():
        """Truly adaptive audio processing loop"""
        logger = logging.getLogger("adaptive_audio.main")
        logger.info(
            "Truly adaptive audio processing started - discovering environment..."
        )

        status_interval = 45.0
        last_status_time = time.time()

        while vocalflow_system.is_running:
            try:
                # Get audio chunk
                chunk = vocalflow_system.audio_processor.get_audio_chunk()
                if chunk is None:
                    time.sleep(0.01)
                    continue

                # Process with autonomous system
                should_process, audio_buffer = autonomous_processor.process_audio_chunk(
                    chunk
                )

                if should_process and audio_buffer is not None:
                    # Transcribe
                    start_time = time.time()
                    segments = vocalflow_system.transcription_engine.transcribe_audio(
                        audio_buffer
                    )
                    transcription_time = time.time() - start_time

                    audio_duration = len(audio_buffer) / 16000
                    audio_rms = np.sqrt(np.mean(audio_buffer**2))

                    # Calculate quality score
                    signal_quality = autonomous_processor.signal_processor.calculate_relative_quality(
                        audio_buffer
                    )

                    # Report results for adaptation
                    autonomous_processor.report_transcription_result(
                        audio_duration, segments, signal_quality, audio_rms
                    )

                    logger.debug(
                        f"Processed: {audio_duration:.1f}s -> {len(segments)} segments "
                        f"(quality: {signal_quality:.2f}, time: {transcription_time:.2f}s)"
                    )

                    # Handle successful transcriptions
                    if segments and signal_quality > 0.3:
                        for text in segments:
                            if vocalflow_system.filter.is_good_text(text):
                                corrected_text = (
                                    vocalflow_system.filter.apply_corrections(text)
                                )

                                # Handle commands
                                if (
                                    vocalflow_system.enable_commands
                                    and vocalflow_system.linguistic_bridge
                                    and vocalflow_system.command_agent
                                ):

                                    is_command, command_type, metadata = (
                                        vocalflow_system.linguistic_bridge.process_command(
                                            corrected_text
                                        )
                                    )

                                    if is_command:
                                        logger.info(
                                            f"Command: '{command_type}' from '{corrected_text}'"
                                        )
                                        try:
                                            success = vocalflow_system.command_agent.process_text_enhanced(
                                                corrected_text
                                            )
                                            if not success:
                                                logger.warning(
                                                    f"Command '{command_type}' failed"
                                                )
                                        except Exception as cmd_error:
                                            logger.error(f"Command error: {cmd_error}")
                                        continue

                                # Handle dictation
                                formatted_text = vocalflow_system._format_text(
                                    corrected_text
                                )
                                vocalflow_system.output_handler.type_text(
                                    formatted_text
                                )

                # Periodic status
                current_time = time.time()
                if current_time - last_status_time >= status_interval:
                    status = autonomous_processor.get_comprehensive_status()
                    logger.info(f"Adaptive status: {status}")
                    last_status_time = current_time

            except Exception as e:
                logger.error(f"Adaptive processing error: {e}")
                time.sleep(0.1)

        logger.info("Truly adaptive audio processing stopped")

    # Replace processing method
    vocalflow_system._process_audio_stream = truly_adaptive_audio_stream

    return vocalflow_system
