# VocalFlow

<div align="center">
  <h3>Universal Voice Dictation System with Adaptive Learning</h3>
  <p>Cross-platform voice-to-text that learns and adapts to individual speech patterns</p>
  
  <p>
    <a href="#installation"><img src="https://img.shields.io/badge/Install-Quick%20Start-blue?style=for-the-badge" alt="Installation"></a>
    <a href="#usage"><img src="https://img.shields.io/badge/Usage-Documentation-green?style=for-the-badge" alt="Usage"></a>
    <a href="#contributing"><img src="https://img.shields.io/badge/Contribute-Guidelines-orange?style=for-the-badge" alt="Contributing"></a>
  </p>
  
  <p>
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" alt="Platform Support">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </p>
</div>

## Overview

VocalFlow is an advanced voice dictation system that transcribes speech to text in real-time while continuously learning from individual users. Unlike traditional speech-to-text solutions, VocalFlow creates personalized voice profiles that improve accuracy and adapt to unique speech patterns, vocabulary, and environmental conditions.

### Key Features

<table>
  <tr>
    <td><strong>Universal Compatibility</strong></td>
    <td>Native support for Windows, Linux, and macOS with automatic platform detection</td>
  </tr>
  <tr>
    <td><strong>Adaptive Learning Engine</strong></td>
    <td>Continuously learns speech patterns, vocabulary, and environmental audio characteristics</td>
  </tr>
  <tr>
    <td><strong>Multi-User Profiles</strong></td>
    <td>Individual voice profiles with isolated learning data and preferences</td>
  </tr>
  <tr>
    <td><strong>Real-Time Processing</strong></td>
    <td>Low-latency transcription with optimized voice activity detection</td>
  </tr>
  <tr>
    <td><strong>Smart Filtering</strong></td>
    <td>Automatic removal of filler words, artifacts, and background noise</td>
  </tr>
  <tr>
    <td><strong>Multiple Output Methods</strong></td>
    <td>Direct typing, clipboard integration, or custom output handlers</td>
  </tr>
</table>

## Installation

### Prerequisites

<details>
<summary><strong>System Requirements</strong></summary>

- **Operating System**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.14+
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Audio**: Microphone with 16kHz+ sample rate support
- **Storage**: 2GB free space for models and profiles

</details>

### Platform-Specific Setup

<details>
<summary><strong>Linux Installation</strong></summary>

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv portaudio19-dev xdotool xclip

# Clone repository
git clone https://github.com/yourusername/VocalFlow.git
cd VocalFlow

# Create virtual environment
python3 -m venv vocalflow-env
source vocalflow-env/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -m vocalflow.main --help
```

</details>

<details>
<summary><strong>macOS Installation</strong></summary>

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install portaudio python3

# Clone repository
git clone https://github.com/yourusername/VocalFlow.git
cd VocalFlow

# Create virtual environment
python3 -m venv vocalflow-env
source vocalflow-env/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -m vocalflow.main --help
```

</details>

<details>
<summary><strong>Windows Installation</strong></summary>

```powershell
# Install Python 3.8+ from python.org if not installed

# Clone repository
git clone https://github.com/yourusername/VocalFlow.git
cd VocalFlow

# Create virtual environment
python -m venv vocalflow-env
vocalflow-env\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install Windows-specific packages
pip install pyautogui

# Verify installation
python -m vocalflow.main --help
```

</details>

### Quick Install Script

```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/VocalFlow/main/scripts/install.sh | bash
```

## Usage

### Basic Operation

```bash
# Start with default profile
python -m vocalflow.main

# Start with named profile
python -m vocalflow.main --profile john

# Use clipboard output (cross-platform)
python -m vocalflow.main --output clipboard

# List available profiles
python -m vocalflow.main --list-profiles
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--profile`, `-p` | Voice profile name | `default` |
| `--output`, `-o` | Output method (`auto`, `clipboard`, `xdotool`, `wintype`) | `auto` |
| `--model`, `-m` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) | `base` |
| `--language`, `-l` | Recognition language code | `en` |
| `--list-profiles` | Show all available profiles | - |
| `--verbose`, `-v` | Enable verbose logging | - |

### Configuration

VocalFlow automatically creates configuration directories:

```
~/.vocalflow/
├── profiles/           # Voice profiles
│   ├── default.json
│   └── username.json
├── logs/              # System logs
│   └── vocalflow.log
└── models/            # Cached Whisper models
```

## Adaptive Learning System

### Voice Profile Components

VocalFlow maintains comprehensive user profiles containing:

<table>
  <tr>
    <th>Component</th>
    <th>Learning Method</th>
    <th>Purpose</th>
  </tr>
  <tr>
    <td><strong>Audio Characteristics</strong></td>
    <td>Statistical analysis of volume, tone, and pace</td>
    <td>Optimize voice activity detection and gain control</td>
  </tr>
  <tr>
    <td><strong>Environmental Adaptation</strong></td>
    <td>Background noise profiling and threshold adjustment</td>
    <td>Improve speech detection in various acoustic environments</td>
  </tr>
  <tr>
    <td><strong>Vocabulary Learning</strong></td>
    <td>Frequency analysis of words and phrases</td>
    <td>Enhance transcription accuracy for commonly used terms</td>
  </tr>
  <tr>
    <td><strong>Correction Patterns</strong></td>
    <td>User correction tracking and pattern recognition</td>
    <td>Automatically apply learned fixes to common transcription errors</td>
  </tr>
  <tr>
    <td><strong>Speech Timing</strong></td>
    <td>Analysis of natural pause lengths and speech cadence</td>
    <td>Improve sentence boundary detection and processing timing</td>
  </tr>
</table>

### Learning Process

1. **Initial Calibration**: System analyzes first 30 seconds of audio to establish baseline parameters
2. **Continuous Adaptation**: Real-time adjustment of thresholds and parameters during use
3. **Session Learning**: Collection and analysis of vocabulary, corrections, and speech patterns
4. **Profile Updates**: Automatic saving of learned parameters at session end
5. **Cross-Session Memory**: Persistent storage and retrieval of user-specific optimizations

### Logging and Analytics

VocalFlow maintains detailed logs for performance monitoring and debugging:

- **Audio Quality Metrics**: Signal-to-noise ratios, volume levels, detection accuracy
- **Transcription Performance**: Processing times, confidence scores, error rates
- **Learning Progress**: Profile evolution, adaptation effectiveness, vocabulary growth
- **System Performance**: Resource usage, processing latency, error tracking

## API Reference

### Core Classes

```python
from vocalflow import VocalFlowSystem, VoiceProfile

# Initialize system with custom profile
system = VocalFlowSystem("user_profile")

# Configure transcription parameters
system.transcription_config.model_size = "small"
system.transcription_config.language = "en"

# Set output method
system.output_config.method = "clipboard"

# Start dictation
system.start()
```

### Profile Management

```python
from vocalflow.profiles import ProfileManager

manager = ProfileManager()

# Create new profile
profile = manager.create_profile("new_user")

# Load existing profile
profile = manager.load_profile("existing_user")

# List all profiles
profiles = manager.list_profiles()
```

## Performance Optimization

### Model Selection

| Model Size | Speed | Accuracy | Memory Usage | Recommended Use |
|------------|-------|----------|--------------|-----------------|
| `tiny` | Fastest | Good | 39 MB | Resource-constrained systems |
| `base` | Fast | Better | 74 MB | **Default - balanced performance** |
| `small` | Medium | Better | 244 MB | Higher accuracy needs |
| `medium` | Slower | Best | 769 MB | Professional transcription |
| `large` | Slowest | Excellent | 1550 MB | Maximum accuracy requirements |

### Hardware Recommendations

- **CPU**: Multi-core processor (Intel i5+ or AMD Ryzen 5+ recommended)
- **RAM**: 8GB+ for optimal performance with larger models
- **Audio**: USB or XLR microphone for best recognition accuracy
- **Storage**: SSD recommended for faster model loading

## Troubleshooting

<details>
<summary><strong>Common Issues</strong></summary>

### Audio Device Not Found
```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Specify device ID
python -m vocalflow.main --audio-device 1
```

### Permission Errors (Linux)
```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# Logout and login again
```

### Model Download Issues
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/
rm -rf ~/.vocalflow/models/

# Retry with verbose logging
python -m vocalflow.main --verbose
```

</details>

## Contributing

We welcome contributions from the community. Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Development Setup

```bash
# Clone development branch
git clone -b develop https://github.com/yourusername/VocalFlow.git
cd VocalFlow

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 vocalflow/
black vocalflow/
```

### Issue Reporting

When reporting issues, please include:
- Operating system and version
- Python version
- Complete error messages
- Steps to reproduce
- Audio device information

## Roadmap

- [ ] GUI interface for non-technical users
- [ ] Custom vocabulary import/export
- [ ] Multiple language support within single profiles
- [ ] Cloud profile synchronization
- [ ] Integration with popular text editors
- [ ] Voice command recognition
- [ ] Batch audio file transcription

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [faster-whisper](https://github.com/guillaumekln/faster-whisper) for high-performance speech recognition
- Uses [sounddevice](https://github.com/spatialaudio/python-sounddevice) for cross-platform audio capture
- Inspired by the need for personalized, adaptive voice recognition systems

---

<div align="center">
  <p><strong>Made with precision for developers, writers, and voice-first productivity</strong></p>
  <p>
    <a href="https://github.com/yourusername/VocalFlow">GitHub</a> •
    <a href="https://github.com/yourusername/VocalFlow/issues">Issues</a> •
    <a href="https://github.com/yourusername/VocalFlow/discussions">Discussions</a>
  </p>
</div>
