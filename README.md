# VocalFlow

<div align="center">
  <h3>Universal Voice Dictation System with Enhanced Command Processing</h3>
  <p>Cross-platform voice-to-text with adaptive learning and intelligent voice commands</p>
  
  <p>
    <img src="https://img.shields.io/badge/status-stable-brightgreen?style=for-the-badge" alt="Stable Status">
    <img src="https://img.shields.io/badge/development-active-brightgreen?style=for-the-badge" alt="Active Development">
    <a href="#installation"><img src="https://img.shields.io/badge/Install-Quick%20Start-blue?style=for-the-badge" alt="Installation"></a>
  </p>
  
  <p>
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey" alt="Platform Support">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </p>
  
  <p>
    <strong>Production Ready</strong><br>
    Unified architecture with real-time voice dictation and intelligent command processing.<br>
    Linux is fully tested. Windows/macOS have basic support.
  </p>
</div>

## Overview

VocalFlow is a unified voice dictation system that combines real-time speech-to-text transcription with intelligent voice command processing. The system features adaptive learning, context awareness, and supports over 600 natural language command variations. It creates personalized user profiles that adapt to individual speech patterns and usage habits.

### Key Features

<table>
  <tr>
    <td><strong>Real-Time Voice Dictation</strong></td>
    <td>High-quality speech-to-text using Whisper models with adaptive voice learning</td>
  </tr>
  <tr>
    <td><strong>Intelligent Voice Commands</strong></td>
    <td>607+ natural language command variations with regional dialect support</td>
  </tr>
  <tr>
    <td><strong>Context Awareness</strong></td>
    <td>Automatically adapts to coding, writing, email, and documentation contexts</td>
  </tr>
  <tr>
    <td><strong>Auto-Completion System</strong></td>
    <td>Smart phrase completion and expansion with learning capabilities</td>
  </tr>
  <tr>
    <td><strong>Workflow Learning</strong></td>
    <td>Learns usage patterns and provides time-based workflow suggestions</td>
  </tr>
  <tr>
    <td><strong>Cross-Platform Support</strong></td>
    <td>Linux (fully tested), Windows/macOS (basic support)</td>
  </tr>
</table>

## Installation

### Prerequisites

- **Python**: 3.8+
- **Operating System**: Linux (Ubuntu 18.04+), Windows 10+, macOS 10.14+
- **Memory**: 4GB RAM minimum
- **Audio**: Working microphone

### Linux Installation (Recommended)

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv portaudio19-dev xdotool xclip gnome-screenshot

# Clone repository
git clone https://github.com/R3DK3LL/VocalFlow.git
cd VocalFlow

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
pip install pyyaml

# Test installation
python3 -m vocalflow.main --command-status
```

### Windows/macOS Installation

```bash
# Clone repository
git clone https://github.com/R3DK3LL/VocalFlow.git
cd VocalFlow

# Create virtual environment
python3 -m venv venv
# Windows: venv\Scripts\activate
# macOS: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install pyyaml

# Test basic functionality
python3 -m vocalflow.main --list-devices
```

## Quick Start

### Basic Voice Dictation

```bash
# Start dictation with default settings
python3 -m vocalflow.main

# Create named profile
python3 -m vocalflow.main --profile work

# Use clipboard output
python3 -m vocalflow.main --output clipboard
```

### Voice Commands Enabled

```bash
# Start with full command processing
python3 -m vocalflow.main --enable-commands

# Check command system status
python3 -m vocalflow.main --command-status

# Disable commands for dictation only
python3 -m vocalflow.main --disable-commands
```

### Available Voice Commands

Once running, you can use natural voice commands:

**Application Control:**
- "open browser" / "launch browser" / "fire up browser"
- "open terminal" / "launch terminal"
- "take screenshot" / "capture screen"

**System Control:**
- "volume up" / "increase volume" / "louder"
- "volume down" / "decrease volume" / "quieter"
- "volume mute" / "mute"

**Text Operations:**
- Natural dictation automatically appears as typed text
- Commands are executed instead of being transcribed

## Architecture

### Unified Processing Pipeline

```
Microphone → Audio Processing → Whisper Transcription → Enhanced Processing
                                                             ↓
                                               Is it a voice command?
                                                    ↙        ↘
                                          Execute Command    Format & Output Text
```

### Core Components

- **vocalflow/main.py**: Core dictation engine with integrated command processing
- **linguistic_bridge.py**: Command recognition and processing logic
- **enhanced_agents.py**: Intelligent agent system with context awareness
- **linguistic/**: Command variations database and matching engine

### Intelligent Features

**Context Awareness**: Automatically detects and adapts to:
- Coding contexts (Python, JavaScript, etc.)
- Email composition
- Documentation writing
- Creative writing

**Auto-Completion**: Expands common phrases:
- "good morning" → "Good morning, I hope you're doing well."
- "thank you for" → "Thank you for your time and consideration."
- "def main" → "def main():\n    pass"

**Workflow Learning**: Tracks patterns and suggests improvements based on usage time and frequency.

## User Profiles

### Profile Management

```bash
# List available profiles
python3 -m vocalflow.main --list-profiles

# Create new profile
python3 -m vocalflow.main --profile newuser

# Use existing profile
python3 -m vocalflow.main --profile work
```

### Profile Features

Profiles automatically learn and adapt:
- Voice characteristics and audio thresholds
- Common vocabulary and corrections
- Command usage patterns
- Context preferences

Profiles are stored in `~/.vocalflow/profiles/` with individual learning data.

## Command System

### Natural Language Processing

The system recognizes commands using:
- **607 command variations** across multiple categories
- **Regional dialects**: UK, US, Australian/New Zealand
- **Natural speech patterns**: including fillers, polite prefixes
- **Confidence scoring**: ensures accurate command recognition

### Command Categories

**Browser Operations**: Open/close web browsers with intelligent application discovery
**System Control**: Volume, brightness, screenshot functionality  
**File Operations**: Create, save, open files with context awareness
**Navigation**: Scroll, page navigation, window switching
**Text Editing**: Select, copy, paste, undo operations

### Adaptive Learning

When commands aren't recognized, the system:
- Analyzes failed attempts for learning opportunities
- Suggests appropriate command categories
- Provides confidence scores for suggestions
- Learns from user corrections over time

## Configuration Options

### Command Line Options

```bash
# Profile management
--profile NAME          Use specific voice profile
--list-profiles         Show available profiles

# Command processing
--enable-commands       Enable voice commands (default)
--disable-commands      Dictation only mode
--command-status        Show command system status

# Audio settings
--audio-device ID       Use specific audio device
--list-devices         Show available audio devices

# Output options
--output METHOD         clipboard, xdotool, wintype
--model SIZE           tiny, base, small, medium, large
--language CODE        Language for recognition (en, es, etc.)

# Debugging
--verbose              Enable detailed logging
```

### Advanced Configuration

The system supports extensive customization through:
- Custom command variations in YAML format
- Configurable confidence thresholds
- Context-specific vocabulary
- Learning rate adjustments

## Troubleshooting

### Common Issues

**Audio Device Problems:**
```bash
# List available devices
python3 -m vocalflow.main --list-devices

# Test specific device
python3 -m vocalflow.main --audio-device 0
```

**Command Recognition Issues:**
```bash
# Check command system
python3 -m vocalflow.main --command-status

# Test with verbose logging
python3 -m vocalflow.main --verbose
```

**Missing Dependencies:**
```bash
# Linux: Install system packages
sudo apt install portaudio19-dev xdotool xclip gnome-screenshot

# Python: Install missing packages
pip install pyyaml numpy sounddevice faster-whisper pyperclip
```

### Performance Optimization

- Use smaller Whisper models for faster processing
- Adjust audio device buffer sizes
- Configure appropriate confidence thresholds
- Use GPU acceleration when available

## Development

### Project Structure

```
VocalFlow/
├── vocalflow/              # Core dictation system
│   ├── main.py            # Main application with integrated processing
│   └── ...
├── linguistic/            # Command recognition system
│   ├── command_variations.yaml    # 607+ command variations
│   └── enhanced_command_matcher.py
├── enhanced_agents.py      # Complete intelligent agent system
├── linguistic_bridge.py   # Integration bridge
└── requirements.txt       # Python dependencies
```

### Adding Custom Commands

Extend the system by modifying `linguistic/command_variations.yaml`:

```yaml
command_categories:
  custom_operations:
    my_command:
      variations:
        - "custom phrase"
        - "alternative phrase" 
      base_regex: "(custom|alternative).*phrase"
      confidence_threshold: 0.7
```

### API Usage

```python
from vocalflow.main import VocalFlowSystem
from enhanced_agents import EnhancedVocalFlowAgent

# Create system with enhanced commands
system = VocalFlowSystem(enable_commands=True)
system.start()

# Or use agent directly
agent = EnhancedVocalFlowAgent()
result = agent.process_text_enhanced("open browser")
```

## Contributing

Contributions are welcome, particularly for:
- Windows/macOS platform improvements
- Additional command variations and languages
- Performance optimizations
- Documentation enhancements

### Development Setup

```bash
git clone https://github.com/R3DK3LL/VocalFlow.git
cd VocalFlow
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pyyaml

# Run tests
python3 -m vocalflow.main --command-status
python3 -m vocalflow.main --verbose --disable-commands
```

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- Audio input device
- 2GB disk space (including Whisper models)

### Recommended Requirements
- Python 3.10+
- 8GB RAM
- Quality microphone
- GPU for faster processing
- Linux environment for full feature support

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for efficient speech recognition
- [sounddevice](https://github.com/spatialaudio/python-sounddevice) for cross-platform audio
- Computational linguistics research for natural language processing techniques

---

<div align="center">
  <p><strong>A production-ready voice dictation system with intelligent command processing</strong></p>
  <p>
    <a href="https://github.com/R3DK3LL/VocalFlow">GitHub</a> •
    <a href="https://github.com/R3DK3LL/VocalFlow/issues">Issues</a> •
    <a href="https://github.com/R3DK3LL/VocalFlow/discussions">Discussions</a>
  </p>
</div></content>
