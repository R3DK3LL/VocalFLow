# VocalFlow

<div align="center">
  <h3>Voice Dictation System with Enhanced Linguistic Processing</h3>
  <p>Cross-platform voice-to-text with adaptive command recognition and intelligent agents</p>
  
  <p>
    <img src="https://img.shields.io/badge/status-beta-yellow?style=for-the-badge" alt="Beta Status">
    <img src="https://img.shields.io/badge/development-active-brightgreen?style=for-the-badge" alt="Active Development">
    <a href="#installation"><img src="https://img.shields.io/badge/Install-Quick%20Start-blue?style=for-the-badge" alt="Installation"></a>
  </p>
  
  <p>
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey" alt="Platform Support">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </p>
  
  <p>
    <strong>⚠️ BETA SOFTWARE</strong><br>
    This project is in active development. Core functionality is stable but some features are experimental.<br>
    Linux is the primary development platform with the most testing.
  </p>
</div>

## Overview

VocalFlow is a voice dictation system that transcribes speech to text while learning user patterns. It includes an enhanced linguistic processing system that can recognize voice commands using 607+ command variations with adaptive learning capabilities. The system creates user profiles that adapt to individual speech patterns and usage.

### Current Features

<table>
  <tr>
    <td><strong>Voice Dictation</strong></td>
    <td>Real-time speech-to-text using Whisper models with user profile adaptation</td>
  </tr>
  <tr>bet
    <td><strong>Enhanced Command Recognition</strong></td>
    <td>607 command variations with regional dialect support (UK, US, AU/NZ) and adaptive learning</td>
  </tr>
  <tr>
    <td><strong>Smart Agent System</strong></td>
    <td>Voice commands for browser, terminal, volume control, and screenshot functionality</td>
  </tr>
  <tr>
    <td><strong>Adaptive Learning</strong></td>
    <td>Analyzes failed commands and suggests appropriate categories based on word overlap</td>
  </tr>
  <tr>
    <td><strong>User Profiles</strong></td>
    <td>Individual voice profiles with learning data and preferences</td>
  </tr>
  <tr>
    <td><strong>Cross-Platform Support</strong></td>
    <td>Linux (fully tested), Windows/macOS (theoretical support)</td>
  </tr>
</table>

## Installation

### Prerequisites

- **Python**: 3.8+
- **Operating System**: Linux (Ubuntu 18.04+ recommended), Windows 10+, macOS 10.14+
- **Memory**: 4GB RAM minimum
- **Audio**: Microphone input

### Linux Installation (Primary Platform)

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

# Install dependencies
pip install -r requirements.txt

# Test basic functionality
python3 -m vocalflow.main --help
```

### Windows/macOS Installation

Basic installation should work following similar steps, but these platforms have limited testing. Some system-specific commands may not function correctly.

## Usage

### Basic Voice Dictation

```bash
# Start basic dictation
python3 -m vocalflow.main

# Use specific profile
python3 -m vocalflow.main --profile myprofile

# Use clipboard output
python3 -m vocalflow.main --output clipboard
```

### Enhanced Agent System

```bash
# Start with enhanced linguistic processing
python3 enhanced_agents.py

# Test the integration system
python3 tests/test_linguistic_only.py

# Run comprehensive demo
python3 vocalflow_enhanced_fixed.py
```

## Enhanced Linguistic System

### Voice Commands

The system recognizes various command patterns with high confidence:

**Browser Control:**
- "open browser please" / "fire up browser" / "launch browser"
- "start browser up" / "bring up browser"

**System Control:**
- "take screenshot" / "capture screen" / "snap screen"
- "volume up" / "volume down" / "increase volume"
- "open terminal" / "launch terminal" / "fire up terminal"

### Adaptive Learning

When commands fail both enhanced and fallback matching, the system analyzes them for learning opportunities:

- Suggests appropriate categories based on word overlap
- Uses adaptive thresholds based on command length
- Filters suggestions to valid, executable categories only
- Provides confidence scores for learning suggestions

### Regional Variations

The YAML database includes:
- **607 total command variations**
- **UK, US, Australian/New Zealand dialects**
- **Casual and formal speech patterns**
- **Polite prefixes and natural language fillers**

## Command Recognition Architecture

### Three-Layer Processing

1. **Enhanced Matching**: Uses YAML variations database (confidence: 1.00)
2. **Fallback Patterns**: Regex-based matching for core commands (confidence: 0.80)
3. **Adaptive Learning**: Suggests categories for unrecognized commands with overlapping keywords

### Integration Components

- `linguistic_bridge.py` - Core integration layer
- `enhanced_agents.py` - Agent wrapper with linguistic processing
- `enhanced_command_matcher.py` - YAML-based pattern matching
- `command_variations.yaml` - 607-variation command database

## Profiles and Learning

### User Profiles

VocalFlow creates profiles at `~/.vocalflow/profiles/`:

```json
{
  "audio_settings": { "volume_threshold": 0.02, "silence_duration": 1.5 },
  "vocabulary": { "learned_words": [], "corrections": {} },
  "command_history": { "successful_commands": [], "failed_attempts": [] }
}
```

### Learning Process

1. **Audio Calibration**: Adjusts thresholds based on user's voice
2. **Vocabulary Learning**: Tracks commonly used words and phrases
3. **Command Analysis**: Records successful and failed voice commands
4. **Pattern Adaptation**: Learns user preferences and speech patterns

## Current Status and Limitations

### Working Features
- Core dictation with Whisper integration
- Enhanced linguistic command recognition (607 variations)
- Adaptive learning system with category suggestions
- User profile management
- Cross-platform compatibility layer

### Known Limitations
- Linux is the primary tested platform
- YAML structure has some corruption (handled by category filtering)
- Windows/macOS support is theoretical and untested
- Some voice commands may require multiple attempts
- Screenshot functionality requires gnome-screenshot on Linux

### Recent Completions
- Enhanced linguistic integration bridge
- Adaptive learning with mathematical validation
- Category filtering system for valid suggestions
- Comprehensive test suite integration

## API Usage

### Basic Integration

```python
from enhanced_agents import EnhancedVocalFlowAgent

# Create enhanced agent
agent = EnhancedVocalFlowAgent()

# Process voice command
result = agent.process_text_enhanced("open browser please")
print(f"Command executed: {result}")

# Check system status
status = agent.get_status()
print(f"Enhanced mode: {status['enhanced_mode']}")
```

### Linguistic Bridge

```python
from linguistic_bridge import LinguisticBridge

# Create bridge with adaptive learning
bridge = LinguisticBridge()

# Process command with learning
is_command, command_type, metadata = bridge.process_command("fire up terminal")
print(f"Recognized: {is_command}, Type: {command_type}")
print(f"Method: {metadata.get('method')}, Confidence: {metadata.get('confidence')}")
```

## Development Status

### Recently Completed (v3.0)
- Enhanced linguistic integration with 607 command variations
- Adaptive learning system with category filtering
- Integration bridge connecting all components
- Comprehensive testing framework

### Current Phase
- Phase 1A: Enhanced linguistic integration (Complete)
- Next: Phase 1B targeting mathematical validation integration

### Architecture
- Modular design with loose coupling
- Fallback systems for graceful degradation
- Comprehensive error handling and logging
- Production-ready core components

## Troubleshooting

### Common Issues

**Audio Device Not Found:**
```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

**Command Recognition Issues:**
```bash
# Test linguistic system
python3 tests/test_linguistic_only.py

# Check enhanced agent status
python3 -c "from enhanced_agents import EnhancedVocalFlowAgent; print(EnhancedVocalFlowAgent().get_status())"
```

**Linux Dependencies:**
```bash
# Install missing system packages
sudo apt install xdotool xclip gnome-screenshot portaudio19-dev
```

## Contributing

This project welcomes contributions, particularly for:
- Windows/macOS platform testing and fixes
- Additional command variations and regional dialects
- Performance optimizations
- Documentation improvements

### Development Setup

```bash
git clone https://github.com/R3DK3LL/VocalFlow.git
cd VocalFlow
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 tests/test_linguistic_only.py
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for speech recognition
- [sounddevice](https://github.com/spatialaudio/python-sounddevice) for audio capture
- Enhanced linguistic processing inspired by computational linguistics research

---

<div align="center">
  <p><strong>A practical voice dictation system with intelligent command recognition</strong></p>
  <p>
    <a href="https://github.com/R3DK3LL/VocalFlow">GitHub</a> •
    <a href="https://github.com/R3DK3LL/VocalFlow/issues">Issues</a>
  </p>
</div>
