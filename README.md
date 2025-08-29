# VocalFlow

<div align="center">
  <h3>Universal Voice Dictation System with Intelligent Agents</h3>
  <p>Cross-platform voice-to-text that learns and adapts to individual speech patterns with smart command execution</p>
  
  <p>
    <img src="https://img.shields.io/badge/status-beta-yellow?style=for-the-badge" alt="Beta Status">
    <img src="https://img.shields.io/badge/development-active-brightgreen?style=for-the-badge" alt="Active Development">
    <a href="#installation"><img src="https://img.shields.io/badge/Install-Quick%20Start-blue?style=for-the-badge" alt="Installation"></a>
    <a href="#usage"><img src="https://img.shields.io/badge/Usage-Documentation-green?style=for-the-badge" alt="Usage"></a>
  </p>
  
  <p>
    <a href="#agents"><img src="https://img.shields.io/badge/Agents-Voice%20Commands-orange?style=for-the-badge" alt="Agents"></a>
    <a href="#contributing"><img src="https://img.shields.io/badge/Contribute-Guidelines-red?style=for-the-badge" alt="Contributing"></a>
  </p>
  
  <p>
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" alt="Platform Support">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
    <img src="https://img.shields.io/badge/agents-voice%20commands-purple.svg" alt="Agents">
  </p>
  
  <p>
    <strong>⚠️ DEVELOPMENT STATUS</strong><br>
    This project is currently in active development and beta testing phase.<br>
    Features may be unstable and breaking changes can occur between versions.<br>
    Bug reports and feedback are highly appreciated.
  </p>
</div>

## Overview

VocalFlow is an advanced voice dictation system that transcribes speech to text in real-time while continuously learning from individual users. The system features an intelligent agent layer that can execute voice commands, detect context, and provide smart auto-completion. Unlike traditional speech-to-text solutions, VocalFlow creates personalized voice profiles that improve accuracy and adapt to unique speech patterns, vocabulary, and environmental conditions.

### Key Features

<table>
  <tr>
    <td><strong>Universal Compatibility</strong></td>
    <td>Native support for Windows, Linux, and macOS with automatic platform detection</td>
  </tr>
  <tr>
    <td><strong>Intelligent Agents</strong></td>
    <td>Voice command execution, context detection, and smart auto-completion with application discovery</td>
  </tr>
  <tr>
    <td><strong>Adaptive Learning Engine</strong></td>
    <td>Continuously learns speech patterns, vocabulary, environmental audio, and user preferences</td>
  </tr>
  <tr>
    <td><strong>Multi-User Profiles</strong></td>
    <td>Individual voice profiles with isolated learning data, preferences, and command histories</td>
  </tr>
  <tr>
    <td><strong>Real-Time Processing</strong></td>
    <td>Low-latency transcription with optimized voice activity detection and command recognition</td>
  </tr>
  <tr>
    <td><strong>Smart Application Discovery</strong></td>
    <td>Automatically detects installed applications and provides fallback options for voice commands</td>
  </tr>
  <tr>
    <td><strong>Context-Aware Operation</strong></td>
    <td>Detects coding, writing, email contexts and adapts vocabulary and suggestions accordingly</td>
  </tr>
  <tr>
    <td><strong>Multiple Output Methods</strong></td>
    <td>Direct typing, clipboard integration, or custom output handlers with cross-platform support</td>
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
sudo apt install -y python3-pip python3-venv portaudio19-dev xdotool xclip gnome-screenshot

# Clone repository
git clone https://github.com/R3DK3LL/VocalFLow.git
cd VocalFLow

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -m vocalflow.main --help
python agents.py --help
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
git clone https://github.com/R3DK3LL/VocalFLow.git
cd VocalFLow

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -m vocalflow.main --help
python agents.py --help
```

</details>

<details>
<summary><strong>Windows Installation</strong></summary>

```powershell
# Install Python 3.8+ from python.org if not installed

# Clone repository
git clone https://github.com/R3DK3LL/VocalFLow.git
cd VocalFLow

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install Windows-specific packages
pip install pyautogui

# Verify installation
python -m vocalflow.main --help
python agents.py --help
```

</details>

## Usage

### Basic Voice Dictation

```bash
# Start basic dictation with default profile
python -m vocalflow.main

# Start with named profile
python -m vocalflow.main --profile john

# Use clipboard output (cross-platform)
python -m vocalflow.main --output clipboard

# List available profiles
python -m vocalflow.main --list-profiles
```

### Intelligent Agents System

```bash
# Start with intelligent agents enabled
python agents.py --profile john --enable-commands --enable-context --enable-autocomplete

# Enable all agent features
python agents.py --profile work --enable-commands --enable-context --enable-autocomplete --enable-workflow --enable-proactive

# Test agents in standalone mode (without voice input)
python agents.py --standalone --enable-commands --enable-context
```

### Command Line Options

#### Core VocalFlow Options

| Option | Description | Default |
|--------|-------------|---------|
| `--profile`, `-p` | Voice profile name | `default` |
| `--output`, `-o` | Output method (`auto`, `clipboard`, `xdotool`, `wintype`) | `auto` |
| `--model`, `-m` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) | `base` |
| `--language`, `-l` | Recognition language code | `en` |
| `--list-profiles` | Show all available profiles | - |
| `--verbose`, `-v` | Enable verbose logging | - |

#### Agent System Options

| Option | Description | Default |
|--------|-------------|---------|
| `--enable-commands` | Enable voice command execution | `False` |
| `--enable-context` | Enable context-aware detection | `False` |
| `--enable-autocomplete` | Enable smart auto-completion | `False` |
| `--enable-workflow` | Enable workflow pattern learning | `False` |
| `--enable-proactive` | Enable proactive suggestions | `False` |
| `--standalone` | Run agents without voice input (testing) | `False` |

## Intelligent Agents

### Voice Commands

The agent system can execute voice commands by saying "execute [command]":

#### Application Control
- `execute open browser` - Opens available browser (Firefox, Chrome, Chromium)
- `execute open terminal` - Opens system terminal
- `execute open code` - Opens code editor (VS Code, Codium, etc.)
- `execute open files` - Opens file manager

#### System Control
- `execute volume up/down/mute` - Controls system volume
- `execute take screenshot` - Captures screenshot
- `execute brightness up/down` - Controls screen brightness

#### Navigation & Text Editing
- `execute scroll up/down` - Scrolls current window
- `execute select all` - Selects all text
- `execute copy this/paste here` - Clipboard operations
- `execute undo that` - Undo last action

#### Workflow Commands
- `execute start coding session` - Opens code editor, browser, and terminal
- `execute end session` - Saves work and closes applications

### Context Detection

The system automatically detects and adapts to different contexts:

- **Coding Context**: Detects programming keywords and adapts vocabulary
- **Email Context**: Recognizes email patterns and suggests professional phrases  
- **Documentation Context**: Identifies documentation writing and formatting
- **Creative Writing**: Adapts to narrative and creative content

### Auto-Completion

Smart phrase completion based on learned patterns:

- `good morning` → "Good morning, I hope you're doing well."
- `thank you for` → "Thank you for your time and consideration."
- `def main` → "def main():\n    pass"
- Custom completions learned from your usage patterns

### Application Discovery

The agent system intelligently discovers installed applications:

```python
# Browser detection priority
['firefox', 'google-chrome', 'chromium', 'opera', 'brave-browser']

# Terminal detection priority  
['gnome-terminal', 'konsole', 'xterm', 'alacritty', 'terminator']

# Editor detection priority
['code', 'codium', 'atom', 'subl', 'gedit', 'kate']
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
    <td><strong>Command History</strong></td>
    <td>Voice command usage tracking and optimization</td>
    <td>Improve command recognition and suggest shortcuts</td>
  </tr>
  <tr>
    <td><strong>Correction Patterns</strong></td>
    <td>User correction tracking and pattern recognition</td>
    <td>Automatically apply learned fixes to common transcription errors</td>
  </tr>
  <tr>
    <td><strong>Context Preferences</strong></td>
    <td>Analysis of work patterns and context switching</td>
    <td>Proactive context detection and vocabulary adaptation</td>
  </tr>
  <tr>
    <td><strong>Workflow Patterns</strong></td>
    <td>Time-based activity analysis and routine detection</td>
    <td>Suggest workflow improvements and automate repetitive tasks</td>
  </tr>
</table>

### Learning Process

1. **Initial Calibration**: System analyzes first 30 seconds of audio to establish baseline parameters
2. **Continuous Adaptation**: Real-time adjustment of thresholds and parameters during use
3. **Session Learning**: Collection and analysis of vocabulary, corrections, commands, and speech patterns
4. **Agent Training**: Command success rates, application preferences, and context switches
5. **Profile Updates**: Automatic saving of learned parameters and agent data at session end
6. **Cross-Session Memory**: Persistent storage and retrieval of user-specific optimizations

### Configuration Structure

VocalFlow automatically creates configuration directories:

```
~/.vocalflow/
├── profiles/              # Voice profiles with learning data
│   ├── default.json      # Default user profile
│   ├── john.json         # Personal profile for John
│   └── work.json         # Work environment profile
├── logs/                 # System and agent logs
│   ├── vocalflow.log     # Core system logs
│   └── agents.log        # Agent activity logs
└── models/               # Cached Whisper models
    └── base/             # Whisper base model files
```

### Logging and Analytics

VocalFlow maintains detailed logs for performance monitoring and debugging:

- **Audio Quality Metrics**: Signal-to-noise ratios, volume levels, detection accuracy
- **Transcription Performance**: Processing times, confidence scores, error rates
- **Agent Activity**: Command execution success rates, application discovery results
- **Learning Progress**: Profile evolution, adaptation effectiveness, vocabulary growth
- **System Performance**: Resource usage, processing latency, error tracking

## API Reference

### Core System

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

### Agent Integration

```python
from agents import VocalFlowAgentSystem, AgentConfig

# Configure agents
config = AgentConfig(
    enable_commands=True,
    enable_context=True,
    enable_autocomplete=True
)

# Initialize with agents
system = VocalFlowAgentSystem("profile_name", config)
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
| `tiny` | Fastest | Good | 39 MB | Resource-constrained systems, fast commands |
| `base` | Fast | Better | 74 MB | **Default - balanced performance and accuracy** |
| `small` | Medium | Better | 244 MB | Higher accuracy needs, professional use |
| `medium` | Slower | Best | 769 MB | Professional transcription, detailed work |
| `large` | Slowest | Excellent | 1550 MB | Maximum accuracy, production environments |

### Hardware Recommendations

- **CPU**: Multi-core processor (Intel i5+ or AMD Ryzen 5+ recommended)
- **RAM**: 8GB+ for optimal performance with larger models and agent processing
- **Audio**: USB or XLR microphone for best recognition accuracy and command detection
- **Storage**: SSD recommended for faster model loading and profile access

### Agent Performance

- **Command Detection**: Real-time processing with <50ms latency
- **Application Discovery**: Cached results for faster subsequent launches
- **Context Switching**: Optimized pattern matching with minimal CPU overhead
- **Learning Updates**: Background processing to avoid interrupting dictation

## Troubleshooting

<details>
<summary><strong>Known Issues (Beta)</strong></summary>

**Current Known Issues:**
- Some voice commands may require multiple attempts on certain Linux distributions
- Ctrl+C shutdown occasionally requires force termination (use `killall python`)
- Screenshot functionality requires `gnome-screenshot` on Linux systems
- Audio device selection may need manual configuration on some systems
- Context detection is still learning and may misidentify content types

**Under Active Development:**
- Improved shutdown handling for cleaner exits
- Enhanced cross-platform application discovery
- Better error recovery for failed voice commands
- Optimized learning algorithms for faster adaptation

</details>

<details>
<summary><strong>Common Issues</strong></summary>

### Audio Device Not Found
```bash
# List available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Specify device ID
python -m vocalflow.main --audio-device 1
python agents.py --profile user --audio-device 1
```

### Voice Commands Not Working
```bash
# Check if applications are installed
which firefox
which gnome-terminal
which code

# Test agent system standalone
python agents.py --standalone --enable-commands
```

### Permission Errors (Linux)
```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# Install required system packages
sudo apt install xdotool xclip gnome-screenshot
```

### Model Download Issues
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/
rm -rf ~/.vocalflow/models/

# Retry with verbose logging
python -m vocalflow.main --verbose
python agents.py --profile user --verbose
```

### Agent Application Discovery Issues
```bash
# Check what applications are detected
python -c "
import subprocess
apps = ['firefox', 'code', 'gnome-terminal']
for app in apps:
    result = subprocess.run(['which', app], capture_output=True)
    print(f'{app}: {\"Found\" if result.returncode == 0 else \"Not found\"}')"
```

</details>

## Contributing

We welcome contributions from the community. Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Development Setup

```bash
# Clone development branch
git clone -b develop https://github.com/R3DK3LL/VocalFLow.git
cd VocalFLow

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 vocalflow/ agents.py
black vocalflow/ agents.py
```

### Issue Reporting

When reporting issues, please include:
- Operating system and version
- Python version
- Complete error messages
- Steps to reproduce
- Audio device information
- Agent features enabled
- Output of `python agents.py --help`

## Roadmap

### Core System
- [ ] GUI interface for non-technical users
- [ ] Custom vocabulary import/export
- [ ] Multiple language support within single profiles
- [ ] Cloud profile synchronization
- [ ] Integration with popular text editors
- [ ] Batch audio file transcription

### Agent System
- [ ] Custom voice command creation interface
- [ ] Workflow automation builder
- [ ] Integration with productivity apps (Slack, Teams, etc.)
- [ ] Smart meeting transcription with action item detection
- [ ] Code comment generation from voice descriptions
- [ ] Email composition with template suggestions

### Advanced Features
- [ ] Multi-speaker recognition and separation
- [ ] Real-time collaboration features
- [ ] Plugin system for custom agents
- [ ] Advanced context learning with AI models
- [ ] Voice-controlled IDE integration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [faster-whisper](https://github.com/guillaumekln/faster-whisper) for high-performance speech recognition
- Uses [sounddevice](https://github.com/spatialaudio/python-sounddevice) for cross-platform audio capture
- Agent system inspired by modern voice assistants and productivity automation tools
- Intelligent application discovery adapted from various Linux desktop environment methods

---

<div align="center">
  <p><strong>Made for developers, writers, and voice-first productivity enthusiasts</strong></p>
  <p>
    <a href="https://github.com/R3DK3LL/VocalFLow">GitHub</a> •
    <a href="https://github.com/R3DK3LL/VocalFLow/issues">Issues</a> •
    <a href="https://github.com/R3DK3LL/VocalFLow/discussions">Discussions</a> •
    <a href="https://github.com/R3DK3LL/VocalFLow/wiki">Wiki</a>
  </p>
</div>
