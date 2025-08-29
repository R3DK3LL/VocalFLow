#!/usr/bin/env python3
"""
VocalFlow Agents - Intelligent Agent System for Voice Dictation
Standalone script that adds agentic capabilities to VocalFlow

Usage:
    python agents.py --profile john
    python agents.py --profile work --enable-commands --enable-context
"""

import re
import time
import json
import subprocess
import threading
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, asdict
import logging

# Import VocalFlow if available
try:
    from vocalflow.main import VocalFlowSystem, VoiceProfile
    VOCALFLOW_AVAILABLE = True
except ImportError:
    VOCALFLOW_AVAILABLE = False
    print("Warning: VocalFlow main system not found. Running in standalone mode.")

@dataclass
class AgentConfig:
    """Configuration for agent system"""
    enable_commands: bool = True
    enable_context: bool = True
    enable_autocomplete: bool = True
    enable_workflow: bool = True
    enable_proactive: bool = False
    command_trigger: str = "execute"  # Say "execute open browser" to run commands
    learning_rate: float = 0.1
    context_window: int = 50  # words to consider for context

class VoiceCommandAgent:
    """Processes voice commands and executes system actions"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger('vocalflow.agents.commands')
        
        # Command patterns and their actions
        self.commands = {
            # Application control
            r"open (\w+)": self._open_application,
            r"close (\w+)": self._close_application,
            r"switch to (\w+)": self._switch_application,
            
            # File operations
            r"new (file|document)": self._new_file,
            r"save (file|this|document)": self._save_file,
            r"open file (.+)": self._open_file,
            
            # System control
            r"volume (up|down|mute)": self._control_volume,
            r"brightness (up|down)": self._control_brightness,
            
            # Navigation
            r"scroll (up|down)": self._scroll_page,
            r"go to top": lambda: self._key_combo("ctrl+Home"),
            r"go to bottom": lambda: self._key_combo("ctrl+End"),
            
            # Text editing
            r"select all": lambda: self._key_combo("ctrl+a"),
            r"copy this": lambda: self._key_combo("ctrl+c"),
            r"paste here": lambda: self._key_combo("ctrl+v"),
            r"undo that": lambda: self._key_combo("ctrl+z"),
            r"redo that": lambda: self._key_combo("ctrl+y"),
            
            # Custom workflows
            r"start coding session": self._start_coding_session,
            r"end session": self._end_session,
            r"take screenshot": self._take_screenshot,
        }
        
        self.command_history = deque(maxlen=100)
        
    def process_text(self, text: str) -> bool:
        """Process text for commands. Returns True if command executed."""
        if not self.config.enable_commands:
            return False
            
        # Check if text contains command trigger
        if self.config.command_trigger.lower() not in text.lower():
            return False
            
        # Remove trigger word and process
        clean_text = text.lower().replace(self.config.command_trigger.lower(), "").strip()
        
        for pattern, action in self.commands.items():
            match = re.search(pattern, clean_text)
            if match:
                try:
                    # Call action with match parameter
                    action(match)
                    
                    self.command_history.append({
                        'command': clean_text,
                        'timestamp': datetime.now().isoformat(),
                        'pattern': pattern
                    })
                    
                    self.logger.info(f"Executed command: {clean_text}")
                    print(f"Executed: {clean_text}")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Command execution failed: {e}")
                    print(f"❌ Command failed: {clean_text}")
                    
        return False
    
    def _open_application(self, match):
        """Open application by name with intelligent discovery"""
        app_name = match.group(1).lower()
        
        # Try to find the best executable for the requested app
        executable = self._discover_application(app_name)
        
        if executable:
            try:
                subprocess.Popen([executable])
                self.logger.info(f"Successfully opened {executable}")
            except Exception as e:
                self.logger.error(f"Failed to launch {executable}: {e}")
                # Try fallback
                self._try_fallback_launch(app_name)
        else:
            self.logger.warning(f"No suitable application found for: {app_name}")
            print(f"Application '{app_name}' not found on system")
    
    def _discover_application(self, app_type: str) -> Optional[str]:
        """Intelligently discover application executable"""
        
        # Application search mappings with priority order
        app_searches = {
            'browser': ['firefox', 'google-chrome', 'chromium-browser', 'chromium', 'opera', 'brave-browser'],
            'firefox': ['firefox'],
            'chrome': ['google-chrome', 'chromium-browser', 'chromium'],
            'code': ['code', 'codium', 'atom', 'subl'],
            'vscode': ['code', 'codium'],
            'terminal': ['gnome-terminal', 'konsole', 'xterm', 'alacritty', 'terminator', 'tilix'],
            'files': ['nautilus', 'dolphin', 'thunar', 'pcmanfm', 'nemo'],
            'calculator': ['gnome-calculator', 'kcalc', 'galculator'],
            'text': ['gedit', 'kate', 'mousepad', 'leafpad', 'nano'],
            'editor': ['gedit', 'kate', 'mousepad', 'vim', 'nano'],
            'notepad': ['gedit', 'kate', 'mousepad', 'leafpad']
        }
        
        # Check if app_type has specific search list
        candidates = app_searches.get(app_type, [app_type])
        
        # Try each candidate in priority order
        for candidate in candidates:
            if self._check_executable_exists(candidate):
                return candidate
        
        # If no specific matches, try the app name directly
        if self._check_executable_exists(app_type):
            return app_type
            
        return None
    
    def _check_executable_exists(self, executable: str) -> bool:
        """Check if executable exists in PATH"""
        try:
            result = subprocess.run(['which', executable], 
                                  capture_output=True, 
                                  text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def _try_fallback_launch(self, app_name: str):
        """Try fallback launch methods"""
        fallback_methods = [
            # Try with xdg-open (opens default application)
            lambda: subprocess.Popen(['xdg-open', '.']),
            # Try to find in common installation paths
            lambda: self._check_common_paths(app_name),
        ]
        
        for method in fallback_methods:
            try:
                method()
                self.logger.info(f"Fallback launch successful for {app_name}")
                return
            except Exception:
                continue
        
        print(f"Unable to launch {app_name} - try installing it first")
    
    def _check_common_paths(self, app_name: str):
        """Check common installation paths"""
        common_paths = [
            f'/usr/bin/{app_name}',
            f'/usr/local/bin/{app_name}',
            f'/snap/bin/{app_name}',
            f'/opt/{app_name}/{app_name}'
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                subprocess.Popen([path])
                return
        
        raise FileNotFoundError(f"Application {app_name} not found in common paths")
    
    def _close_application(self, match):
        """Close application by name"""
        app_name = match.group(1)
        subprocess.run(["pkill", "-f", app_name])
    
    def _switch_application(self, match):
        """Switch to application window"""
        app_name = match.group(1)
        subprocess.run(["wmctrl", "-a", app_name])
    
    def _new_file(self, match):
        """Create new file"""
        file_type = match.group(1)
        if file_type == "file":
            subprocess.Popen(["code", "untitled.txt"])
        else:
            subprocess.Popen(["code"])
    
    def _save_file(self, match):
        """Save current file"""
        self._key_combo("ctrl+s")
    
    def _open_file(self, match):
        """Open specific file"""
        filename = match.group(1)
        subprocess.Popen(["code", filename])
    
    def _control_volume(self, match):
        """Control system volume"""
        action = match.group(1)
        volume_commands = {
            'up': ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+5%"],
            'down': ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-5%"],
            'mute': ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"]
        }
        subprocess.run(volume_commands[action])
    
    def _control_brightness(self, match):
        """Control screen brightness"""
        action = match.group(1)
        if action == "up":
            subprocess.run(["xbacklight", "-inc", "10"])
        else:
            subprocess.run(["xbacklight", "-dec", "10"])
    
    def _scroll_page(self, match):
        """Scroll page up or down"""
        direction = match.group(1)
        if direction == "up":
            self._key_combo("Page_Up")
        else:
            self._key_combo("Page_Down")
    
    def _key_combo(self, keys: str):
        """Send keyboard combination"""
        subprocess.run(["xdotool", "key", keys])
    
    def _start_coding_session(self):
        """Start a coding session workflow"""
        subprocess.Popen(["code"])
        time.sleep(2)
        subprocess.Popen(["firefox", "https://github.com"])
        subprocess.Popen(["gnome-terminal"])
    
    def _end_session(self):
        """End work session"""
        # Auto-save all open files
        subprocess.run(["xdotool", "key", "ctrl+s"])
        print("🎯 Session ended, files saved")
    
    def _take_screenshot(self):
        """Take a screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subprocess.run(["gnome-screenshot", "-f", f"screenshot_{timestamp}.png"])

class ContextAwareAgent:
    """Detects and adapts to different contexts (coding, writing, email, etc.)"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger('vocalflow.agents.context')
        
        # Context keywords and patterns
        self.context_keywords = {
            'coding': {
                'keywords': ['function', 'class', 'import', 'def', 'variable', 'loop', 'array', 'string', 'integer', 'boolean', 'method', 'object', 'algorithm', 'debug', 'compile', 'execute'],
                'patterns': [r'def \w+\(', r'class \w+:', r'import \w+', r'from \w+ import'],
                'vocabulary': ['Python', 'JavaScript', 'HTML', 'CSS', 'API', 'JSON', 'SQL']
            },
            'email': {
                'keywords': ['dear', 'regards', 'sincerely', 'meeting', 'schedule', 'appointment', 'follow up', 'attached', 'please', 'thank you'],
                'patterns': [r'dear \w+', r'best regards', r'sincerely'],
                'vocabulary': ['Dear', 'Sincerely', 'Best regards', 'Looking forward']
            },
            'documentation': {
                'keywords': ['example', 'usage', 'installation', 'requirements', 'tutorial', 'guide', 'reference', 'documentation'],
                'patterns': [r'## \w+', r'### \w+', r'```\w+'],
                'vocabulary': ['Installation', 'Usage', 'Example', 'Documentation']
            },
            'creative_writing': {
                'keywords': ['character', 'story', 'plot', 'scene', 'dialogue', 'narrative', 'chapter', 'protagonist'],
                'patterns': [r'chapter \d+', r'"[^"]*"'],
                'vocabulary': ['character', 'narrative', 'dialogue', 'scene']
            }
        }
        
        self.current_context = 'general'
        self.context_history = deque(maxlen=10)
        self.word_buffer = deque(maxlen=config.context_window)
        
    def update_context(self, text: str) -> str:
        """Update context based on recent text"""
        if not self.config.enable_context:
            return self.current_context
            
        # Add words to buffer
        words = text.lower().split()
        self.word_buffer.extend(words)
        
        # Calculate context scores
        context_scores = {}
        recent_text = ' '.join(list(self.word_buffer))
        
        for context_name, context_data in self.context_keywords.items():
            score = 0
            
            # Keyword matching
            for keyword in context_data['keywords']:
                score += recent_text.count(keyword.lower()) * 1
            
            # Pattern matching
            for pattern in context_data['patterns']:
                score += len(re.findall(pattern, recent_text)) * 2
            
            context_scores[context_name] = score
        
        # Determine new context
        if context_scores:
            best_context = max(context_scores, key=context_scores.get)
            if context_scores[best_context] > 2:  # Minimum confidence threshold
                if best_context != self.current_context:
                    self.logger.info(f"Context changed: {self.current_context} -> {best_context}")
                    print(f"🎯 Context: {best_context}")
                    self.current_context = best_context
                    self.context_history.append({
                        'context': best_context,
                        'timestamp': datetime.now().isoformat(),
                        'confidence': context_scores[best_context]
                    })
        
        return self.current_context
    
    def get_context_vocabulary(self) -> List[str]:
        """Get vocabulary suggestions for current context"""
        if self.current_context in self.context_keywords:
            return self.context_keywords[self.current_context]['vocabulary']
        return []
    
    def suggest_context_action(self) -> Optional[str]:
        """Suggest context-appropriate actions"""
        suggestions = {
            'coding': "💡 Tip: Say 'execute open terminal' to open terminal",
            'email': "📧 Tip: Say 'execute new document' for email template",
            'documentation': "📝 Tip: Use markdown formatting for headers",
            'creative_writing': "✍️ Tip: Focus on dialogue and scene description"
        }
        return suggestions.get(self.current_context)

class AutoCompleteAgent:
    """Provides intelligent auto-completion and text suggestions"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger('vocalflow.agents.autocomplete')
        
        # Common phrase completions
        self.phrase_completions = {
            'good morning': 'Good morning, I hope you\'re doing well.',
            'thank you for': 'Thank you for your time and consideration.',
            'looking forward': 'I\'m looking forward to hearing from you.',
            'please let me know': 'Please let me know if you have any questions.',
            'best regards': 'Best regards,',
            'following up': 'Following up on our previous conversation,',
            'def main': 'def main():\n    pass',
            'if name': 'if __name__ == "__main__":',
            'import': 'import numpy as np',
        }
        
        # Dynamic completions learned from user
        self.learned_completions = {}
        self.completion_usage = Counter()
        
    def suggest_completion(self, partial_text: str) -> Optional[str]:
        """Suggest completion for partial text"""
        if not self.config.enable_autocomplete:
            return None
            
        text_lower = partial_text.lower().strip()
        
        # Check static completions
        for trigger, completion in self.phrase_completions.items():
            if text_lower.endswith(trigger):
                self.completion_usage[trigger] += 1
                return completion
        
        # Check learned completions
        for trigger, completion in self.learned_completions.items():
            if text_lower.endswith(trigger):
                self.completion_usage[trigger] += 1
                return completion
                
        return None
    
    def learn_completion(self, trigger: str, completion: str):
        """Learn a new completion pattern"""
        self.learned_completions[trigger.lower()] = completion
        self.logger.info(f"Learned completion: '{trigger}' -> '{completion[:30]}...'")

class WorkflowAgent:
    """Learns and suggests workflow patterns based on time and usage"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger('vocalflow.agents.workflow')
        
        # Time-based patterns
        self.hourly_patterns = defaultdict(list)
        self.daily_patterns = defaultdict(list)
        
        # Activity tracking
        self.session_start = datetime.now()
        self.activity_log = []
        
    def log_activity(self, activity: str, context: str = "general"):
        """Log user activity for pattern learning"""
        if not self.config.enable_workflow:
            return
            
        timestamp = datetime.now()
        self.activity_log.append({
            'activity': activity,
            'context': context,
            'timestamp': timestamp.isoformat(),
            'hour': timestamp.hour,
            'weekday': timestamp.weekday()
        })
        
        # Learn hourly patterns
        self.hourly_patterns[timestamp.hour].append(activity)
        self.daily_patterns[timestamp.weekday()].append(activity)
    
    def suggest_workflow(self) -> Optional[str]:
        """Suggest workflow based on current time and patterns"""
        if not self.config.enable_workflow:
            return None
            
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Find most common activity for this hour
        if current_hour in self.hourly_patterns:
            activities = self.hourly_patterns[current_hour]
            if activities:
                most_common = Counter(activities).most_common(1)[0][0]
                return f"💡 Usually at this time you: {most_common}"
        
        return None
    
    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        duration = datetime.now() - self.session_start
        return {
            'duration_minutes': duration.total_seconds() / 60,
            'activities': len(self.activity_log),
            'contexts': list(set(a['context'] for a in self.activity_log))
        }

class ProactiveAgent:
    """Proactively suggests improvements and automations"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger('vocalflow.agents.proactive')
        
        # Pattern detection thresholds
        self.repetition_threshold = 3
        self.suggestion_cooldown = 300  # 5 minutes
        self.last_suggestions = {}
        
        # Improvement tracking
        self.correction_patterns = Counter()
        self.frequent_phrases = Counter()
        
    def analyze_patterns(self, text: str, corrections: List[Tuple[str, str]]):
        """Analyze patterns and make proactive suggestions"""
        if not self.config.enable_proactive:
            return []
            
        suggestions = []
        
        # Track corrections
        for original, corrected in corrections:
            self.correction_patterns[f"{original}->{corrected}"] += 1
        
        # Track frequent phrases
        for phrase in self._extract_phrases(text):
            self.frequent_phrases[phrase] += 1
        
        # Generate suggestions
        suggestions.extend(self._suggest_automation())
        suggestions.extend(self._suggest_vocabulary_improvements())
        suggestions.extend(self._suggest_shortcuts())
        
        return suggestions
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract meaningful phrases from text"""
        # Simple phrase extraction (can be improved)
        words = text.split()
        phrases = []
        
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrases.append(phrase.lower())
            
        return phrases
    
    def _suggest_automation(self) -> List[str]:
        """Suggest automation opportunities"""
        suggestions = []
        
        # Find repetitive correction patterns
        for pattern, count in self.correction_patterns.most_common(5):
            if count >= self.repetition_threshold:
                if self._can_suggest('correction', pattern):
                    suggestions.append(
                        f"🔧 I notice you often correct '{pattern.split('->')[0]}' to "
                        f"'{pattern.split('->')[1]}'. Should I automate this?"
                    )
        
        return suggestions
    
    def _suggest_vocabulary_improvements(self) -> List[str]:
        """Suggest vocabulary improvements"""
        suggestions = []
        
        # Find frequent phrases that could be shortcuts
        for phrase, count in self.frequent_phrases.most_common(5):
            if count >= self.repetition_threshold and len(phrase.split()) >= 3:
                if self._can_suggest('vocabulary', phrase):
                    suggestions.append(
                        f"💬 You often say '{phrase}'. Want to create a shortcut?"
                    )
        
        return suggestions
    
    def _suggest_shortcuts(self) -> List[str]:
        """Suggest useful shortcuts"""
        suggestions = []
        
        # Context-based suggestions
        if self._can_suggest('general', 'shortcuts'):
            suggestions.append("⌨️ Tip: Say 'execute' before commands like 'open browser'")
        
        return suggestions
    
    def _can_suggest(self, category: str, item: str) -> bool:
        """Check if we can make a suggestion (cooldown logic)"""
        key = f"{category}:{item}"
        now = time.time()
        
        if key in self.last_suggestions:
            if now - self.last_suggestions[key] < self.suggestion_cooldown:
                return False
        
        self.last_suggestions[key] = now
        return True

class AgentSystem:
    """Main agent system that coordinates all agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger('vocalflow.agents.system')
        
        # Initialize agents
        self.command_agent = VoiceCommandAgent(config)
        self.context_agent = ContextAwareAgent(config)
        self.autocomplete_agent = AutoCompleteAgent(config)
        self.workflow_agent = WorkflowAgent(config)
        self.proactive_agent = ProactiveAgent(config)
        
        # Agent communication
        self.agent_messages = deque(maxlen=50)
        
    def process_transcription(self, text: str) -> Dict:
        """Process transcription through all agents"""
        result = {
            'original_text': text,
            'command_executed': False,
            'context': 'general',
            'completion': None,
            'suggestions': [],
            'final_text': text
        }
        
        # 1. Check for commands first
        if self.command_agent.process_text(text):
            result['command_executed'] = True
            return result
        
        # 2. Update context
        result['context'] = self.context_agent.update_context(text)
        
        # 3. Check for auto-completions
        completion = self.autocomplete_agent.suggest_completion(text)
        if completion:
            result['completion'] = completion
            result['final_text'] = completion
        
        # 4. Log workflow activity
        self.workflow_agent.log_activity(f"dictated: {text[:30]}...", result['context'])
        
        # 5. Get proactive suggestions
        result['suggestions'] = self.proactive_agent.analyze_patterns(text, [])
        
        # 6. Add workflow suggestions
        workflow_suggestion = self.workflow_agent.suggest_workflow()
        if workflow_suggestion:
            result['suggestions'].append(workflow_suggestion)
        
        return result
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            'config': asdict(self.config),
            'current_context': self.context_agent.current_context,
            'session_summary': self.workflow_agent.get_session_summary(),
            'command_history_size': len(self.command_agent.command_history),
            'learned_completions': len(self.autocomplete_agent.learned_completions)
        }

class VocalFlowAgentSystem:
    """Enhanced VocalFlow system with agents"""
    
    def __init__(self, profile_name: str = "default", config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.profile_name = profile_name
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('vocalflow.agents')
        
        # Initialize agent system
        self.agent_system = AgentSystem(self.config)
        
        # Initialize VocalFlow if available
        self.vocalflow_system = None
        if VOCALFLOW_AVAILABLE:
            self.vocalflow_system = VocalFlowSystem(profile_name)
        
        self.is_running = False
        
    def start(self):
        """Start the enhanced VocalFlow system with agents"""
        self.logger.info("Starting VocalFlow Agent System...")
        
        if self.vocalflow_system:
            # Integrate with main VocalFlow system
            original_process_buffer = self.vocalflow_system._process_audio_buffer
            
            def enhanced_process_buffer(audio_buffer):
                # Get transcription from VocalFlow
                texts = self.vocalflow_system.transcription_engine.transcribe_audio(audio_buffer)
                
                # Process through agents
                for text in texts:
                    if self.vocalflow_system.filter.is_good_text(text):
                        # Process through agent system
                        result = self.agent_system.process_transcription(text)
                        
                        if not result['command_executed']:
                            # Use completion if available, otherwise original text
                            final_text = result['completion'] or result['final_text']
                            formatted_text = self.vocalflow_system._format_text(final_text)
                            self.vocalflow_system.output_handler.type_text(formatted_text)
                        
                        # Show suggestions
                        for suggestion in result['suggestions']:
                            print(suggestion)
            
            # Replace the processing method
            self.vocalflow_system._process_audio_buffer = enhanced_process_buffer
            
            # Start VocalFlow with agent integration
            self.vocalflow_system.start()
            self.is_running = True
            
        else:
            # Standalone mode - simple text input for testing
            self.logger.info("Running in standalone mode (no VocalFlow integration)")
            self._run_standalone()
    
    def stop(self):
        """Stop the system"""
        self.is_running = False
        if self.vocalflow_system:
            self.vocalflow_system.stop()
        
        # Print session summary
        status = self.agent_system.get_system_status()
        print(f"\n📊 Session Summary:")
        print(f"Context: {status['current_context']}")
        print(f"Duration: {status['session_summary']['duration_minutes']:.1f} minutes")
        print(f"Commands executed: {status['command_history_size']}")
        print(f"Learned completions: {status['learned_completions']}")
    
    def _run_standalone(self):
        """Run in standalone mode for testing"""
        self.is_running = True
        print("🤖 VocalFlow Agents - Standalone Mode")
        print("Type text to process through agents, or 'quit' to exit")
        print("Try: 'execute open browser', 'good morning', etc.\n")
        
        while self.is_running:
            try:
                text = input(">>> ")
                if text.lower() in ['quit', 'exit']:
                    break
                
                result = self.agent_system.process_transcription(text)
                
                if not result['command_executed']:
                    if result['completion']:
                        print(f"💡 Completion: {result['completion']}")
                    print(f"📝 Context: {result['context']}")
                
                for suggestion in result['suggestions']:
                    print(suggestion)
                
            except KeyboardInterrupt:
                break
        
        self.stop()

def main():
    """Main entry point for VocalFlow Agents"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VocalFlow Agents - Intelligent Voice Assistant")
    parser.add_argument("--profile", "-p", default="default", help="Voice profile name")
    parser.add_argument("--enable-commands", action="store_true", help="Enable voice commands")
    parser.add_argument("--enable-context", action="store_true", help="Enable context awareness")
    parser.add_argument("--enable-autocomplete", action="store_true", help="Enable auto-completion")
    parser.add_argument("--enable-workflow", action="store_true", help="Enable workflow learning")
    parser.add_argument("--enable-proactive", action="store_true", help="Enable proactive suggestions")
    parser.add_argument("--standalone", action="store_true", help="Run in standalone mode")
    
    args = parser.parse_args()
    
    # Create configuration
    config = AgentConfig(
        enable_commands=args.enable_commands,
        enable_context=args.enable_context,
        enable_autocomplete=args.enable_autocomplete,
        enable_workflow=args.enable_workflow,
        enable_proactive=args.enable_proactive
    )
    
    # Create and start system
    system = VocalFlowAgentSystem(args.profile, config)
    
    try:
        print("🚀 Starting VocalFlow with Intelligent Agents...")
        print("🎙️ Say 'execute [command]' for voice commands")
        print("💡 System learns your patterns automatically")
        print("⛔ Press Ctrl+C to stop\n")
        
        system.start()
        
        if not args.standalone:
            # Keep running until interrupted
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n👋 Shutting down VocalFlow Agents...")
    finally:
        system.stop()

if __name__ == "__main__":
    main()
