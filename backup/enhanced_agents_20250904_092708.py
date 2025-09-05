#!/usr/bin/env python3
"""
Enhanced VocalFlow Agents
Intelligent Agent System with Linguistic Integration, Context Awareness, 
Auto-completion, Workflow Learning, and Proactive Suggestions
"""

import sys
import os
import time
import re
import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, asdict
import logging

# Import linguistic components
sys.path.append(os.path.dirname(__file__))
try:
    from linguistic_bridge import LinguisticBridge, AgentLinguisticIntegrator
    LINGUISTIC_AVAILABLE = True
    print("Enhanced linguistic processing available")
except ImportError as e:
    print(f"Enhanced mode not available: {e}")
    LINGUISTIC_AVAILABLE = False

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
                    print(f"Context: {best_context}")
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
            'coding': "Tip: Say 'execute open terminal' to open terminal",
            'email': "Tip: Say 'execute new document' for email template", 
            'documentation': "Tip: Use markdown formatting for headers",
            'creative_writing': "Tip: Focus on dialogue and scene description"
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
                return f"Usually at this time you: {most_common}"
        
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
        
    def analyze_patterns(self, text: str, corrections: List[Tuple[str, str]] = None):
        """Analyze patterns and make proactive suggestions"""
        if not self.config.enable_proactive:
            return []
        
        if corrections is None:
            corrections = []
            
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
                        f"I notice you often correct '{pattern.split('->')[0]}' to "
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
                        f"You often say '{phrase}'. Want to create a shortcut?"
                    )
        
        return suggestions
    
    def _suggest_shortcuts(self) -> List[str]:
        """Suggest useful shortcuts"""
        suggestions = []
        
        # Context-based suggestions
        if self._can_suggest('general', 'shortcuts'):
            suggestions.append("Tip: Say 'execute' before commands like 'open browser'")
        
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

class EnhancedVocalFlowAgent:
    """Complete enhanced agent with all features integrated"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.logger = logging.getLogger('vocalflow.agents.enhanced')
        
        # Initialize linguistic components
        self.bridge = None
        self.integrator = None
        
        if LINGUISTIC_AVAILABLE:
            try:
                self.bridge = LinguisticBridge()
                self.integrator = AgentLinguisticIntegrator(self.bridge)
                print(f"Linguistic Status: {self.bridge.get_status()}")
            except Exception as e:
                print(f"Warning: Linguistic components failed: {e}")
                
        # Initialize all agent components
        self.context_agent = ContextAwareAgent(self.config)
        self.autocomplete_agent = AutoCompleteAgent(self.config)
        self.workflow_agent = WorkflowAgent(self.config)
        self.proactive_agent = ProactiveAgent(self.config)
        
        # Command patterns and their actions (from original agents.py)
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
        self.agent_messages = deque(maxlen=50)
    
    def process_text_enhanced(self, text: str) -> bool:
        """Enhanced text processing with full agent integration"""
        # Update context first
        current_context = self.context_agent.update_context(text)
        
        # Try linguistic processing if available
        if LINGUISTIC_AVAILABLE and self.bridge:
            is_command, command_type, metadata = self.bridge.process_command(text)
            
            if is_command:
                confidence = metadata.get('confidence', 0)
                method = metadata.get('method', 'unknown')
                print(f"Enhanced match: '{command_type}' (confidence: {confidence:.2f}, method: {method})")
                
                # Execute the enhanced command
                success = self._execute_enhanced_command(command_type, text, metadata)
                if success:
                    self.workflow_agent.log_activity(f"command: {command_type}", current_context)
                    return True
        
        # Try traditional command patterns with trigger word
        if self.config.command_trigger.lower() in text.lower():
            clean_text = text.lower().replace(self.config.command_trigger.lower(), "").strip()
            
            for pattern, action in self.commands.items():
                match = re.search(pattern, clean_text)
                if match:
                    try:
                        if callable(action):
                            if hasattr(action, '__code__') and action.__code__.co_argcount > 1:
                                action(match)
                            else:
                                action()
                        
                        self.command_history.append({
                            'command': clean_text,
                            'timestamp': datetime.now().isoformat(),
                            'pattern': pattern
                        })
                        
                        self.logger.info(f"Executed command: {clean_text}")
                        print(f"Executed: {clean_text}")
                        self.workflow_agent.log_activity(f"command: {clean_text}", current_context)
                        return True
                        
                    except Exception as e:
                        self.logger.error(f"Command execution failed: {e}")
                        print(f"Command failed: {clean_text}")
        
        return False
    
    def process_transcription_complete(self, text: str) -> Dict:
        """Complete transcription processing through all agents"""
        result = {
            'original_text': text,
            'command_executed': False,
            'context': 'general',
            'completion': None,
            'suggestions': [],
            'final_text': text
        }
        
        # 1. Check for commands first
        if self.process_text_enhanced(text):
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
        
        # 7. Add context suggestions
        context_suggestion = self.context_agent.suggest_context_action()
        if context_suggestion:
            result['suggestions'].append(context_suggestion)
        
        return result
    
    def _execute_enhanced_command(self, command_type: str, text: str, metadata: Dict) -> bool:
        """Execute commands using enhanced linguistic processing"""
        
        # Browser operations
        if command_type in ['browser_operations', 'open_browser']:
            return self._handle_browser_command(text)
        
        # Volume control
        elif command_type in ['volume_control', 'volume_up', 'volume_down']:
            return self._handle_volume_command(text)
        
        # Screenshot
        elif command_type in ['screenshot', 'take_screenshot']:
            return self._handle_screenshot_command()
        
        # Terminal
        elif command_type in ['terminal_operations', 'open_terminal']:
            return self._handle_terminal_command()
        
        return False
    
    def _handle_browser_command(self, text: str) -> bool:
        """Handle browser commands with intelligent discovery"""
        app_type = 'browser'
        executable = self._discover_application(app_type)
        
        if executable:
            try:
                subprocess.Popen([executable])
                print(f"Opened {executable}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to launch {executable}: {e}")
        
        print("No browser found")
        return False
    
    def _discover_application(self, app_type: str) -> Optional[str]:
        """Intelligently discover application executable"""
        app_searches = {
            'browser': ['firefox', 'google-chrome', 'chromium-browser', 'chromium', 'opera', 'brave-browser'],
            'terminal': ['gnome-terminal', 'konsole', 'xterm', 'alacritty', 'terminator', 'tilix'],
            'files': ['nautilus', 'dolphin', 'thunar', 'pcmanfm', 'nemo'],
            'calculator': ['gnome-calculator', 'kcalc', 'galculator'],
            'text': ['gedit', 'kate', 'mousepad', 'leafpad', 'nano'],
            'editor': ['gedit', 'kate', 'mousepad', 'vim', 'nano'],
            'code': ['code', 'codium', 'atom', 'subl']
        }
        
        candidates = app_searches.get(app_type, [app_type])
        
        for candidate in candidates:
            if self._check_executable_exists(candidate):
                return candidate
        
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
    
    def _handle_volume_command(self, text: str) -> bool:
        """Handle volume commands"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['up', 'increase', 'louder', 'higher']):
            return self._volume_up()
        elif any(word in text_lower for word in ['down', 'decrease', 'lower', 'quieter']):
            return self._volume_down()
        elif any(word in text_lower for word in ['mute', 'silence']):
            return self._volume_mute()
        
        return False
    
    def _volume_up(self) -> bool:
        """Increase system volume"""
        try:
            subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', '+10%'], 
                         check=True, capture_output=True)
            print("Volume increased")
            return True
        except subprocess.CalledProcessError:
            try:
                subprocess.run(['amixer', 'set', 'Master', '10%+'], 
                             check=True, capture_output=True)
                print("Volume increased (amixer)")
                return True
            except:
                print("Could not control volume")
                return False
    
    def _volume_down(self) -> bool:
        """Decrease system volume"""
        try:
            subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', '-10%'], 
                         check=True, capture_output=True)
            print("Volume decreased")
            return True
        except subprocess.CalledProcessError:
            try:
                subprocess.run(['amixer', 'set', 'Master', '10%-'], 
                             check=True, capture_output=True)
                print("Volume decreased (amixer)")
                return True
            except:
                print("Could not control volume")
                return False
    
    def _volume_mute(self) -> bool:
        """Toggle mute"""
        try:
            subprocess.run(['pactl', 'set-sink-mute', '@DEFAULT_SINK@', 'toggle'], 
                         check=True, capture_output=True)
            print("Volume muted/unmuted")
            return True
        except subprocess.CalledProcessError:
            try:
                subprocess.run(['amixer', 'set', 'Master', 'toggle'], 
                             check=True, capture_output=True)
                print("Volume muted/unmuted (amixer)")
                return True
            except:
                print("Could not mute volume")
                return False
    
    def _handle_screenshot_command(self) -> bool:
        """Take a screenshot"""
        screenshots_dir = f"/home/{os.getenv('USER', 'user')}/Pictures/VocalFlow_Screenshots"
        os.makedirs(screenshots_dir, exist_ok=True)
        
        timestamp = int(time.time())
        filename = f"{screenshots_dir}/screenshot_{timestamp}.png"
        
        screenshot_tools = [
            ['gnome-screenshot', '-f', filename],
            ['scrot', filename],
            ['import', filename],
            ['maim', filename]
        ]
        
        for tool in screenshot_tools:
            try:
                subprocess.run(tool, check=True, capture_output=True)
                print(f"Screenshot saved: {filename}")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        print("No screenshot tool found")
        return False
    
    def _handle_terminal_command(self) -> bool:
        """Open terminal application"""
        executable = self._discover_application('terminal')
        
        if executable:
            try:
                subprocess.Popen([executable])
                print(f"Opened {executable}")
                return True
            except Exception:
                pass
        
        print("No terminal found")
        return False
    
    def _try_open_application(self, app_name: str) -> bool:
        """Try to open an application"""
        import shutil
        
        if not shutil.which(app_name):
            return False
        
        try:
            subprocess.Popen([app_name], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False
    
    # Additional command methods from agents.py
    def _open_application(self, match):
        """Open application by name with intelligent discovery"""
        app_name = match.group(1).lower()
        executable = self._discover_application(app_name)
        
        if executable:
            try:
                subprocess.Popen([executable])
                self.logger.info(f"Successfully opened {executable}")
            except Exception as e:
                self.logger.error(f"Failed to launch {executable}: {e}")
        else:
            print(f"Application '{app_name}' not found on system")
    
    def _close_application(self, match):
        """Close application by name"""
        app_name = match.group(1)
        subprocess.run(["pkill", "-f", app_name])
    
    def _switch_application(self, match):
        """Switch to application window"""
        app_name = match.group(1)
        try:
            subprocess.run(["wmctrl", "-a", app_name])
        except FileNotFoundError:
            print("wmctrl not available for window switching")
    
    def _new_file(self, match):
        """Create new file"""
        file_type = match.group(1)
        editor = self._discover_application('code') or self._discover_application('editor')
        if editor:
            if file_type == "file":
                subprocess.Popen([editor, "untitled.txt"])
            else:
                subprocess.Popen([editor])
    
    def _save_file(self, match):
        """Save current file"""
        self._key_combo("ctrl+s")
    
    def _open_file(self, match):
        """Open specific file"""
        filename = match.group(1)
        editor = self._discover_application('code') or self._discover_application('editor')
        if editor:
            subprocess.Popen([editor, filename])
    
    def _control_volume(self, match):
        """Control system volume"""
        action = match.group(1)
        volume_commands = {
            'up': ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+5%"],
            'down': ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-5%"],
            'mute': ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"]
        }
        try:
            subprocess.run(volume_commands[action])
        except KeyError:
            print(f"Unknown volume action: {action}")
    
    def _control_brightness(self, match):
        """Control screen brightness"""
        action = match.group(1)
        try:
            if action == "up":
                subprocess.run(["xbacklight", "-inc", "10"])
            else:
                subprocess.run(["xbacklight", "-dec", "10"])
        except FileNotFoundError:
            print("xbacklight not available for brightness control")
    
    def _scroll_page(self, match):
        """Scroll page up or down"""
        direction = match.group(1)
        if direction == "up":
            self._key_combo("Page_Up")
        else:
            self._key_combo("Page_Down")
    
    def _key_combo(self, keys: str):
        """Send keyboard combination"""
        try:
            subprocess.run(["xdotool", "key", keys])
        except FileNotFoundError:
            print("xdotool not available for key combinations")
    
    def _start_coding_session(self, match=None):
        """Start a coding session workflow"""
        editor = self._discover_application('code')
        browser = self._discover_application('browser') 
        terminal = self._discover_application('terminal')
        
        if editor:
            subprocess.Popen([editor])
        if browser:
            time.sleep(2)
            subprocess.Popen([browser, "https://github.com"])
        if terminal:
            subprocess.Popen([terminal])
    
    def _end_session(self, match=None):
        """End work session"""
        self._key_combo("ctrl+s")
        print("Session ended, files saved")
    
    def _take_screenshot(self, match=None):
        """Take a screenshot with timestamp"""
        return self._handle_screenshot_command()
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete enhanced agent status"""
        base_status = {
            'enhanced_mode': LINGUISTIC_AVAILABLE and self.bridge is not None,
            'linguistic_available': LINGUISTIC_AVAILABLE,
            'current_context': self.context_agent.current_context,
            'config': asdict(self.config),
            'available_commands': ['browser', 'volume', 'screenshot', 'terminal'],
            'command_history_size': len(self.command_history),
            'learned_completions': len(self.autocomplete_agent.learned_completions)
        }
        
        if LINGUISTIC_AVAILABLE and self.bridge:
            base_status['linguistic_status'] = self.bridge.get_status()
        
        return base_status
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            'config': asdict(self.config),
            'current_context': self.context_agent.current_context,
            'session_summary': self.workflow_agent.get_session_summary(),
            'command_history_size': len(self.command_history),
            'learned_completions': len(self.autocomplete_agent.learned_completions),
            'context_vocabulary': self.context_agent.get_context_vocabulary()
        }
    
    def learn_custom_completion(self, trigger: str, completion: str):
        """Learn a custom completion pattern"""
        self.autocomplete_agent.learn_completion(trigger, completion)
    
    def get_suggestions(self) -> List[str]:
        """Get current suggestions from all agents"""
        suggestions = []
        
        # Context suggestions
        context_suggestion = self.context_agent.suggest_context_action()
        if context_suggestion:
            suggestions.append(context_suggestion)
        
        # Workflow suggestions  
        workflow_suggestion = self.workflow_agent.suggest_workflow()
        if workflow_suggestion:
            suggestions.append(workflow_suggestion)
        
        # Proactive suggestions
        proactive_suggestions = self.proactive_agent.analyze_patterns("", [])
        suggestions.extend(proactive_suggestions)
        
        return suggestions

# Standalone testing and compatibility functions
def test_enhanced_agent():
    """Test the complete enhanced agent functionality"""
    print("Testing Complete Enhanced VocalFlow Agent...")
    
    config = AgentConfig(
        enable_commands=True,
        enable_context=True, 
        enable_autocomplete=True,
        enable_workflow=True,
        enable_proactive=False
    )
    
    agent = EnhancedVocalFlowAgent(config)
    print(f"Agent Status: {agent.get_status()}")
    
    # Test commands
    test_commands = [
        "open browser please",
        "take a screenshot", 
        "volume up",
        "launch terminal",
        "execute open browser",
        "execute new file",
        "good morning",
        "def main",
        "following up"
    ]
    
    print("\nTesting Enhanced Commands and Features:")
    for cmd in test_commands:
        print(f"\nTesting: '{cmd}'")
        
        # Test complete processing
        result = agent.process_transcription_complete(cmd)
        
        if result['command_executed']:
            print(f"  Command executed successfully")
        elif result['completion']:
            print(f"  Auto-completion: {result['completion']}")
        else:
            print(f"  Processed as text in context: {result['context']}")
        
        # Show suggestions
        for suggestion in result['suggestions']:
            print(f"  Suggestion: {suggestion}")
    
    print(f"\nFinal Status: {agent.get_system_status()}")
    print("\nComplete enhanced agent testing complete!")

# Backward compatibility with original interface
class VocalFlowAgentSystem:
    """Wrapper for backward compatibility with enhanced system"""
    
    def __init__(self, profile_name: str = "default", config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.profile_name = profile_name
        self.agent = EnhancedVocalFlowAgent(self.config)
        self.is_running = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('vocalflow.agents')
    
    def start(self):
        """Start the enhanced system"""
        self.logger.info("Starting VocalFlow Agent System...")
        self.is_running = True
        
        # In standalone mode - simple text input for testing  
        self._run_standalone()
    
    def stop(self):
        """Stop the system"""
        self.is_running = False
        
        # Print session summary
        status = self.agent.get_system_status()
        print(f"\nSession Summary:")
        print(f"Context: {status['current_context']}")
        print(f"Duration: {status['session_summary']['duration_minutes']:.1f} minutes")
        print(f"Commands executed: {status['command_history_size']}")
        print(f"Learned completions: {status['learned_completions']}")
    
    def _run_standalone(self):
        """Run in standalone mode for testing"""
        print("VocalFlow Agents - Enhanced Mode")
        print("Type text to process through agents, or 'quit' to exit")
        print("Try: 'execute open browser', 'good morning', 'def main', etc.\n")
        
        while self.is_running:
            try:
                text = input(">>> ")
                if text.lower() in ['quit', 'exit']:
                    break
                
                result = self.agent.process_transcription_complete(text)
                
                if not result['command_executed']:
                    if result['completion']:
                        print(f"Auto-completion: {result['completion']}")
                    print(f"Context: {result['context']}")
                
                for suggestion in result['suggestions']:
                    print(suggestion)
                
            except KeyboardInterrupt:
                break
        
        self.stop()
    
    def process_transcription(self, text: str) -> Dict:
        """Process transcription - compatibility method"""
        return self.agent.process_transcription_complete(text)

def main():
    """Main entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced VocalFlow Agents")
    parser.add_argument("--profile", "-p", default="default", help="Voice profile name")
    parser.add_argument("--enable-commands", action="store_true", default=True, help="Enable voice commands")
    parser.add_argument("--enable-context", action="store_true", default=True, help="Enable context awareness")
    parser.add_argument("--enable-autocomplete", action="store_true", default=True, help="Enable auto-completion")
    parser.add_argument("--enable-workflow", action="store_true", default=True, help="Enable workflow learning")
    parser.add_argument("--enable-proactive", action="store_true", help="Enable proactive suggestions")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--standalone", action="store_true", help="Run in standalone mode")
    
    args = parser.parse_args()
    
    if args.test:
        test_enhanced_agent()
        return
    
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
        print("Starting Enhanced VocalFlow with Complete Agent System...")
        print("Features: Linguistic Commands, Context Awareness, Auto-completion, Workflow Learning")
        print("Say 'execute [command]' for voice commands")
        print("System learns your patterns automatically")
        print("Press Ctrl+C to stop\n")
        
        system.start()
        
    except KeyboardInterrupt:
        print("\nShutting down Enhanced VocalFlow Agents...")
    finally:
        system.stop()

if __name__ == "__main__":
    main()
