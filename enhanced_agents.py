#!/usr/bin/env python3
"""
Enhanced VocalFlow Agents with Linguistic Integration
Non-disruptive wrapper around existing agents.py
"""

import sys
import os
import time
from typing import Dict, Any, Optional

# Import original agents
sys.path.append(os.path.dirname(__file__))
try:
    import agents
    from linguistic_bridge import LinguisticBridge, AgentLinguisticIntegrator
    ENHANCED_MODE = True
    print("🎯 VocalFlow Enhanced Mode: ENABLED")
except ImportError as e:
    print(f"⚠️  Enhanced mode not available: {e}")
    ENHANCED_MODE = False


class EnhancedVocalFlowAgent:
    """Enhanced agent with linguistic integration"""
    
    def __init__(self):
        # Initialize linguistic components
        if ENHANCED_MODE:
            self.bridge = LinguisticBridge()
            self.integrator = AgentLinguisticIntegrator(self.bridge)
            print(f"📊 Linguistic Status: {self.bridge.get_status()}")
        else:
            self.bridge = None
            self.integrator = None
        
        # Store original agent functions for fallback
        self.original_agent = None
    
    def process_text_enhanced(self, text: str) -> bool:
        """Enhanced text processing with linguistic integration"""
        if not ENHANCED_MODE:
            return False
        
        # Try linguistic processing
        is_command, command_type, metadata = self.bridge.process_command(text)
        
        if is_command:
            confidence = metadata.get('confidence', 0)
            method = metadata.get('method', 'unknown')
            print(f"🎯 Enhanced match: '{command_type}' (confidence: {confidence:.2f}, method: {method})")
            
            # Execute the enhanced command
            success = self._execute_enhanced_command(command_type, text, metadata)
            if success:
                return True
        
        return False
    
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
        """Handle browser commands"""
        browsers = ['firefox', 'google-chrome', 'chromium-browser', 'brave-browser']
        
        for browser in browsers:
            if self._try_open_application(browser):
                print(f"🌐 Opened {browser}")
                return True
        
        print("⚠️  No browser found")
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
        import subprocess
        try:
            subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', '+10%'], 
                         check=True, capture_output=True)
            print("🔊 Volume increased")
            return True
        except subprocess.CalledProcessError:
            try:
                subprocess.run(['amixer', 'set', 'Master', '10%+'], 
                             check=True, capture_output=True)
                print("🔊 Volume increased (amixer)")
                return True
            except:
                print("⚠️  Could not control volume")
                return False
    
    def _volume_down(self) -> bool:
        """Decrease system volume"""
        import subprocess
        try:
            subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', '-10%'], 
                         check=True, capture_output=True)
            print("🔉 Volume decreased")
            return True
        except subprocess.CalledProcessError:
            try:
                subprocess.run(['amixer', 'set', 'Master', '10%-'], 
                             check=True, capture_output=True)
                print("🔉 Volume decreased (amixer)")
                return True
            except:
                print("⚠️  Could not control volume")
                return False
    
    def _volume_mute(self) -> bool:
        """Toggle mute"""
        import subprocess
        try:
            subprocess.run(['pactl', 'set-sink-mute', '@DEFAULT_SINK@', 'toggle'], 
                         check=True, capture_output=True)
            print("🔇 Volume muted/unmuted")
            return True
        except subprocess.CalledProcessError:
            try:
                subprocess.run(['amixer', 'set', 'Master', 'toggle'], 
                             check=True, capture_output=True)
                print("🔇 Volume muted/unmuted (amixer)")
                return True
            except:
                print("⚠️  Could not mute volume")
                return False
    
    def _handle_screenshot_command(self) -> bool:
        """Take a screenshot"""
        import subprocess
        
        # Create screenshots directory if it doesn't exist
        screenshots_dir = f"/home/{os.getenv('USER', 'user')}/Pictures/VocalFlow_Screenshots"
        os.makedirs(screenshots_dir, exist_ok=True)
        
        timestamp = int(time.time())
        filename = f"{screenshots_dir}/screenshot_{timestamp}.png"
        
        # Try different screenshot tools
        screenshot_tools = [
            ['gnome-screenshot', '-f', filename],
            ['scrot', filename],
            ['import', filename],  # ImageMagick
            ['maim', filename]
        ]
        
        for tool in screenshot_tools:
            try:
                subprocess.run(tool, check=True, capture_output=True)
                print(f"📸 Screenshot saved: {filename}")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        print("⚠️  No screenshot tool found")
        return False
    
    def _handle_terminal_command(self) -> bool:
        """Open terminal application"""
        terminals = [
            'gnome-terminal',
            'konsole', 
            'xfce4-terminal',
            'terminator',
            'alacritty',
            'xterm'
        ]
        
        for terminal in terminals:
            if self._try_open_application(terminal):
                print(f"💻 Opened {terminal}")
                return True
        
        print("⚠️  No terminal found")
        return False
    
    def _try_open_application(self, app_name: str) -> bool:
        """Try to open an application"""
        import subprocess
        import shutil
        
        # Check if application exists
        if not shutil.which(app_name):
            return False
        
        try:
            subprocess.Popen([app_name], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get enhanced agent status"""
        if ENHANCED_MODE and self.bridge:
            return {
                'enhanced_mode': True,
                'linguistic_status': self.bridge.get_status(),
                'available_commands': ['browser', 'volume', 'screenshot', 'terminal']
            }
        else:
            return {
                'enhanced_mode': False,
                'reason': 'Linguistic components not available'
            }


def test_enhanced_agent():
    """Test the enhanced agent functionality"""
    print("🧪 Testing Enhanced VocalFlow Agent...")
    
    agent = EnhancedVocalFlowAgent()
    print(f"📊 Agent Status: {agent.get_status()}")
    
    # Test commands
    test_commands = [
        "open browser please",
        "take a screenshot",
        "volume up",
        "launch terminal",
        "fire up browser",
        "capture screen"
    ]
    
    print("\n🎯 Testing Enhanced Commands:")
    for cmd in test_commands:
        print(f"\n📝 Testing: '{cmd}'")
        result = agent.process_text_enhanced(cmd)
        print(f"✅ Success: {result}")
    
    print("\n✅ Enhanced agent testing complete!")


if __name__ == "__main__":
    test_enhanced_agent()
