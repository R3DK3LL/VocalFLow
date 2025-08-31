#!/usr/bin/env python3
"""
Enhanced VocalFlow with Linguistic Integration
Wrapper that enhances existing VocalFlow without modifying core files
"""

import sys
import os
from typing import Any, Dict

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from vocalflow.main import VocalFlowEngine
    from enhanced_agents import EnhancedVocalFlowAgent
    print("🎯 VocalFlow Enhanced Edition Starting...")
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Integration not available: {e}")
    INTEGRATION_AVAILABLE = False
    sys.exit(1)


class EnhancedVocalFlow:
    """Enhanced VocalFlow with linguistic command processing"""
    
    def __init__(self):
        # Initialize core VocalFlow
        self.vocalflow = VocalFlowEngine()
        
        # Initialize enhanced agent
        self.enhanced_agent = EnhancedVocalFlowAgent()
        
        print("🚀 Enhanced VocalFlow initialized")
        print(f"📊 Status: {self.enhanced_agent.get_status()}")
    
    def process_transcription_enhanced(self, text: str) -> bool:
        """Enhanced transcription processing"""
        print(f"🎙️  Processing: '{text}'")
        
        # First try enhanced command processing
        if self.enhanced_agent.process_text_enhanced(text):
            print("✅ Processed as enhanced command")
            return True
        
        # Fall back to regular transcription
        print("📝 Processing as regular transcription")
        # Here you would call the original VocalFlow transcription processing
        return False
    
    def start_enhanced(self):
        """Start enhanced VocalFlow"""
        print("\n🎯 Enhanced VocalFlow Ready!")
        print("📋 Available enhanced commands:")
        print("   • 'open browser' - Launch web browser")
        print("   • 'take screenshot' - Capture screen")
        print("   • 'volume up/down' - Control system volume")
        print("   • 'open terminal' - Launch terminal")
        print("   • Plus 227+ command variations!")
        print("\n💡 Say 'exit enhanced mode' to return to normal VocalFlow")
        print("🎤 Listening for enhanced commands...\n")


if __name__ == "__main__":
    try:
        enhanced_vf = EnhancedVocalFlow()
        enhanced_vf.start_enhanced()
        
        # Demo mode for testing
        test_commands = [
            "open browser please",
            "take a quick screenshot", 
            "turn the volume up",
            "launch terminal"
        ]
        
        print("🧪 Demo Mode - Testing Commands:")
        for cmd in test_commands:
            print(f"\n🔊 Simulating: '{cmd}'")
            enhanced_vf.process_transcription_enhanced(cmd)
            
    except KeyboardInterrupt:
        print("\n👋 Enhanced VocalFlow stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
