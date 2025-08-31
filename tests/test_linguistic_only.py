#!/usr/bin/env python3
"""
Test just the linguistic components without VocalFlow main
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced_agents import EnhancedVocalFlowAgent

print("🧪 Testing Enhanced Agent (without VocalFlow main)...")

agent = EnhancedVocalFlowAgent()
print(f"📊 Status: {agent.get_status()}")

# Test commands
test_commands = [
    "open browser please",
    "take screenshot", 
    "volume up"
]

for cmd in test_commands:
    print(f"\n🔊 Testing: '{cmd}'")
    result = agent.process_text_enhanced(cmd)
    print(f"✅ Result: {result}")

print("✅ Linguistic testing complete!")
