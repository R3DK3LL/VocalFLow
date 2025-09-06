#!/usr/bin/env python3
"""
Comprehensive test of unified VocalFlow enhanced agent system
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_agents import EnhancedVocalFlowAgent, AgentConfig

print("Testing Complete Enhanced VocalFlow Agent System...")

# Test with full configuration
config = AgentConfig(
    enable_commands=True,
    enable_context=True,
    enable_autocomplete=True,
    enable_workflow=True,
    enable_proactive=False,
)

agent = EnhancedVocalFlowAgent(config)
print(f"System Status: {agent.get_status()}")

# Test comprehensive command variations
test_commands = [
    # Enhanced linguistic variations
    "open browser please",
    "fire up browser",
    "launch terminal",
    "take a screenshot",
    "capture screen",
    "volume up",
    "increase volume",
    "execute open browser",
    # Auto-completion tests
    "good morning",
    "thank you for",
    "def main",
    # Context detection tests
    "def calculate_sum(a, b):",  # coding context
    "Dear John,",  # email context
    "The character walked slowly",  # creative writing
]

print("\nTesting Enhanced Features:")
for cmd in test_commands:
    print(f"\nTesting: '{cmd}'")
    result = agent.process_transcription_complete(cmd)

    if result["command_executed"]:
        print(f"  Command executed successfully")
    elif result["completion"]:
        print(f"  Auto-completion: {result['completion']}")
    else:
        print(f"  Processed as text in context: {result['context']}")

    for suggestion in result["suggestions"]:
        print(f"  Suggestion: {suggestion}")

print(f"\nFinal System Status: {agent.get_system_status()}")
print("Comprehensive testing complete!")
