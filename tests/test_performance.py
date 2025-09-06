#!/usr/bin/env python3
"""
Simple performance testing for VocalFlow system
Tests basic response times and memory usage without stress testing
"""
import sys
import os
import time
import psutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_agents import EnhancedVocalFlowAgent, AgentConfig


def test_initialization_time():
    """Test system initialization time"""
    print("Testing system initialization...")

    start_time = time.time()
    agent = EnhancedVocalFlowAgent()
    init_time = time.time() - start_time

    print(f"Initialization time: {init_time:.3f} seconds")
    return agent, init_time


def test_command_processing(agent):
    """Test individual command processing times"""
    print("Testing command processing speed...")

    test_commands = [
        "open browser",
        "take screenshot",
        "volume up",
        "good morning",
        "thank you for",
    ]

    processing_times = []

    for command in test_commands:
        start_time = time.time()
        result = agent.process_text_enhanced(command)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)

        print(f"  '{command}' -> {processing_time:.4f}s (executed: {result})")

    avg_time = sum(processing_times) / len(processing_times)
    print(f"Average processing time: {avg_time:.4f} seconds")

    return processing_times


def test_memory_usage():
    """Test memory usage"""
    print("Testing memory usage...")

    process = psutil.Process()
    baseline = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Baseline memory: {baseline:.2f} MB")

    # Create agent and measure
    agent = EnhancedVocalFlowAgent()
    after_init = process.memory_info().rss / 1024 / 1024
    print(f"After initialization: {after_init:.2f} MB")
    print(f"Initialization overhead: {after_init - baseline:.2f} MB")

    # Process a few commands
    for i in range(5):
        agent.process_text_enhanced(f"test command {i}")

    after_commands = process.memory_info().rss / 1024 / 1024
    print(f"After processing commands: {after_commands:.2f} MB")
    print(f"Processing overhead: {after_commands - after_init:.2f} MB")

    return {
        "baseline": baseline,
        "after_init": after_init,
        "after_commands": after_commands,
    }


def test_status_check(agent):
    """Test system status reporting"""
    print("Testing system status...")

    start_time = time.time()
    status = agent.get_status()
    status_time = time.time() - start_time

    print(f"Status check time: {status_time:.4f} seconds")
    print(f"Enhanced mode: {status.get('enhanced_mode', 'Unknown')}")
    print(f"Current context: {status.get('current_context', 'Unknown')}")
    print(f"Available commands: {len(status.get('available_commands', []))}")

    return status


def main():
    """Run simple performance tests"""
    print("VocalFlow Simple Performance Test")
    print("=" * 40)

    try:
        # Test initialization
        agent, init_time = test_initialization_time()
        print()

        # Test command processing
        processing_times = test_command_processing(agent)
        print()

        # Test memory usage
        memory_stats = test_memory_usage()
        print()

        # Test status
        status = test_status_check(agent)
        print()

        # Summary
        print("Performance Summary:")
        print(f"- Initialization: {init_time:.3f}s")
        print(
            f"- Average command processing: {sum(processing_times)/len(processing_times):.4f}s"
        )
        print(
            f"- Memory overhead: {memory_stats['after_init'] - memory_stats['baseline']:.2f} MB"
        )

        # Performance assessment
        if init_time < 3.0 and sum(processing_times) / len(processing_times) < 0.1:
            print("- Overall assessment: Good performance")
        elif init_time < 10.0 and sum(processing_times) / len(processing_times) < 0.5:
            print("- Overall assessment: Acceptable performance")
        else:
            print("- Overall assessment: May need optimization")

        print("\nSimple performance test completed successfully!")
        return 0

    except Exception as e:
        print(f"Performance test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
