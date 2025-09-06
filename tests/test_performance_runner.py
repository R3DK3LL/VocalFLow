#!/usr/bin/env python3
"""
Performance testing for VocalFlow system
Tests response time, memory usage, and processing efficiency
"""
import sys
import os
import time
import psutil
import threading
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_agents import EnhancedVocalFlowAgent, AgentConfig
from linguistic_bridge import LinguisticBridge


class PerformanceTestRunner:
    """Comprehensive performance testing for VocalFlow components"""

    def __init__(self):
        self.results = {
            "command_processing_times": [],
            "memory_usage": [],
            "initialization_time": 0,
            "throughput_tests": [],
            "stress_test_results": {},
        }

    def measure_initialization_time(self) -> float:
        """Test how long it takes to initialize the system"""
        print("Testing system initialization time...")

        start_time = time.time()

        # Initialize enhanced agent system
        config = AgentConfig(
            enable_commands=True,
            enable_context=True,
            enable_autocomplete=True,
            enable_workflow=True,
        )

        agent = EnhancedVocalFlowAgent(config)
        bridge = LinguisticBridge()

        initialization_time = time.time() - start_time
        self.results["initialization_time"] = initialization_time

        print(f"  Initialization time: {initialization_time:.3f} seconds")
        return initialization_time

    def test_command_processing_speed(self) -> Dict[str, float]:
        """Test response time for various commands"""
        print("Testing command processing speed...")

        agent = EnhancedVocalFlowAgent()

        test_commands = [
            "open browser",
            "take screenshot",
            "volume up",
            "launch terminal",
            "fire up browser please",
            "capture the screen",
            "increase system volume",
            "good morning",
            "thank you for your time",
            "def main function",
        ]

        processing_times = {}

        for command in test_commands:
            start_time = time.time()

            # Test complete processing pipeline
            result = agent.process_transcription_complete(command)

            processing_time = time.time() - start_time
            processing_times[command] = processing_time
            self.results["command_processing_times"].append(processing_time)

            print(f"  '{command}' -> {processing_time:.4f}s")

        avg_time = sum(processing_times.values()) / len(processing_times)
        print(f"  Average processing time: {avg_time:.4f}s")

        return processing_times

    def test_memory_usage(self) -> Dict[str, float]:
        """Monitor memory usage during operation"""
        print("Testing memory usage...")

        # Get baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"  Baseline memory: {baseline_memory:.2f} MB")

        # Initialize system and measure memory
        agent = EnhancedVocalFlowAgent()
        after_init_memory = process.memory_info().rss / 1024 / 1024

        print(f"  After initialization: {after_init_memory:.2f} MB")
        print(
            f"  Initialization overhead: {after_init_memory - baseline_memory:.2f} MB"
        )

        # Process commands and monitor memory
        test_commands = [
            "open browser please",
            "take screenshot now",
            "volume up quickly",
            "launch terminal application",
            "good morning everyone",
        ] * 10  # Run each 10 times

        memory_samples = []

        for i, command in enumerate(test_commands):
            agent.process_transcription_complete(command)

            if i % 5 == 0:  # Sample every 5 commands
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                self.results["memory_usage"].append(current_memory)

        max_memory = max(memory_samples)
        avg_memory = sum(memory_samples) / len(memory_samples)

        print(f"  Maximum memory usage: {max_memory:.2f} MB")
        print(f"  Average memory usage: {avg_memory:.2f} MB")
        print(
            f"  Memory stability: {'Stable' if max_memory - baseline_memory < 50 else 'Concerning'}"
        )

        return {
            "baseline": baseline_memory,
            "after_init": after_init_memory,
            "max_usage": max_memory,
            "avg_usage": avg_memory,
        }

    def test_throughput(self) -> Dict[str, Any]:
        """Test how many commands can be processed per second"""
        print("Testing command throughput...")

        agent = EnhancedVocalFlowAgent()

        # Simple commands for throughput testing
        commands = ["open browser", "volume up", "take screenshot", "good morning"]

        # Test sustained processing
        total_commands = 100
        start_time = time.time()

        for i in range(total_commands):
            command = commands[i % len(commands)]
            agent.process_transcription_complete(command)

        total_time = time.time() - start_time
        throughput = total_commands / total_time

        self.results["throughput_tests"].append(
            {
                "commands_processed": total_commands,
                "total_time": total_time,
                "commands_per_second": throughput,
            }
        )

        print(f"  Processed {total_commands} commands in {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} commands/second")

        return {
            "total_commands": total_commands,
            "total_time": total_time,
            "throughput": throughput,
        }

    def test_stress_conditions(self) -> Dict[str, Any]:
        """Test system under stress conditions"""
        print("Testing stress conditions...")

        agent = EnhancedVocalFlowAgent()

        # Test concurrent processing simulation
        def process_commands_batch(batch_size: int, batch_name: str):
            start_time = time.time()
            commands = ["open browser", "volume up", "take screenshot"] * (
                batch_size // 3
            )

            for command in commands:
                agent.process_transcription_complete(command)

            return time.time() - start_time

        stress_results = {}

        # Test different batch sizes
        batch_sizes = [10, 50, 100]

        for batch_size in batch_sizes:
            batch_time = process_commands_batch(batch_size, f"batch_{batch_size}")
            stress_results[f"batch_{batch_size}"] = {
                "commands": batch_size,
                "time": batch_time,
                "avg_per_command": batch_time / batch_size,
            }

            print(
                f"  Batch {batch_size}: {batch_time:.2f}s total, {batch_time/batch_size:.4f}s per command"
            )

        self.results["stress_test_results"] = stress_results
        return stress_results

    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete performance test suite"""
        print("Starting VocalFlow Performance Tests")
        print("=" * 50)

        # Run all performance tests
        self.measure_initialization_time()
        print()

        self.test_command_processing_speed()
        print()

        self.test_memory_usage()
        print()

        self.test_throughput()
        print()

        self.test_stress_conditions()
        print()

        # Generate summary report
        self.generate_summary_report()

        return self.results

    def generate_summary_report(self):
        """Generate performance summary report"""
        print("Performance Test Summary")
        print("=" * 30)

        # Overall assessment
        init_time = self.results["initialization_time"]
        avg_processing = sum(self.results["command_processing_times"]) / len(
            self.results["command_processing_times"]
        )

        print(f"System Initialization: {init_time:.3f}s")
        print(f"Average Command Processing: {avg_processing:.4f}s")

        if self.results["throughput_tests"]:
            throughput = self.results["throughput_tests"][0]["commands_per_second"]
            print(f"Command Throughput: {throughput:.1f} commands/second")

        # Performance rating
        if init_time < 2.0 and avg_processing < 0.1:
            rating = "Excellent"
        elif init_time < 5.0 and avg_processing < 0.5:
            rating = "Good"
        elif init_time < 10.0 and avg_processing < 1.0:
            rating = "Acceptable"
        else:
            rating = "Needs Optimization"

        print(f"Overall Performance Rating: {rating}")

        # Recommendations
        print("\nRecommendations:")
        if avg_processing > 0.5:
            print("- Consider optimizing command processing pipeline")
        if init_time > 5.0:
            print("- Consider lazy loading of components")
        if self.results["memory_usage"] and max(self.results["memory_usage"]) > 200:
            print("- Monitor memory usage for potential leaks")

        print("\nPerformance testing complete!")


def main():
    """Run performance tests"""
    try:
        test_runner = PerformanceTestRunner()
        results = test_runner.run_all_tests()

        # Save results to file for analysis
        import json

        with open("performance_test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nDetailed results saved to: performance_test_results.json")

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install required packages: pip install psutil")
        return 1
    except Exception as e:
        print(f"Performance test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
