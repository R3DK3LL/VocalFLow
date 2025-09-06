#!/usr/bin/env python3
"""
Accuracy testing for VocalFlow command recognition
Tests command recognition reliability, false positives, and confidence scoring
"""
import sys
import os
from typing import Dict, List, Tuple, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_agents import EnhancedVocalFlowAgent
from linguistic_bridge import LinguisticBridge


class AccuracyTestRunner:
    """Test command recognition accuracy and reliability"""

    def __init__(self):
        self.agent = EnhancedVocalFlowAgent()
        self.bridge = LinguisticBridge()
        self.results = {
            "command_recognition": [],
            "false_positives": [],
            "confidence_scores": [],
            "variation_coverage": {},
        }

    def test_command_variations(self) -> Dict[str, Any]:
        """Test recognition of different command variations"""
        print("Testing command variation recognition...")

        # Test different ways to say the same commands
        command_variations = {
            "browser_operations": [
                "open browser",
                "launch browser",
                "fire up browser",
                "start browser",
                "bring up browser",
                "open the browser please",
                "could you open browser",
                "launch web browser",
            ],
            "volume_control": [
                "volume up",
                "increase volume",
                "louder",
                "turn up volume",
                "boost volume",
                "volume higher",
                "make it louder",
                "crank up volume",
            ],
            "screenshot": [
                "take screenshot",
                "capture screen",
                "screenshot",
                "snap screen",
                "grab screenshot",
                "screen capture",
                "take picture of screen",
            ],
            "terminal_operations": [
                "open terminal",
                "launch terminal",
                "start terminal",
                "bring up terminal",
                "fire up terminal",
                "open command line",
            ],
        }

        category_results = {}
        total_tested = 0
        total_recognized = 0

        for category, variations in command_variations.items():
            category_correct = 0
            category_total = len(variations)

            print(f"\n  Testing {category}:")

            for variation in variations:
                is_command, command_type, metadata = self.bridge.process_command(
                    variation
                )
                confidence = metadata.get("confidence", 0)
                method = metadata.get("method", "none")

                # Map actual command types to test categories
                command_mappings = {
                    "open_browser": "browser_operations",
                    "browser_operations": "browser_operations",
                    "screenshot": "screenshot",
                    "take_screenshot": "screenshot",
                    "volume_control": "volume_control",
                    "volume_up": "volume_control",
                    "volume_down": "volume_control",
                    "open_terminal": "terminal_operations",
                    "terminal_operations": "terminal_operations",
                }

                # Check if recognized correctly using mapping
                expected_command = command_mappings.get(command_type, command_type)
                correct = is_command and expected_command == category

                if correct:
                    category_correct += 1
                    total_recognized += 1

                total_tested += 1

                print(
                    f"    '{variation}' -> {is_command} ({command_type}, conf: {confidence:.2f}, {method}) {'✓' if correct else '✗'}"
                )

                self.results["command_recognition"].append(
                    {
                        "text": variation,
                        "expected_category": category,
                        "recognized": is_command,
                        "actual_type": command_type,
                        "confidence": confidence,
                        "method": method,
                        "correct": correct,
                    }
                )

            accuracy = category_correct / category_total
            category_results[category] = {
                "correct": category_correct,
                "total": category_total,
                "accuracy": accuracy,
            }

            print(
                f"    Category accuracy: {accuracy:.2%} ({category_correct}/{category_total})"
            )

        overall_accuracy = total_recognized / total_tested
        print(
            f"\nOverall command recognition accuracy: {overall_accuracy:.2%} ({total_recognized}/{total_tested})"
        )

        return category_results

    def test_false_positives(self) -> Dict[str, Any]:
        """Test for false positive command recognition"""
        print("Testing false positive rates...")

        # Text that should NOT be recognized as commands
        non_command_texts = [
            "Hello, how are you today?",
            "The weather is nice outside.",
            "I need to write a report for work.",
            "Python is a programming language.",
            "The meeting is scheduled for tomorrow.",
            "Can you help me with this problem?",
            "This is just normal dictation text.",
            "I am writing an email to my colleague.",
            "The browser crashed yesterday.",
            "My computer screen is too bright.",
            "The terminal window was open.",
            "I took a screenshot of the error.",
            "Random words: elephant, mountain, bicycle, coffee",
        ]

        false_positives = 0
        total_tests = len(non_command_texts)

        for text in non_command_texts:
            is_command, command_type, metadata = self.bridge.process_command(text)
            confidence = metadata.get("confidence", 0)

            if is_command:
                false_positives += 1
                print(
                    f"    FALSE POSITIVE: '{text}' -> {command_type} (conf: {confidence:.2f})"
                )

                self.results["false_positives"].append(
                    {
                        "text": text,
                        "command_type": command_type,
                        "confidence": confidence,
                    }
                )
            else:
                print(f"    ✓ Correctly ignored: '{text}'")

        false_positive_rate = false_positives / total_tests
        print(
            f"\nFalse positive rate: {false_positive_rate:.2%} ({false_positives}/{total_tests})"
        )

        return {
            "false_positives": false_positives,
            "total_tests": total_tests,
            "rate": false_positive_rate,
        }

    def test_confidence_scoring(self) -> Dict[str, Any]:
        """Test confidence score reliability"""
        print("Testing confidence scoring accuracy...")

        # Test commands with expected high confidence
        high_confidence_tests = ["open browser", "take screenshot", "volume up"]

        # Test commands with expected lower confidence (edge cases)
        medium_confidence_tests = [
            "could you possibly open the browser please",
            "maybe take a screenshot if you can",
            "turn the volume up a bit",
        ]

        confidence_results = {"high_confidence": [], "medium_confidence": []}

        print("\n  High confidence commands:")
        for text in high_confidence_tests:
            is_command, command_type, metadata = self.bridge.process_command(text)
            confidence = metadata.get("confidence", 0)

            print(f"    '{text}' -> conf: {confidence:.2f}")
            confidence_results["high_confidence"].append(confidence)

            self.results["confidence_scores"].append(
                {"text": text, "confidence": confidence, "expected": "high"}
            )

        print("\n  Medium confidence commands:")
        for text in medium_confidence_tests:
            is_command, command_type, metadata = self.bridge.process_command(text)
            confidence = metadata.get("confidence", 0)

            print(f"    '{text}' -> conf: {confidence:.2f}")
            confidence_results["medium_confidence"].append(confidence)

            self.results["confidence_scores"].append(
                {"text": text, "confidence": confidence, "expected": "medium"}
            )

        # Calculate averages
        avg_high = sum(confidence_results["high_confidence"]) / len(
            confidence_results["high_confidence"]
        )
        avg_medium = sum(confidence_results["medium_confidence"]) / len(
            confidence_results["medium_confidence"]
        )

        print(f"\nAverage high confidence: {avg_high:.3f}")
        print(f"Average medium confidence: {avg_medium:.3f}")
        print(
            f"Confidence differentiation: {'Good' if avg_high > avg_medium else 'Needs improvement'}"
        )

        return {
            "high_avg": avg_high,
            "medium_avg": avg_medium,
            "differentiation": avg_high > avg_medium,
        }

    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and boundary conditions"""
        print("Testing edge cases...")

        edge_cases = [
            # Very short commands
            "browser",
            "screenshot",
            "volume",
            # Commands with noise/fillers
            "uh, open browser please",
            "um, take a screenshot now",
            "well, volume up I guess",
            # Mixed case and punctuation
            "OPEN BROWSER!",
            "Take Screenshot?",
            "volume up...",
            # Similar but different
            "open the door",  # should not match browser
            "take a break",  # should not match screenshot
            "volume of water",  # should not match volume control
        ]

        edge_results = []

        for text in edge_cases:
            is_command, command_type, metadata = self.bridge.process_command(text)
            confidence = metadata.get("confidence", 0)
            method = metadata.get("method", "none")

            print(
                f"    '{text}' -> {is_command} ({command_type}, {confidence:.2f}, {method})"
            )

            edge_results.append(
                {
                    "text": text,
                    "recognized": is_command,
                    "command_type": command_type,
                    "confidence": confidence,
                    "method": method,
                }
            )

        return edge_results

    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete accuracy test suite"""
        print("VocalFlow Accuracy Test Suite")
        print("=" * 40)

        # Run all accuracy tests
        variation_results = self.test_command_variations()
        print()

        false_positive_results = self.test_false_positives()
        print()

        confidence_results = self.test_confidence_scoring()
        print()

        edge_results = self.test_edge_cases()
        print()

        # Generate accuracy summary
        self.generate_accuracy_report(
            variation_results, false_positive_results, confidence_results
        )

        return {
            "variations": variation_results,
            "false_positives": false_positive_results,
            "confidence": confidence_results,
            "edge_cases": edge_results,
            "detailed_results": self.results,
        }

    def generate_accuracy_report(self, variations, false_positives, confidence):
        """Generate accuracy summary report"""
        print("Accuracy Test Summary")
        print("=" * 25)

        # Calculate overall metrics
        total_commands_tested = sum(cat["total"] for cat in variations.values())
        total_commands_correct = sum(cat["correct"] for cat in variations.values())
        overall_accuracy = total_commands_correct / total_commands_tested

        print(f"Command Recognition Accuracy: {overall_accuracy:.1%}")
        print(f"False Positive Rate: {false_positives['rate']:.1%}")
        print(
            f"Confidence Scoring: {'Reliable' if confidence['differentiation'] else 'Needs Work'}"
        )

        # Quality assessment
        if overall_accuracy >= 0.9 and false_positives["rate"] <= 0.1:
            quality = "Excellent"
        elif overall_accuracy >= 0.8 and false_positives["rate"] <= 0.2:
            quality = "Good"
        elif overall_accuracy >= 0.7 and false_positives["rate"] <= 0.3:
            quality = "Acceptable"
        else:
            quality = "Needs Improvement"

        print(f"Overall Quality Rating: {quality}")

        # Recommendations
        print("\nRecommendations:")
        if overall_accuracy < 0.8:
            print("- Consider expanding command variations database")
        if false_positives["rate"] > 0.2:
            print("- Review confidence thresholds to reduce false positives")
        if not confidence["differentiation"]:
            print("- Improve confidence scoring algorithm")

        print("\nAccuracy testing complete!")


def main():
    """Run accuracy tests"""
    try:
        test_runner = AccuracyTestRunner()
        results = test_runner.run_all_tests()

        # Save results for analysis
        import json

        with open("accuracy_test_results.json", "w") as f:
            # Convert any non-serializable objects to strings
            serializable_results = json.loads(json.dumps(results, default=str))
            json.dump(serializable_results, f, indent=2)

        print(f"\nDetailed results saved to: accuracy_test_results.json")
        return 0

    except Exception as e:
        print(f"Accuracy test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
