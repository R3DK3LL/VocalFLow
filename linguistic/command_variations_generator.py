#!/usr/bin/env python3
"""
Command Variations Generator for VocalFlow
Generates comprehensive command variations across English dialects
Outputs to YAML for dynamic regex pattern updates
"""

import yaml
import re
from typing import Dict, List
from pathlib import Path


class CommandVariationsGenerator:
    """Generates command variations across regional dialects and speech patterns"""

    def __init__(self):
        self.variations = {
            "browser_operations": {
                "patterns": {
                    "open_browser": {
                        "uk": [
                            "open the browser",
                            "fire up browser",
                            "get browser going",
                            "bring browser up",
                            "start the browser",
                            "launch browser please",
                            "get on the internet",
                        ],
                        "us": [
                            "pull up browser",
                            "boot up browser",
                            "get browser running",
                            "fire up the browser",
                            "bring up browser",
                            "start browser up",
                            "get on the web",
                        ],
                        "aus_nz": [
                            "chuck open the browser",
                            "bring up browser",
                            "get on the browser",
                            "fire up the browser mate",
                            "get browser happening",
                            "open up browser",
                        ],
                        "casual": [
                            "browser",
                            "firefox",
                            "chrome",
                            "internet",
                            "web",
                            "online",
                        ],
                        "formal": [
                            "please open browser",
                            "launch web browser",
                            "initiate browser application",
                        ],
                        "colloquial": [
                            "get me online",
                            "hop on the internet",
                            "web time",
                            "browsing time",
                        ],
                    },
                    "close_browser": {
                        "uk": [
                            "close the browser",
                            "shut browser",
                            "close browser down",
                            "finish with browser",
                        ],
                        "us": [
                            "shut down browser",
                            "close browser out",
                            "kill the browser",
                            "end browser",
                        ],
                        "aus_nz": [
                            "shut browser down",
                            "close browser off",
                            "finish browser",
                        ],
                        "casual": ["close browser", "quit browser", "browser off"],
                        "formal": [
                            "terminate browser session",
                            "close browser application",
                        ],
                    },
                },
                "regex_base": r"(open|launch|start|fire|pull|bring|chuck|get).*browser|browser|firefox|chrome",
            },
            "file_operations": {
                "patterns": {
                    "save_file": {
                        "uk": [
                            "save this file",
                            "file save",
                            "put this away",
                            "save this document",
                            "store this file",
                            "keep this file",
                            "file this away",
                        ],
                        "us": [
                            "save it up",
                            "file this away",
                            "store this",
                            "put this in a file",
                            "save this out",
                            "file it",
                            "keep this",
                        ],
                        "aus_nz": [
                            "save it",
                            "chuck it in a file",
                            "file this",
                            "put this away",
                            "save this thing",
                            "keep this",
                        ],
                        "casual": ["save", "ctrl s", "file save", "keep it"],
                        "formal": [
                            "save current document",
                            "store this file",
                            "preserve this document",
                        ],
                        "urgent": [
                            "save now",
                            "quick save",
                            "save this quick",
                            "emergency save",
                        ],
                    },
                    "open_file": {
                        "uk": [
                            "open the file",
                            "bring up file",
                            "get file open",
                            "load this file",
                        ],
                        "us": [
                            "pull up file",
                            "open file up",
                            "load file",
                            "get this file",
                        ],
                        "aus_nz": [
                            "chuck open the file",
                            "get file going",
                            "open this file up",
                        ],
                        "casual": ["open file", "load", "file open"],
                        "formal": ["access file", "retrieve file", "load document"],
                    },
                },
                "regex_base": r"(save|store|keep|file|put).*(?:file|document|this)|ctrl\s*s",
            },
            "text_operations": {
                "patterns": {
                    "select_all": {
                        "uk": [
                            "select the lot",
                            "highlight everything",
                            "select all text",
                            "choose everything",
                            "mark all text",
                            "select the whole thing",
                        ],
                        "us": [
                            "select everything",
                            "highlight all this stuff",
                            "mark all",
                            "choose all this",
                            "select all this text",
                            "grab everything",
                        ],
                        "aus_nz": [
                            "grab everything",
                            "select the whole lot",
                            "highlight all this",
                            "mark the lot",
                            "select all this stuff",
                        ],
                        "casual": ["ctrl a", "select all", "all of it", "everything"],
                        "formal": [
                            "select entire document",
                            "highlight all content",
                            "mark complete text",
                        ],
                    },
                    "copy_text": {
                        "uk": [
                            "copy all of it",
                            "copy this text",
                            "copy the lot",
                            "duplicate this",
                        ],
                        "us": [
                            "copy all this stuff",
                            "copy everything",
                            "duplicate all this",
                        ],
                        "aus_nz": [
                            "copy all this",
                            "copy the whole thing",
                            "duplicate this lot",
                        ],
                        "casual": ["copy", "ctrl c", "copy all", "copy this"],
                        "formal": ["duplicate selected text", "copy current selection"],
                    },
                    "paste_text": {
                        "uk": [
                            "paste it here",
                            "put it here",
                            "paste this text",
                            "stick it here",
                        ],
                        "us": [
                            "paste it",
                            "put this here",
                            "paste all this",
                            "drop it here",
                        ],
                        "aus_nz": ["chuck it here", "paste this here", "put this down"],
                        "casual": ["paste", "ctrl v", "paste here"],
                        "formal": ["insert copied text", "apply clipboard content"],
                    },
                },
                "regex_base": r"(select|copy|paste|highlight|mark|grab|choose).*(?:all|everything|text|this|here)",
            },
            "system_operations": {
                "patterns": {
                    "volume_control": {
                        "uk": [
                            "turn volume up",
                            "make it louder",
                            "volume higher",
                            "increase sound",
                            "pump up volume",
                            "crank it up",
                            "louder please",
                        ],
                        "us": [
                            "volume up",
                            "turn it up",
                            "make it loud",
                            "pump the volume",
                            "crank up sound",
                            "boost volume",
                            "louder",
                        ],
                        "aus_nz": [
                            "turn it up mate",
                            "crank the volume",
                            "make it louder",
                            "pump it up",
                            "volume higher",
                        ],
                        "casual": ["louder", "volume up", "turn up", "up"],
                        "formal": ["increase audio level", "raise volume setting"],
                    },
                    "screenshot": {
                        "uk": [
                            "take a screenshot",
                            "capture screen",
                            "screen grab",
                            "take picture of screen",
                        ],
                        "us": [
                            "screenshot this",
                            "capture the screen",
                            "grab screen",
                            "snap the screen",
                        ],
                        "aus_nz": [
                            "grab a screenshot",
                            "snap the screen",
                            "capture this screen",
                        ],
                        "casual": ["screenshot", "screen cap", "capture", "snap"],
                        "formal": ["capture screen image", "take screen photograph"],
                    },
                },
                "regex_base": r"(volume|sound|audio).*(?:up|down|higher|lower)|screenshot|capture|snap.*screen",
            },
            "navigation_operations": {
                "patterns": {
                    "scroll_operations": {
                        "uk": [
                            "scroll down",
                            "move down page",
                            "go down",
                            "page down",
                            "scroll lower",
                        ],
                        "us": ["scroll down", "page down", "move down", "go down page"],
                        "aus_nz": ["scroll down", "go down", "move down"],
                        "casual": ["down", "scroll", "page down"],
                        "formal": ["navigate downward", "advance page content"],
                    },
                    "go_back": {
                        "uk": ["go back", "back page", "previous page", "go backwards"],
                        "us": ["go back", "back up", "previous", "back page"],
                        "aus_nz": ["go back", "back up", "previous"],
                        "casual": ["back", "previous"],
                        "formal": ["return to previous page", "navigate backward"],
                    },
                },
                "regex_base": r"(scroll|page|go|move).*(?:up|down|back|forward)|back|previous|next",
            },
        }

        # Common speech patterns and fillers to handle
        self.speech_patterns = {
            "polite_prefixes": ["please", "could you", "can you", "would you"],
            "casual_prefixes": ["just", "quickly", "real quick", "go ahead and"],
            "uk_patterns": ["rather", "quite", "a bit", "proper", "right then"],
            "us_patterns": ["real quick", "go ahead", "kinda", "sorta", "like"],
            "aus_nz_patterns": ["mate", "chuck", "heaps", "fair dinkum", "no worries"],
            "fillers": ["um", "uh", "er", "like", "you know", "sort of", "kind of"],
        }

    def generate_comprehensive_patterns(self) -> Dict:
        """Generate comprehensive command patterns with regex"""
        output = {
            "command_categories": {},
            "regex_patterns": {},
            "speech_variations": self.speech_patterns,
            "meta": {
                "generated_patterns": 0,
                "total_variations": 0,
                "supported_regions": ["uk", "us", "aus_nz", "casual", "formal"],
                "update_frequency": "weekly_or_as_needed",
            },
        }

        total_patterns = 0
        total_variations = 0

        for category, data in self.variations.items():
            category_patterns = {}

            for command_type, variations in data["patterns"].items():
                all_variations = []

                # Collect all regional variations
                for region, phrases in variations.items():
                    all_variations.extend(phrases)
                    total_variations += len(phrases)

                # Generate permutations with speech patterns
                enhanced_variations = self._add_speech_pattern_variations(
                    all_variations
                )

                category_patterns[command_type] = {
                    "variations": list(set(enhanced_variations)),  # Remove duplicates
                    "base_regex": self._generate_regex_pattern(enhanced_variations),
                    "priority": self._assign_priority(command_type),
                    "confidence_threshold": self._get_confidence_threshold(
                        command_type
                    ),
                }

                total_patterns += 1

            output["command_categories"][category] = category_patterns
            output["regex_patterns"][category] = data["regex_base"]

        output["meta"]["generated_patterns"] = total_patterns
        output["meta"]["total_variations"] = total_variations

        return output

    def _add_speech_pattern_variations(self, base_variations: List[str]) -> List[str]:
        """Add common speech pattern variations to base commands"""
        enhanced = list(base_variations)  # Start with originals

        for variation in base_variations[:10]:  # Limit to prevent explosion
            # Add polite prefixes
            for prefix in self.speech_patterns["polite_prefixes"][:2]:
                enhanced.append(f"{prefix} {variation}")

            # Add casual prefixes
            for prefix in self.speech_patterns["casual_prefixes"][:2]:
                enhanced.append(f"{prefix} {variation}")

            # Add with common fillers removed (simulate natural speech)
            cleaned = variation
            for filler in self.speech_patterns["fillers"]:
                cleaned = re.sub(rf"\b{filler}\b", "", cleaned).strip()
            if cleaned and cleaned != variation:
                enhanced.append(cleaned)

        return enhanced

    def _generate_regex_pattern(self, variations: List[str]) -> str:
        """Generate regex pattern from variations"""
        if not variations:
            return ""

        # Extract key terms and create flexible pattern
        key_terms = set()
        for variation in variations:
            words = re.findall(r"\b\w+\b", variation.lower())
            key_terms.update(words[:3])  # Take first 3 words as key terms

        # Remove common words
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
        }
        key_terms = key_terms - common_words

        if key_terms:
            pattern = f"({'|'.join(sorted(key_terms))})"
            return pattern

        return ""

    def _assign_priority(self, command_type: str) -> str:
        """Assign priority level to command types"""
        high_priority = ["save_file", "open_browser", "select_all", "copy_text"]
        medium_priority = ["volume_control", "screenshot", "scroll_operations"]

        if command_type in high_priority:
            return "high"
        elif command_type in medium_priority:
            return "medium"
        else:
            return "low"

    def _get_confidence_threshold(self, command_type: str) -> float:
        """Get confidence threshold for command matching"""
        critical_commands = ["save_file", "close_browser"]
        if command_type in critical_commands:
            return 0.8  # Higher threshold for critical commands
        return 0.6  # Standard threshold

    def export_to_yaml(self, output_file: str = "command_variations.yaml"):
        """Export variations to YAML file"""
        patterns = self.generate_comprehensive_patterns()

        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                patterns,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        print(f"Generated {patterns['meta']['total_variations']} command variations")
        print(f"Created {patterns['meta']['generated_patterns']} pattern categories")
        print(f"Exported to: {output_path.absolute()}")

        return output_path

    def generate_agent_integration_code(self) -> str:
        """Generate Python code for agents.py integration"""
        code = '''
# Auto-generated command variations integration for agents.py
import yaml
import re
from typing import Dict, List, Optional

class EnhancedCommandMatcher:
    """Enhanced command matching with regional variations support"""

    def __init__(self, yaml_file: str = "command_variations.yaml"):
        with open(yaml_file, 'r') as f:
            self.patterns = yaml.safe_load(f)
        self.compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict:
        """Compile regex patterns for faster matching"""
        compiled = {}
        for category, commands in self.patterns["command_categories"].items():
            compiled[category] = {}
            for command_type, data in commands.items():
                if data["base_regex"]:
                    compiled[category][command_type] = re.compile(
                        data["base_regex"],
                        re.IGNORECASE | re.VERBOSE
                    )
        return compiled

    def match_command(self, text: str) -> Optional[Dict]:
        """Match text against command patterns"""
        text = self._preprocess_text(text)

        for category, commands in self.compiled_patterns.items():
            for command_type, pattern in commands.items():
                if pattern.search(text):
                    confidence = self._calculate_confidence(text, command_type, category)
                    threshold = self.patterns["command_categories"][category][command_type]["confidence_threshold"]

                    if confidence >= threshold:
                        return {
                            "category": category,
                            "command": command_type,
                            "confidence": confidence,
                            "matched_text": text
                        }

        return None

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for matching"""
        # Remove common fillers
        for filler in self.patterns["speech_variations"]["fillers"]:
            text = re.sub(rf'\\b{filler}\\b', '', text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r'\\s+', ' ', text).strip()

        return text.lower()

    def _calculate_confidence(self, text: str, command_type: str, category: str) -> float:
        """Calculate matching confidence score"""
        variations = self.patterns["command_categories"][category][command_type]["variations"]

        # Simple word overlap confidence
        text_words = set(text.split())
        max_confidence = 0.0

        for variation in variations:
            var_words = set(variation.lower().split())
            if var_words:
                overlap = len(text_words & var_words) / len(var_words)
                max_confidence = max(max_confidence, overlap)

        return max_confidence

# Usage in agents.py:
# matcher = EnhancedCommandMatcher()
# result = matcher.match_command("please open the browser")
# if result:
#     execute_command(result["category"], result["command"])
'''
        return code


def main():
    """Main execution function"""
    generator = CommandVariationsGenerator()

    # Generate and export YAML
    generator.export_to_yaml()

    # Generate integration code
    integration_code = generator.generate_agent_integration_code()

    # Save integration code
    with open("enhanced_command_matcher.py", "w") as f:
        f.write(integration_code)

    print(f"\\nIntegration code saved to: enhanced_command_matcher.py")
    print(f"\\nTo update agents.py, import and use EnhancedCommandMatcher class")


if __name__ == "__main__":
    main()
