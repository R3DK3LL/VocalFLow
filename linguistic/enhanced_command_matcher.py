# Auto-generated command variations integration for agents.py
import yaml
import re
from typing import Dict, Optional


class EnhancedCommandMatcher:
    """Enhanced command matching with regional variations support"""

    def __init__(self, yaml_file: str = "command_variations.yaml"):
        with open(yaml_file, "r") as f:
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
                        data["base_regex"], re.IGNORECASE | re.VERBOSE
                    )
        return compiled

    def match_command(self, text: str) -> Optional[Dict]:
        """Match text against command patterns"""
        text = self._preprocess_text(text)

        for category, commands in self.compiled_patterns.items():
            for command_type, pattern in commands.items():
                if pattern.search(text):
                    confidence = self._calculate_confidence(
                        text, command_type, category
                    )
                    threshold = self.patterns["command_categories"][category][
                        command_type
                    ]["confidence_threshold"]

                    if confidence >= threshold:
                        return {
                            "category": category,
                            "command": command_type,
                            "confidence": confidence,
                            "matched_text": text,
                        }

        return None

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for matching"""
        # Remove common fillers
        for filler in self.patterns["speech_variations"]["fillers"]:
            text = re.sub(rf"\b{filler}\b", "", text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text.lower()

    def _calculate_confidence(
        self, text: str, command_type: str, category: str
    ) -> float:
        """Calculate matching confidence score"""
        variations = self.patterns["command_categories"][category][command_type][
            "variations"
        ]

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
