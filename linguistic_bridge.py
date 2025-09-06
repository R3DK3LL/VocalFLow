#!/usr/bin/env python3
"""
Linguistic Integration Bridge for VocalFlow - Fixed Adaptive Learning
Connects enhanced command matching with existing agent system.
"""

import os
import sys
import yaml
import re
from typing import Dict, List, Optional, Tuple, Any, Set

# Add linguistic directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "linguistic"))

try:
    from linguistic.enhanced_command_matcher import EnhancedCommandMatcher
except ImportError:
    print(
        "Warning: Enhanced command matcher not available, falling back to basic matching"
    )
    EnhancedCommandMatcher = None


class LinguisticBridge:
    """
    Bridge between enhanced linguistic processing and existing agent system.
    Fixed adaptive learning that only activates on complete failures.
    """

    def __init__(self, yaml_path: str = None, enable_enhanced: bool = True):
        self.enable_enhanced = enable_enhanced and EnhancedCommandMatcher is not None
        self.enhanced_matcher = None
        self.fallback_patterns = self._build_fallback_patterns()

        if self.enable_enhanced:
            try:
                yaml_file = yaml_path or os.path.join(
                    "linguistic", "command_variations.yaml"
                )
                self.enhanced_matcher = EnhancedCommandMatcher(yaml_file)
                print("✅ Enhanced linguistic matching enabled")
            except Exception as e:
                print(f"⚠️  Enhanced matching failed to load: {e}")
                print("📋 Falling back to basic pattern matching")
                self.enable_enhanced = False

    def _build_fallback_patterns(self) -> Dict[str, List[str]]:
        """Basic fallback patterns for core commands"""
        return {
            "browser_operations": [
                r"open\s+browser",
                r"launch\s+browser",
                r"start\s+browser",
                r"fire\s+up\s+browser",
                r"bring\s+up\s+browser",
            ],
            "volume_control": [
                r"volume\s+up",
                r"volume\s+down",
                r"mute",
                r"unmute",
                r"increase\s+volume",
                r"decrease\s+volume",
            ],
            "screenshot": [
                r"take\s+screenshot",
                r"capture\s+screen",
                r"screenshot",
                r"snap\s+screen",
                r"grab\s+screenshot",
            ],
            "terminal_operations": [
                r"open\s+terminal",
                r"launch\s+terminal",
                r"start\s+terminal",
                r"bring\s+up\s+terminal",
                r"fire\s+up\s+terminal",
            ],
        }

    def process_command(self, text: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Process voice command with enhanced or fallback matching.
        Enhanced with adaptive learning for truly failed commands only.
        """
        text_clean = text.lower().strip()

        # Try enhanced matching first
        if self.enable_enhanced:
            try:
                result = self._enhanced_process(text_clean)
                if result[0]:  # Command found
                    return result
            except Exception as e:
                print(f"Enhanced matching error: {e}, falling back...")

        # Fallback to basic pattern matching
        fallback_result = self._fallback_process(text_clean)
        if fallback_result[0]:  # Command found in fallback
            return fallback_result

        # ONLY check for learning if BOTH systems completely failed
        learning_suggestion = self.analyze_failed_command(text_clean)
        if learning_suggestion:
            print(
                f"💡 Learning opportunity: '{text}' might be '{learning_suggestion['suggested_category']}' (confidence: {learning_suggestion['confidence']:.2f})"
            )
            print(f"   Matching words: {learning_suggestion['matching_words']}")

        return False, "", {}

    def _enhanced_process(self, text: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Process using enhanced linguistic matcher"""
        if not self.enhanced_matcher:
            return False, "", {}

        # Use enhanced matcher's match_command method
        match_result = self.enhanced_matcher.match_command(text)

        if match_result and match_result.get("confidence", 0) > 0.7:
            return (
                True,
                match_result.get("command", ""),
                {
                    "confidence": match_result.get("confidence", 0),
                    "variation": match_result.get("matched_text", ""),
                    "category": match_result.get("category", ""),
                    "method": "enhanced",
                },
            )

        return False, "", {}

    def _fallback_process(self, text: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Fallback pattern matching"""
        for category, patterns in self.fallback_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return (
                        True,
                        category,
                        {"confidence": 0.8, "pattern": pattern, "method": "fallback"},
                    )

        return False, "", {}

    def analyze_failed_command(self, text: str, alpha: float = 0.4) -> Optional[Dict]:
        """
        Analyze failed commands - ONLY suggest categories that exist in fallback system
        """
        if not self.enable_enhanced or not self.enhanced_matcher:
            return None

        words = set(text.lower().split())
        n = len(words)

        if n < 2:
            return None

        # ONLY use the 4 valid fallback categories
        valid_categories = set(
            self.fallback_patterns.keys()
        )  # browser_operations,  volume_control, screenshot, terminal_operations

        yaml_categories = self._get_yaml_categories()
        if not yaml_categories:
            return None

        best_suggestion = None
        best_score = 0

        for category, variations in yaml_categories.items():
            # Skip invalid categories
            if category not in valid_categories:
                continue

            category_words = self._extract_yaml_keywords(variations)
            overlap = words.intersection(category_words)
            threshold = max(1, int(alpha * n))

            if len(overlap) >= threshold:
                confidence = len(overlap) / n

                if confidence > best_score:
                    best_score = confidence
                    best_suggestion = {
                        "suggested_category": category,
                        "confidence": confidence,
                        "matching_words": list(overlap),
                        "original_text": text,
                    }

        # Higher threshold to avoid random suggestions
        return best_suggestion if best_score > 0.4 else None

    def _get_yaml_categories(self) -> Dict[str, List[str]]:
        """Extract categories and variations from YAML file"""
        try:
            yaml_file = os.path.join("linguistic", "command_variations.yaml")
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)

            categories = {}
            for category, commands in data.get("command_categories", {}).items():
                variations = []
                for command_data in commands.values():
                    if isinstance(command_data, dict) and "variations" in command_data:
                        variations.extend(command_data["variations"])
                categories[category] = variations

            return categories
        except Exception:
            return {}

    def _extract_yaml_keywords(self, variations: List[str]) -> Set[str]:
        """Extract keywords from YAML variations (not regex patterns)"""
        stopwords = {
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
            "of",
            "with",
            "by",
        }
        keywords = set()

        for variation in variations:
            words = variation.lower().split()
            keywords.update(
                word for word in words if word not in stopwords and len(word) > 2
            )

        return keywords

    def get_command_variations(self, command_type: str) -> List[str]:
        """Get all variations for a specific command type"""
        if self.enable_enhanced and self.enhanced_matcher:
            try:
                return self.enhanced_matcher.get_variations_for_command(command_type)
            except Exception as e:
                pass

        # Fallback variations
        return self.fallback_patterns.get(command_type, [])

    def add_custom_pattern(self, category: str, pattern: str):
        """Add custom pattern to fallback system"""
        if category not in self.fallback_patterns:
            self.fallback_patterns[category] = []
        self.fallback_patterns[category].append(pattern)
        print(f"➕ Added custom pattern '{pattern}' to category '{category}'")

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learning opportunities and patterns"""
        yaml_categories = self._get_yaml_categories()
        yaml_stats = {}

        if yaml_categories:
            total_variations = sum(
                len(variations) for variations in yaml_categories.values()
            )
            total_keywords = sum(
                len(self._extract_yaml_keywords(variations))
                for variations in yaml_categories.values()
            )
            yaml_stats = {
                "yaml_categories": len(yaml_categories),
                "total_yaml_variations": total_variations,
                "yaml_keywords": total_keywords,
            }

        return {
            "total_fallback_categories": len(self.fallback_patterns),
            "total_fallback_patterns": sum(
                len(patterns) for patterns in self.fallback_patterns.values()
            ),
            "categories": list(self.fallback_patterns.keys()),
            **yaml_stats,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get integration status for debugging"""
        base_status = {
            "enhanced_enabled": self.enable_enhanced,
            "enhanced_loaded": self.enhanced_matcher is not None,
            "fallback_patterns": len(self.fallback_patterns),
            "total_categories": len(self.fallback_patterns),
        }

        # Add learning stats
        base_status.update(self.get_learning_stats())

        return base_status


# Integration helper for agents.py
class AgentLinguisticIntegrator:
    """
    Helper class to integrate linguistic bridge with existing agent methods
    """

    def __init__(self, bridge: LinguisticBridge = None):
        self.bridge = bridge or LinguisticBridge()

    def enhance_process_text(self, original_process_func, text: str) -> bool:
        """
        Enhanced version of agents.py process_text method

        Args:
            original_process_func: The existing process_text method
            text: Input text to process
        """
        # Try linguistic processing first
        is_command, command_type, metadata = self.bridge.process_command(text)

        if is_command:
            print(
                f"🎯 Linguistic match: {command_type} ({metadata.get('method', 'unknown')})"
            )

            # Map linguistic command types to existing agent actions
            success = self._execute_mapped_command(command_type, text, metadata)
            if success:
                return True

        # Fall back to original processing
        return original_process_func(text)

    def _execute_mapped_command(
        self, command_type: str, original_text: str, metadata: Dict
    ) -> bool:
        """Map linguistic command types to agent actions"""
        command_mapping = {
            "browser_operations": "open_browser",
            "open_browser": "open_browser",
            "volume_control": "volume_control",
            "volume_up": "volume_up",
            "volume_down": "volume_down",
            "screenshot": "take_screenshot",
            "terminal_operations": "open_terminal",
        }

        mapped_command = command_mapping.get(command_type)
        if mapped_command:
            print(f"📋 Executing mapped command: {mapped_command}")
            # Here you would call the actual agent method
            # This is where we'll integrate with existing agents.py methods
            return True

        return False


# Easy integration function for agents.py
def create_enhanced_processor():
    """
    Factory function to create enhanced processor for easy integration
    """
    bridge = LinguisticBridge()
    integrator = AgentLinguisticIntegrator(bridge)

    return {"bridge": bridge, "integrator": integrator, "status": bridge.get_status()}


# Test the integration with fixed adaptive learning
if __name__ == "__main__":
    # Test the bridge
    print("🔧 Testing Fixed Linguistic Integration Bridge with Adaptive Learning...")

    # Create bridge
    bridge = LinguisticBridge()
    print(f"📊 Bridge Status: {bridge.get_status()}")

    # Test successful commands (should NOT trigger learning)
    print("\n✅ Testing Known Commands (should NOT learn):")
    known_commands = [
        "open browser please",
        "take a screenshot",
        "volume up",
        "launch terminal",
    ]

    for cmd in known_commands:
        is_cmd, cmd_type, metadata = bridge.process_command(cmd)
        print(
            f"'{cmd}' -> Command: {is_cmd}, Type: {cmd_type}, Method: {metadata.get('method', 'none')}"
        )

    # Test genuine learning opportunities (should trigger learning)
    print("\n🧠 Testing Genuine Learning Opportunities (should learn):")
    learning_commands = [
        "navigate to website",  # Should suggest browser_operations
        "make screen recording",  # Should suggest screenshot
        "adjust system audio",  # Should suggest volume_control
        "command prompt access",  # Should suggest terminal_operations
        "completely random text here",  # Should not suggest anything
    ]

    for cmd in learning_commands:
        is_cmd, cmd_type, metadata = bridge.process_command(cmd)
        print(f"'{cmd}' -> Command: {is_cmd}")

    print(f"\n📈 Learning Stats: {bridge.get_learning_stats()}")
    print("\n✅ Fixed bridge with proper adaptive learning ready!")
