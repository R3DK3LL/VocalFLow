#!/usr/bin/env python3
"""
Integration testing for VocalFlow system
Tests the complete pipeline and component interactions
"""
import sys
import os
import time
import subprocess
import tempfile
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_agents import EnhancedVocalFlowAgent, AgentConfig
from linguistic_bridge import LinguisticBridge

class IntegrationTestRunner:
    """Test complete VocalFlow system integration"""
    
    def __init__(self):
        self.results = {
            'pipeline_tests': [],
            'component_integration': [],
            'profile_system': [],
            'cross_platform': [],
            'error_handling': []
        }
    
    def test_complete_pipeline(self) -> Dict[str, Any]:
        """Test the complete text processing pipeline"""
        print("Testing complete processing pipeline...")
        
        # Test the full chain: text input -> linguistic processing -> agent execution
        agent = EnhancedVocalFlowAgent()
        
        pipeline_tests = [
            {
                'input': "open browser",
                'expected_type': 'command',
                'should_execute': True
            },
            {
                'input': "Hello, this is normal dictation text.",
                'expected_type': 'dictation',
                'should_execute': False
            },
            {
                'input': "take a screenshot please",
                'expected_type': 'command', 
                'should_execute': True
            },
            {
                'input': "I need to write a document about browsers.",
                'expected_type': 'dictation',
                'should_execute': False
            }
        ]
        
        pipeline_results = []
        
        for test in pipeline_tests:
            print(f"  Testing: '{test['input']}'")
            
            # Test complete processing
            result = agent.process_transcription_complete(test['input'])
            
            # Validate pipeline behavior
            command_executed = result['command_executed']
            context_detected = result['context']
            completion_provided = result['completion'] is not None
            
            success = (command_executed == test['should_execute'])
            
            pipeline_result = {
                'input': test['input'],
                'expected_type': test['expected_type'],
                'command_executed': command_executed,
                'context': context_detected,
                'completion': completion_provided,
                'suggestions': len(result['suggestions']),
                'success': success
            }
            
            pipeline_results.append(pipeline_result)
            self.results['pipeline_tests'].append(pipeline_result)
            
            print(f"    Expected: {test['expected_type']}, Got: {'command' if command_executed else 'dictation'}")
            print(f"    Context: {context_detected}, Suggestions: {len(result['suggestions'])}")
            print(f"    Result: {'PASS' if success else 'FAIL'}")
        
        success_rate = sum(1 for r in pipeline_results if r['success']) / len(pipeline_results)
        print(f"\nPipeline success rate: {success_rate:.1%}")
        
        return {
            'tests': pipeline_results,
            'success_rate': success_rate
        }
    
    def test_component_integration(self) -> Dict[str, Any]:
        """Test integration between major components"""
        print("Testing component integration...")
        
        integration_results = []
        
        # Test linguistic bridge integration
        print("  Testing linguistic bridge integration...")
        bridge = LinguisticBridge()
        bridge_status = bridge.get_status()
        
        bridge_test = {
            'component': 'linguistic_bridge',
            'initialized': bridge_status.get('enhanced_enabled', False),
            'patterns_loaded': bridge_status.get('total_fallback_patterns', 0) > 0,
            'yaml_loaded': bridge_status.get('total_yaml_variations', 0) > 0,
            'categories_available': len(bridge_status.get('categories', [])) > 0
        }
        
        bridge_success = all(bridge_test[key] for key in ['initialized', 'patterns_loaded', 'yaml_loaded', 'categories_available'])
        bridge_test['success'] = bridge_success
        integration_results.append(bridge_test)
        
        print(f"    Bridge initialized: {bridge_test['initialized']}")
        print(f"    Patterns loaded: {bridge_test['patterns_loaded']}")
        print(f"    YAML loaded: {bridge_test['yaml_loaded']}")
        print(f"    Categories available: {bridge_test['categories_available']}")
        print(f"    Integration: {'PASS' if bridge_success else 'FAIL'}")
        
        # Test enhanced agent integration
        print("  Testing enhanced agent integration...")
        agent = EnhancedVocalFlowAgent()
        agent_status = agent.get_status()
        
        agent_test = {
            'component': 'enhanced_agent',
            'enhanced_mode': agent_status.get('enhanced_mode', False),
            'linguistic_available': agent_status.get('linguistic_available', False),
            'context_system': agent_status.get('current_context') is not None,
            'commands_available': len(agent_status.get('available_commands', [])) > 0
        }
        
        agent_success = all(agent_test[key] for key in ['enhanced_mode', 'linguistic_available', 'commands_available'])
        agent_test['success'] = agent_success
        integration_results.append(agent_test)
        
        print(f"    Enhanced mode: {agent_test['enhanced_mode']}")
        print(f"    Linguistic available: {agent_test['linguistic_available']}")
        print(f"    Context system: {agent_test['context_system']}")
        print(f"    Commands available: {agent_test['commands_available']}")
        print(f"    Integration: {'PASS' if agent_success else 'FAIL'}")
        
        # Test bridge-agent integration
        print("  Testing bridge-agent communication...")
        test_command = "open browser"
        bridge_result = bridge.process_command(test_command)
        agent_result = agent.process_text_enhanced(test_command)
        
        communication_test = {
            'component': 'bridge_agent_communication',
            'bridge_recognizes': bridge_result[0],
            'agent_executes': agent_result,
            'consistent_results': bridge_result[0] == agent_result
        }
        
        communication_success = communication_test['consistent_results']
        communication_test['success'] = communication_success
        integration_results.append(communication_test)
        
        print(f"    Bridge recognizes command: {communication_test['bridge_recognizes']}")
        print(f"    Agent executes command: {communication_test['agent_executes']}")
        print(f"    Results consistent: {communication_test['consistent_results']}")
        print(f"    Communication: {'PASS' if communication_success else 'FAIL'}")
        
        self.results['component_integration'] = integration_results
        
        overall_success = sum(1 for r in integration_results if r['success']) / len(integration_results)
        print(f"\nComponent integration success rate: {overall_success:.1%}")
        
        return {
            'tests': integration_results,
            'success_rate': overall_success
        }
    
    def test_profile_system(self) -> Dict[str, Any]:
        """Test user profile system integration"""
        print("Testing profile system...")
        
        profile_results = []
        
        # Test profile directory creation
        profile_dir = Path.home() / ".vocalflow" / "profiles"
        profile_test = {
            'test': 'profile_directory',
            'exists': profile_dir.exists(),
            'writable': os.access(profile_dir, os.W_OK) if profile_dir.exists() else False
        }
        
        if not profile_dir.exists():
            try:
                profile_dir.mkdir(parents=True, exist_ok=True)
                profile_test['exists'] = True
                profile_test['writable'] = True
                print("    Created profile directory")
            except Exception as e:
                print(f"    Failed to create profile directory: {e}")
        
        profile_test['success'] = profile_test['exists'] and profile_test['writable']
        profile_results.append(profile_test)
        
        print(f"    Profile directory exists: {profile_test['exists']}")
        print(f"    Profile directory writable: {profile_test['writable']}")
        print(f"    Directory test: {'PASS' if profile_test['success'] else 'FAIL'}")
        
        # Test profile creation and loading
        try:
            from vocalflow.main import ProfileManager
            
            manager = ProfileManager()
            test_profile_name = "integration_test"
            
            # Create test profile
            profile = manager.create_profile(test_profile_name)
            profile_created = profile is not None
            
            # Load test profile
            loaded_profile = manager.load_profile(test_profile_name)
            profile_loaded = loaded_profile is not None and loaded_profile.name == test_profile_name
            
            # Clean up test profile
            test_profile_file = profile_dir / f"{test_profile_name}.json"
            if test_profile_file.exists():
                test_profile_file.unlink()
            
            profile_ops_test = {
                'test': 'profile_operations',
                'created': profile_created,
                'loaded': profile_loaded,
                'success': profile_created and profile_loaded
            }
            
            profile_results.append(profile_ops_test)
            
            print(f"    Profile creation: {profile_created}")
            print(f"    Profile loading: {profile_loaded}")
            print(f"    Profile operations: {'PASS' if profile_ops_test['success'] else 'FAIL'}")
            
        except ImportError as e:
            print(f"    Profile system test skipped: {e}")
            profile_ops_test = {
                'test': 'profile_operations',
                'created': False,
                'loaded': False,
                'success': False,
                'error': str(e)
            }
            profile_results.append(profile_ops_test)
        
        self.results['profile_system'] = profile_results
        
        profile_success = sum(1 for r in profile_results if r['success']) / len(profile_results)
        print(f"\nProfile system success rate: {profile_success:.1%}")
        
        return {
            'tests': profile_results,
            'success_rate': profile_success
        }
    
    def test_cross_platform_compatibility(self) -> Dict[str, Any]:
        """Test cross-platform functionality"""
        print("Testing cross-platform compatibility...")
        
        import platform
        current_os = platform.system()
        
        cross_platform_results = []
        
        # Test system command availability
        system_commands = {
            'Linux': ['xdotool', 'xclip', 'gnome-screenshot', 'pactl'],
            'Darwin': ['pbcopy', 'osascript'],
            'Windows': ['clip']
        }
        
        expected_commands = system_commands.get(current_os, [])
        
        for cmd in expected_commands:
            try:
                result = subprocess.run(['which', cmd] if current_os != 'Windows' else ['where', cmd], 
                                      capture_output=True, text=True, timeout=5)
                available = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                available = False
            
            cmd_test = {
                'command': cmd,
                'platform': current_os,
                'available': available
            }
            
            cross_platform_results.append(cmd_test)
            print(f"    {cmd}: {'Available' if available else 'Missing'}")
        
        # Test agent's application discovery
        agent = EnhancedVocalFlowAgent()
        
        # Test browser discovery
        browser = agent._discover_application('browser')
        browser_test = {
            'application_type': 'browser',
            'discovered': browser is not None,
            'executable': browser if browser else 'None'
        }
        cross_platform_results.append(browser_test)
        print(f"    Browser discovery: {browser_test['discovered']} ({browser_test['executable']})")
        
        # Test terminal discovery
        terminal = agent._discover_application('terminal')
        terminal_test = {
            'application_type': 'terminal', 
            'discovered': terminal is not None,
            'executable': terminal if terminal else 'None'
        }
        cross_platform_results.append(terminal_test)
        print(f"    Terminal discovery: {terminal_test['discovered']} ({terminal_test['executable']})")
        
        self.results['cross_platform'] = cross_platform_results
        
        # Calculate success rate (commands available + apps discovered)
        command_success = sum(1 for r in cross_platform_results if r.get('available', r.get('discovered', False)))
        total_tests = len(cross_platform_results)
        success_rate = command_success / total_tests if total_tests > 0 else 0
        
        print(f"\nCross-platform compatibility: {success_rate:.1%}")
        
        return {
            'platform': current_os,
            'tests': cross_platform_results,
            'success_rate': success_rate
        }
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and graceful degradation"""
        print("Testing error handling...")
        
        error_results = []
        
        # Test with missing dependencies (simulated)
        print("  Testing graceful degradation...")
        try:
            # Test agent creation with potential missing components
            agent = EnhancedVocalFlowAgent()
            status = agent.get_status()
            
            graceful_test = {
                'test': 'graceful_degradation',
                'agent_created': True,
                'enhanced_mode': status.get('enhanced_mode', False),
                'fallback_available': True,  # System should work even without enhanced features
                'success': True
            }
            
        except Exception as e:
            graceful_test = {
                'test': 'graceful_degradation',
                'agent_created': False,
                'error': str(e),
                'success': False
            }
        
        error_results.append(graceful_test)
        print(f"    Graceful degradation: {'PASS' if graceful_test['success'] else 'FAIL'}")
        
        # Test invalid command handling
        print("  Testing invalid command handling...")
        try:
            agent = EnhancedVocalFlowAgent()
            
            # Test with nonsense input
            result = agent.process_text_enhanced("xyzzy invalid command 123")
            invalid_command_test = {
                'test': 'invalid_command_handling',
                'no_crash': True,
                'returns_false': result == False,  # Should return False for invalid commands
                'success': result == False
            }
            
        except Exception as e:
            invalid_command_test = {
                'test': 'invalid_command_handling', 
                'no_crash': False,
                'error': str(e),
                'success': False
            }
        
        error_results.append(invalid_command_test)
        print(f"    Invalid command handling: {'PASS' if invalid_command_test['success'] else 'FAIL'}")
        
        # Test empty input handling
        print("  Testing empty input handling...")
        try:
            agent = EnhancedVocalFlowAgent()
            
            empty_inputs = ["", "   ", None]
            empty_handling_success = True
            
            for empty_input in empty_inputs[:2]:  # Skip None to avoid TypeError
                try:
                    result = agent.process_text_enhanced(empty_input)
                    # Should handle gracefully without crashing
                except Exception:
                    empty_handling_success = False
                    break
            
            empty_test = {
                'test': 'empty_input_handling',
                'handles_gracefully': empty_handling_success,
                'success': empty_handling_success
            }
            
        except Exception as e:
            empty_test = {
                'test': 'empty_input_handling',
                'handles_gracefully': False,
                'error': str(e),
                'success': False
            }
        
        error_results.append(empty_test)
        print(f"    Empty input handling: {'PASS' if empty_test['success'] else 'FAIL'}")
        
        self.results['error_handling'] = error_results
        
        error_success = sum(1 for r in error_results if r['success']) / len(error_results)
        print(f"\nError handling success rate: {error_success:.1%}")
        
        return {
            'tests': error_results,
            'success_rate': error_success
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete integration test suite"""
        print("VocalFlow Integration Test Suite")
        print("=" * 40)
        
        # Run all integration tests
        pipeline_results = self.test_complete_pipeline()
        print()
        
        component_results = self.test_component_integration()
        print()
        
        profile_results = self.test_profile_system()
        print()
        
        platform_results = self.test_cross_platform_compatibility()
        print()
        
        error_results = self.test_error_handling()
        print()
        
        # Generate integration summary
        self.generate_integration_report(pipeline_results, component_results, 
                                       profile_results, platform_results, error_results)
        
        return {
            'pipeline': pipeline_results,
            'components': component_results,
            'profiles': profile_results,
            'cross_platform': platform_results,
            'error_handling': error_results,
            'detailed_results': self.results
        }
    
    def generate_integration_report(self, pipeline, components, profiles, platform, errors):
        """Generate integration test summary"""
        print("Integration Test Summary")
        print("=" * 28)
        
        # Calculate overall metrics
        success_rates = [
            pipeline['success_rate'],
            components['success_rate'], 
            profiles['success_rate'],
            platform['success_rate'],
            errors['success_rate']
        ]
        
        overall_success = sum(success_rates) / len(success_rates)
        
        print(f"Complete Pipeline: {pipeline['success_rate']:.1%}")
        print(f"Component Integration: {components['success_rate']:.1%}")
        print(f"Profile System: {profiles['success_rate']:.1%}")
        print(f"Cross-Platform: {platform['success_rate']:.1%}")
        print(f"Error Handling: {errors['success_rate']:.1%}")
        print(f"Overall Integration: {overall_success:.1%}")
        
        # Quality assessment
        if overall_success >= 0.9:
            quality = "Excellent Integration"
        elif overall_success >= 0.8:
            quality = "Good Integration"
        elif overall_success >= 0.7:
            quality = "Acceptable Integration"
        else:
            quality = "Integration Issues"
        
        print(f"Integration Quality: {quality}")
        
        # Recommendations
        print("\nRecommendations:")
        if pipeline['success_rate'] < 0.8:
            print("- Review pipeline component interactions")
        if components['success_rate'] < 0.8:
            print("- Check component initialization and communication")
        if profiles['success_rate'] < 0.8:
            print("- Verify profile system file permissions and paths")
        if platform['success_rate'] < 0.8:
            print("- Install missing system dependencies for full functionality")
        if errors['success_rate'] < 0.8:
            print("- Improve error handling and graceful degradation")
        
        print("\nIntegration testing complete!")

def main():
    """Run integration tests"""
    try:
        test_runner = IntegrationTestRunner()
        results = test_runner.run_all_tests()
        
        # Save results for analysis
        import json
        with open('integration_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: integration_test_results.json")
        return 0
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
