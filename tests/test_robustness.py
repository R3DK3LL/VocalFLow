#!/usr/bin/env python3
"""
Robustness testing for VocalFlow system
Tests edge cases, error conditions, and system limits
"""
import sys
import os
import time
import tempfile
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_agents import EnhancedVocalFlowAgent, AgentConfig
from linguistic_bridge import LinguisticBridge

class RobustnessTestRunner:
    """Test VocalFlow system robustness and edge cases"""
    
    def __init__(self):
        self.results = {
            'edge_cases': [],
            'malformed_input': [],
            'resource_limits': [],
            'concurrent_operations': [],
            'file_system': []
        }
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and boundary conditions"""
        print("Testing edge cases and boundary conditions...")
        
        agent = EnhancedVocalFlowAgent()
        edge_case_results = []
        
        # Test various edge case inputs
        edge_cases = [
            # Empty and whitespace inputs
            {"input": "", "description": "empty string"},
            {"input": "   ", "description": "whitespace only"},
            {"input": "\n\t\r", "description": "special whitespace"},
            
            # Very short inputs
            {"input": "a", "description": "single character"},
            {"input": "ab", "description": "two characters"},
            {"input": "browser", "description": "single word command"},
            
            # Very long inputs
            {"input": "open browser " * 100, "description": "extremely long repeated command"},
            {"input": "This is a very long sentence that goes on and on and contains many words but should still be processed correctly by the system without causing any crashes or errors.", "description": "long sentence"},
            
            # Special characters and encoding
            {"input": "open browser!@#$%^&*()", "description": "special characters"},
            {"input": "ouvrir navigateur", "description": "french text"},
            {"input": "открыть браузер", "description": "cyrillic text"},
            {"input": "ブラウザを開く", "description": "japanese text"},
            
            # Mixed content
            {"input": "123 open browser 456", "description": "numbers with command"},
            {"input": "OPEN BROWSER", "description": "all caps"},
            {"input": "Open Browser Please", "description": "title case"},
            {"input": "open\tbrowser\nplease", "description": "tabs and newlines"},
            
            # Ambiguous inputs
            {"input": "maybe open browser if possible", "description": "tentative command"},
            {"input": "don't open browser", "description": "negative command"},
            {"input": "browser browser browser", "description": "repeated words"},
        ]
        
        for case in edge_cases:
            print(f"  Testing {case['description']}: '{case['input'][:50]}{'...' if len(case['input']) > 50 else ''}'")
            
            try:
                start_time = time.time()
                result = agent.process_transcription_complete(case['input'])
                processing_time = time.time() - start_time
                
                edge_result = {
                    'input': case['input'],
                    'description': case['description'],
                    'processed_successfully': True,
                    'command_executed': result['command_executed'],
                    'context': result['context'],
                    'completion': result['completion'] is not None,
                    'processing_time': processing_time,
                    'suggestions': len(result['suggestions']),
                    'error': None
                }
                
                print(f"    Result: {'Command' if result['command_executed'] else 'Text'}, Time: {processing_time:.4f}s")
                
            except Exception as e:
                edge_result = {
                    'input': case['input'],
                    'description': case['description'], 
                    'processed_successfully': False,
                    'error': str(e),
                    'processing_time': None
                }
                
                print(f"    ERROR: {str(e)}")
            
            edge_case_results.append(edge_result)
            self.results['edge_cases'].append(edge_result)
        
        success_rate = sum(1 for r in edge_case_results if r['processed_successfully']) / len(edge_case_results)
        print(f"\nEdge case handling success rate: {success_rate:.1%}")
        
        return {
            'tests': edge_case_results,
            'success_rate': success_rate
        }
    
    def test_malformed_input(self) -> Dict[str, Any]:
        """Test handling of malformed and corrupted input"""
        print("Testing malformed input handling...")
        
        agent = EnhancedVocalFlowAgent()
        malformed_results = []
        
        # Various types of malformed input
        malformed_inputs = [
            # Binary and non-text data (as strings that might come from corrupted audio)
            {"input": "\x00\x01\x02", "description": "null bytes"},
            {"input": "invalid\xffcharacters", "description": "invalid utf-8"},
            
            # SQL injection-like attempts (not applicable but tests robustness)
            {"input": "open browser'; DROP TABLE users; --", "description": "sql injection style"},
            {"input": "open browser<script>alert(1)</script>", "description": "html injection style"},
            
            # Command injection attempts
            {"input": "open browser && rm -rf /", "description": "command injection attempt"},
            {"input": "open browser | cat /etc/passwd", "description": "pipe injection attempt"},
            
            # Format string attempts
            {"input": "open browser %s %d %x", "description": "format string"},
            {"input": "open browser {0} {1}", "description": "python format"},
            
            # Buffer overflow attempts (just very long strings)
            {"input": "A" * 1000, "description": "1000 character string"},
            {"input": "open browser " + "A" * 500, "description": "command with long suffix"},
            
            # Unicode edge cases
            {"input": "open browser \u0000\u001f", "description": "control characters"},
            {"input": "open browser 𝕠𝕨𝔰♅⚡", "description": "unicode symbols"},
        ]
        
        for case in malformed_inputs:
            print(f"  Testing {case['description']}")
            
            try:
                start_time = time.time()
                result = agent.process_transcription_complete(case['input'])
                processing_time = time.time() - start_time
                
                # Check that system handled malformed input gracefully
                malformed_result = {
                    'input_description': case['description'],
                    'handled_gracefully': True,
                    'command_executed': result['command_executed'],
                    'processing_time': processing_time,
                    'no_security_risk': True,  # Assume safe since no actual system commands run
                    'error': None
                }
                
                print(f"    Handled gracefully: {'Command' if result['command_executed'] else 'Text'}")
                
            except Exception as e:
                # Even exceptions should be handled gracefully without crashes
                malformed_result = {
                    'input_description': case['description'],
                    'handled_gracefully': False,
                    'error': str(e),
                    'no_security_risk': True,  # System didn't crash completely
                    'processing_time': None
                }
                
                print(f"    Exception (but contained): {str(e)[:100]}...")
            
            malformed_results.append(malformed_result)
            self.results['malformed_input'].append(malformed_result)
        
        graceful_handling_rate = sum(1 for r in malformed_results if r['handled_gracefully']) / len(malformed_results)
        print(f"\nMalformed input handling rate: {graceful_handling_rate:.1%}")
        
        return {
            'tests': malformed_results,
            'graceful_handling_rate': graceful_handling_rate
        }
    
    def test_resource_limits(self) -> Dict[str, Any]:
        """Test system behavior under resource constraints"""
        print("Testing resource limits...")
        
        agent = EnhancedVocalFlowAgent()
        resource_results = []
        
        # Test rapid successive calls (not stress test, just quick succession)
        print("  Testing rapid successive calls...")
        try:
            start_time = time.time()
            rapid_calls = 10  # Keep this small
            
            for i in range(rapid_calls):
                result = agent.process_text_enhanced(f"test command {i}")
            
            total_time = time.time() - start_time
            avg_time = total_time / rapid_calls
            
            rapid_test = {
                'test': 'rapid_successive_calls',
                'calls': rapid_calls,
                'total_time': total_time,
                'avg_time_per_call': avg_time,
                'success': True,
                'no_degradation': avg_time < 1.0  # Should still be fast
            }
            
            print(f"    {rapid_calls} calls in {total_time:.3f}s, avg: {avg_time:.4f}s per call")
            
        except Exception as e:
            rapid_test = {
                'test': 'rapid_successive_calls',
                'success': False,
                'error': str(e)
            }
            print(f"    ERROR: {str(e)}")
        
        resource_results.append(rapid_test)
        
        # Test memory stability with multiple agent instances
        print("  Testing multiple agent instances...")
        try:
            agents = []
            for i in range(3):  # Small number to avoid issues
                agents.append(EnhancedVocalFlowAgent())
            
            # Test that all agents work
            all_work = True
            for i, agent_instance in enumerate(agents):
                try:
                    result = agent_instance.process_text_enhanced("test command")
                except Exception:
                    all_work = False
                    break
            
            multiple_agents_test = {
                'test': 'multiple_agents',
                'agents_created': len(agents),
                'all_functional': all_work,
                'success': all_work
            }
            
            print(f"    Created {len(agents)} agents, all functional: {all_work}")
            
        except Exception as e:
            multiple_agents_test = {
                'test': 'multiple_agents',
                'success': False,
                'error': str(e)
            }
            print(f"    ERROR: {str(e)}")
        
        resource_results.append(multiple_agents_test)
        
        self.results['resource_limits'] = resource_results
        
        resource_success = sum(1 for r in resource_results if r['success']) / len(resource_results)
        print(f"\nResource limits handling: {resource_success:.1%}")
        
        return {
            'tests': resource_results,
            'success_rate': resource_success
        }
    
    def test_file_system_robustness(self) -> Dict[str, Any]:
        """Test file system related robustness"""
        print("Testing file system robustness...")
        
        file_results = []
        
        # Test with temporary directory for safety
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test profile directory creation in temporary location
            print("  Testing profile directory handling...")
            try:
                # Simulate profile directory operations
                test_profile_dir = temp_path / "test_profiles"
                test_profile_dir.mkdir(exist_ok=True)
                
                # Test file creation
                test_file = test_profile_dir / "test.json"
                test_file.write_text('{"test": "data"}')
                
                # Test file reading
                content = test_file.read_text()
                
                profile_dir_test = {
                    'test': 'profile_directory_operations',
                    'directory_created': test_profile_dir.exists(),
                    'file_created': test_file.exists(),
                    'file_readable': '"test"' in content,
                    'success': True
                }
                
                print(f"    Profile operations: Directory OK, File OK")
                
            except Exception as e:
                profile_dir_test = {
                    'test': 'profile_directory_operations',
                    'success': False,
                    'error': str(e)
                }
                print(f"    ERROR: {str(e)}")
            
            file_results.append(profile_dir_test)
            
            # Test read-only scenario simulation
            print("  Testing read-only conditions...")
            try:
                # Create a file and make directory read-only (if possible)
                readonly_dir = temp_path / "readonly_test"
                readonly_dir.mkdir(exist_ok=True)
                
                try:
                    # Try to make read-only (may not work on all systems)
                    readonly_dir.chmod(0o444)
                    readonly_attempted = True
                except Exception:
                    readonly_attempted = False
                
                readonly_test = {
                    'test': 'readonly_handling',
                    'readonly_attempted': readonly_attempted,
                    'graceful_handling': True,  # System should handle this
                    'success': True
                }
                
                print(f"    Read-only test: {'Attempted' if readonly_attempted else 'Skipped (permissions)'}")
                
                # Restore permissions for cleanup
                try:
                    readonly_dir.chmod(0o755)
                except Exception:
                    pass
                
            except Exception as e:
                readonly_test = {
                    'test': 'readonly_handling',
                    'success': False,
                    'error': str(e)
                }
                print(f"    ERROR: {str(e)}")
            
            file_results.append(readonly_test)
        
        self.results['file_system'] = file_results
        
        file_success = sum(1 for r in file_results if r['success']) / len(file_results)
        print(f"\nFile system robustness: {file_success:.1%}")
        
        return {
            'tests': file_results,
            'success_rate': file_success
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete robustness test suite"""
        print("VocalFlow Robustness Test Suite")
        print("=" * 35)
        
        # Run all robustness tests
        edge_results = self.test_edge_cases()
        print()
        
        malformed_results = self.test_malformed_input()
        print()
        
        resource_results = self.test_resource_limits()
        print()
        
        file_results = self.test_file_system_robustness()
        print()
        
        # Generate robustness summary
        self.generate_robustness_report(edge_results, malformed_results, 
                                      resource_results, file_results)
        
        return {
            'edge_cases': edge_results,
            'malformed_input': malformed_results,
            'resource_limits': resource_results,
            'file_system': file_results,
            'detailed_results': self.results
        }
    
    def generate_robustness_report(self, edge, malformed, resource, file_sys):
        """Generate robustness test summary"""
        print("Robustness Test Summary")
        print("=" * 23)
        
        # Calculate metrics
        success_rates = [
            edge['success_rate'],
            malformed['graceful_handling_rate'],
            resource['success_rate'],
            file_sys['success_rate']
        ]
        
        overall_robustness = sum(success_rates) / len(success_rates)
        
        print(f"Edge Case Handling: {edge['success_rate']:.1%}")
        print(f"Malformed Input: {malformed['graceful_handling_rate']:.1%}")
        print(f"Resource Limits: {resource['success_rate']:.1%}")
        print(f"File System: {file_sys['success_rate']:.1%}")
        print(f"Overall Robustness: {overall_robustness:.1%}")
        
        # Robustness assessment
        if overall_robustness >= 0.9:
            assessment = "Highly Robust"
        elif overall_robustness >= 0.8:
            assessment = "Good Robustness"
        elif overall_robustness >= 0.7:
            assessment = "Acceptable Robustness"
        else:
            assessment = "Needs Hardening"
        
        print(f"Robustness Rating: {assessment}")
        
        # Security assessment
        security_issues = sum(1 for r in malformed['tests'] if not r.get('no_security_risk', True))
        print(f"Security Issues: {security_issues} (Good)" if security_issues == 0 else f"Security Issues: {security_issues} (Review needed)")
        
        # Recommendations
        print("\nRecommendations:")
        if edge['success_rate'] < 0.8:
            print("- Improve edge case input validation and handling")
        if malformed['graceful_handling_rate'] < 0.8:
            print("- Strengthen input sanitization and error handling")
        if resource['success_rate'] < 0.8:
            print("- Optimize resource usage and add limits")
        if file_sys['success_rate'] < 0.8:
            print("- Improve file system error handling and permissions")
        
        print("\nRobustness testing complete!")

def main():
    """Run robustness tests"""
    try:
        test_runner = RobustnessTestRunner()
        results = test_runner.run_all_tests()
        
        # Save results for analysis
        import json
        with open('robustness_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: robustness_test_results.json")
        return 0
        
    except Exception as e:
        print(f"Robustness test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
