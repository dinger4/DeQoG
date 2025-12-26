#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeQoG Basic Usage Example

This example demonstrates how to use the DeQoG pipeline
to generate N-version fault-tolerant code.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.pipeline import DeQoGPipeline
from utils.config import Config


def main():
    """Main function demonstrating DeQoG usage."""
    
    print("=" * 60)
    print("DeQoG - Diversity-Driven Quality-Assured Code Generation")
    print("=" * 60)
    
    # 1. Load configuration
    print("\n[1] Loading configuration...")
    
    try:
        config = Config.from_yaml('configs/default_config.yaml')
        print("    Configuration loaded from file")
    except FileNotFoundError:
        print("    Using default configuration")
        config = Config({
            'llm': {
                'model_name': 'gpt-4',
                'api_key': os.environ.get('OPENAI_API_KEY', ''),
                'temperature': 0.7,
                'max_tokens': 2000
            },
            'diversity': {
                'threshold': 0.6,
                'hile': {
                    'num_thoughts': 5,
                    'num_solutions': 3,
                    'num_implementations': 2
                },
                'irqn': {
                    'p_qn1': 0.7,
                    'p_qn2': 0.3,
                    'max_iterations': 5,
                    'theta_diff': 0.3,
                    'theta_ident': 0.7
                }
            },
            'quality': {
                'threshold': 0.9,
                'max_refinement_iterations': 5
            },
            'fsm': {
                'max_retries': 3,
                'enable_rollback': True
            }
        })
    
    # 2. Initialize pipeline
    print("\n[2] Initializing DeQoG pipeline...")
    pipeline = DeQoGPipeline(config)
    print("    Pipeline initialized successfully")
    
    # 3. Define programming task
    print("\n[3] Defining programming task...")
    
    task_description = """
    Design and implement a function to find the longest palindromic substring 
    in a given string.
    
    Requirements:
    - Input: A string s (1 <= len(s) <= 1000)
    - Output: The longest palindromic substring
    - If multiple palindromes have the same maximum length, return the first one
    
    Function signature:
    def longest_palindrome(s: str) -> str:
        pass
    
    Examples:
    Input: "babad"
    Output: "bab" (or "aba" is also valid)
    
    Input: "cbbd"
    Output: "bb"
    """
    
    print(f"    Task: Find longest palindromic substring")
    
    # 4. Define test cases
    print("\n[4] Defining test cases...")
    
    test_cases = [
        {'input': 'babad', 'expected_output': 'bab'},
        {'input': 'cbbd', 'expected_output': 'bb'},
        {'input': 'a', 'expected_output': 'a'},
        {'input': 'ac', 'expected_output': 'a'},
        {'input': 'racecar', 'expected_output': 'racecar'},
        {'input': 'aacabdkacaa', 'expected_output': 'aca'},
    ]
    
    print(f"    {len(test_cases)} test cases defined")
    
    # 5. Generate N-version code
    print("\n[5] Generating N-version code...")
    print("    This may take a few minutes...")
    
    try:
        result = pipeline.generate_n_versions(
            task_description=task_description,
            test_cases=test_cases,
            n=5  # Generate 5 versions
        )
        
        # 6. Display results
        print("\n" + "=" * 60)
        print("Generation Results")
        print("=" * 60)
        
        n_codes = result.get('n_version_codes', [])
        print(f"\nGenerated {len(n_codes)} diverse implementations")
        
        # Diversity metrics
        print("\nDiversity Metrics:")
        div_metrics = result.get('diversity_metrics', {})
        for metric, value in div_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        # Quality metrics
        print("\nQuality Metrics:")
        qual_metrics = result.get('quality_metrics', {})
        for metric, value in qual_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        # Display generated codes
        print("\n" + "=" * 60)
        print("Generated Code Versions")
        print("=" * 60)
        
        for i, code_info in enumerate(n_codes, 1):
            print(f"\n--- Version {i} ---")
            algorithm = code_info.get('meta', {}).get('algorithm', 'Unknown')
            pass_rate = code_info.get('metrics', {}).get('pass_rate', 0)
            print(f"Algorithm: {algorithm}")
            print(f"Pass Rate: {pass_rate:.2%}")
            
            code = code_info.get('code', '')
            # Show first 300 characters
            preview = code[:300] + '...' if len(code) > 300 else code
            print(f"\nCode Preview:\n{preview}")
        
        # 7. Run fault injection experiment (optional)
        print("\n" + "=" * 60)
        print("Running Fault Injection Experiments")
        print("=" * 60)
        
        from experiments.fault_injection import FaultInjectionExperiment
        
        fi_experiment = FaultInjectionExperiment(n_versions=len(n_codes))
        
        fi_results = fi_experiment.run_experiment(
            n_version_codes=[c['code'] for c in n_codes],
            test_cases=test_cases,
            patterns={
                'code_level': ['Pat-CL 0', 'Pat-CL 1', 'Pat-CL 3'],
                'algorithm_level': []  # Skip algorithm-level for demo
            }
        )
        
        print("\nFault Injection Results:")
        for pattern, metrics in fi_results['code_level'].items():
            print(f"\n{pattern}:")
            m = metrics['metrics']
            print(f"  Failure Rate: {m['failure_rate']:.2%}")
            print(f"  Majority Consensus Rate: {m['mcr']:.2%}")
            print(f"  Complete Consensus Rate: {m['ccr']:.2%}")
        
        print("\n" + "=" * 60)
        print("DeQoG Example Complete!")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()

