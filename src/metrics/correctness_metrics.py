"""
DeQoG Correctness Metrics

Metrics for evaluating code functional correctness.
"""

from typing import Any, Dict, List

from ..utils.logger import get_logger

logger = get_logger("correctness_metrics")


class CorrectnessMetrics:
    """
    Code Correctness Metrics.
    
    Implements metrics for evaluating functional correctness:
    - Pass Rate: Percentage of tests passed
    - TPR (Test Pass Rate): Average pass rate across all versions
    """
    
    def compute_pass_rate(self, test_results: List[Dict[str, Any]]) -> float:
        """
        Compute test pass rate.
        
        Args:
            test_results: List of test result dictionaries
                         Each should have 'passed' boolean field
                         
        Returns:
            Pass rate (0-1)
        """
        if not test_results:
            return 0.0
        
        passed = sum(1 for r in test_results if r.get('passed', False))
        return passed / len(test_results)
    
    def compute_tpr(
        self,
        results_per_version: List[List[Dict[str, Any]]]
    ) -> float:
        """
        Compute Test Pass Rate (TPR) across all versions.
        
        Average pass rate across all N versions.
        
        Args:
            results_per_version: List of test results for each version
            
        Returns:
            Average pass rate (0-1)
        """
        if not results_per_version:
            return 0.0
        
        pass_rates = [
            self.compute_pass_rate(results)
            for results in results_per_version
        ]
        
        return sum(pass_rates) / len(pass_rates)
    
    def compute_version_metrics(
        self,
        version_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each version.
        
        Args:
            version_results: Dictionary mapping version_id to test results
            
        Returns:
            Dictionary of metrics per version
        """
        metrics = {}
        
        for version_id, results in version_results.items():
            passed = sum(1 for r in results if r.get('passed', False))
            total = len(results)
            
            metrics[version_id] = {
                'pass_rate': passed / total if total > 0 else 0.0,
                'passed': passed,
                'failed': total - passed,
                'total': total
            }
        
        return metrics
    
    def compute_aggregate_metrics(
        self,
        version_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """
        Compute aggregate correctness metrics.
        
        Args:
            version_results: Dictionary mapping version_id to test results
            
        Returns:
            Aggregate metrics dictionary
        """
        if not version_results:
            return {
                'tpr': 0.0,
                'min_pass_rate': 0.0,
                'max_pass_rate': 0.0,
                'std_pass_rate': 0.0,
                'perfect_versions': 0,
                'total_versions': 0
            }
        
        pass_rates = []
        perfect_count = 0
        
        for version_id, results in version_results.items():
            pr = self.compute_pass_rate(results)
            pass_rates.append(pr)
            if pr >= 1.0:
                perfect_count += 1
        
        import numpy as np
        pass_rates = np.array(pass_rates)
        
        return {
            'tpr': float(np.mean(pass_rates)),
            'min_pass_rate': float(np.min(pass_rates)),
            'max_pass_rate': float(np.max(pass_rates)),
            'std_pass_rate': float(np.std(pass_rates)),
            'perfect_versions': perfect_count,
            'total_versions': len(version_results)
        }
    
    def identify_failed_tests(
        self,
        version_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[int]]:
        """
        Identify which test cases failed for each version.
        
        Args:
            version_results: Test results per version
            
        Returns:
            Dictionary mapping version_id to list of failed test indices
        """
        failed_tests = {}
        
        for version_id, results in version_results.items():
            failed_tests[version_id] = [
                i for i, r in enumerate(results)
                if not r.get('passed', False)
            ]
        
        return failed_tests
    
    def common_failures(
        self,
        version_results: Dict[str, List[Dict[str, Any]]]
    ) -> List[int]:
        """
        Find test cases that failed in all versions.
        
        These may indicate test case issues or inherently difficult cases.
        
        Args:
            version_results: Test results per version
            
        Returns:
            List of test indices that failed in all versions
        """
        if not version_results:
            return []
        
        # Get failed tests for each version
        failed_per_version = self.identify_failed_tests(version_results)
        
        if not failed_per_version:
            return []
        
        # Find intersection of all failed sets
        version_ids = list(failed_per_version.keys())
        common = set(failed_per_version[version_ids[0]])
        
        for version_id in version_ids[1:]:
            common &= set(failed_per_version[version_id])
        
        return sorted(common)

