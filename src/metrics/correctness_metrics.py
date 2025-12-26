"""
DeQoG Correctness Metrics

Metrics for evaluating the correctness of generated N-version code.

Based on the latest paper: "Automated Fault-Tolerant Code Generation via LLMs:
A Diversity-Enhanced and Quality-Assured Approach"

Key Metric (Section 4.5):
- Pass@k: Probability that at least one correct N-version set exists in k trials
"""

import math
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger("correctness_metrics")


@dataclass
class CorrectnessResult:
    """Container for correctness evaluation results."""
    pass_at_k: Dict[int, float]     # Pass@k for various k values
    all_versions_pass: bool          # Whether all N versions pass all tests
    version_pass_rates: List[float]  # Individual version pass rates
    total_tests: int                 # Total number of test cases
    summary: Dict[str, Any]          # Summary statistics


class CorrectnessMetrics:
    """
    Correctness Evaluation Metrics for N-Version Code.
    
    Implements the Pass@k metric from the DeQoG paper (Section 4.5):
    
    Pass@k:
    - Estimates the probability that at least one correct N-version set
      exists in k trials
    - In practice, Pass@k requires ALL N versions in a set to pass
      the complete test suite
    - This reflects practical N-version deployment requirements
    
    Formula (standard Pass@k):
    Pass@k = 1 - C(n-c, k) / C(n, k)
    
    Where:
    - n = total samples
    - c = correct samples
    - k = number of trials
    
    For DeQoG N-version context:
    - A "correct sample" means ALL N versions pass ALL tests
    
    Reference: Section 4.5 of the DeQoG paper
    """
    
    def __init__(self):
        """Initialize correctness metrics calculator."""
        logger.info("CorrectnessMetrics initialized with Pass@k metric")
    
    def compute_pass_at_k(
        self,
        n_samples: int,
        c_correct: int,
        k: int
    ) -> float:
        """
        Compute Pass@k metric.
        
        Formula: Pass@k = 1 - C(n-c, k) / C(n, k)
        
        This estimates the probability of finding at least one correct
        solution when sampling k times from n total solutions where
        c are correct.
        
        Args:
            n_samples: Total number of samples (n)
            c_correct: Number of correct samples (c)
            k: Number of trials
            
        Returns:
            Pass@k probability (0.0 to 1.0)
        """
        if n_samples < k:
            logger.warning(f"n_samples ({n_samples}) < k ({k}), returning 0.0")
            return 0.0
        
        if c_correct < 1:
            return 0.0
        
        if c_correct >= n_samples:
            return 1.0
        
        # Pass@k = 1 - C(n-c, k) / C(n, k)
        # Using log to avoid numerical overflow for large numbers
        try:
            # C(n-c, k) / C(n, k) = product of (n-c-i)/(n-i) for i in 0..k-1
            log_ratio = 0.0
            for i in range(k):
                numerator = n_samples - c_correct - i
                denominator = n_samples - i
                if numerator < 0:
                    return 1.0  # All remaining samples would be correct
                log_ratio += math.log(numerator) - math.log(denominator)
            
            pass_at_k = 1.0 - math.exp(log_ratio)
            return max(0.0, min(1.0, pass_at_k))
        
        except (ValueError, ZeroDivisionError) as e:
            logger.error(f"Error computing Pass@k: {e}")
            return 0.0
    
    def evaluate_n_version_set(
        self,
        version_test_results: List[List[bool]],
        k_values: List[int] = None
    ) -> CorrectnessResult:
        """
        Evaluate correctness of an N-version code set.
        
        In DeQoG context, Pass@k requires ALL N versions in a set
        to pass the complete test suite.
        
        Args:
            version_test_results: List of test results for each version
                                  Each inner list contains bool results for each test
                                  Example: [[True, True, False], [True, True, True], ...]
            k_values: List of k values for Pass@k computation
                     Default: [1, 5, 10]
        
        Returns:
            CorrectnessResult with all metrics
        """
        if k_values is None:
            k_values = [1, 5, 10]
        
        n_versions = len(version_test_results)
        
        if n_versions == 0:
            return CorrectnessResult(
                pass_at_k={k: 0.0 for k in k_values},
                all_versions_pass=False,
                version_pass_rates=[],
                total_tests=0,
                summary={'error': 'No version results provided'}
            )
        
        # Calculate pass rate for each version
        version_pass_rates = []
        versions_fully_passing = 0
        total_tests = len(version_test_results[0]) if version_test_results[0] else 0
        
        for results in version_test_results:
            if not results:
                version_pass_rates.append(0.0)
                continue
            
            pass_rate = sum(results) / len(results)
            version_pass_rates.append(pass_rate)
            
            # A version "passes" only if ALL tests pass
            if all(results):
                versions_fully_passing += 1
        
        # Check if ALL versions pass ALL tests (N-version requirement)
        all_versions_pass = versions_fully_passing == n_versions
        
        # Compute Pass@k for various k values
        pass_at_k = {}
        for k in k_values:
            # For N-version context: we consider the N-version SET
            # In practice, we might generate multiple N-version sets
            # For a single set, Pass@1 = 1.0 if all pass, 0.0 otherwise
            if k == 1:
                pass_at_k[k] = 1.0 if all_versions_pass else 0.0
            else:
                # For k > 1, compute based on individual version pass rates
                pass_at_k[k] = self.compute_pass_at_k(
                    n_samples=n_versions,
                    c_correct=versions_fully_passing,
                    k=min(k, n_versions)
                )
        
        # Summary statistics
        summary = {
            'n_versions': n_versions,
            'n_tests': total_tests,
            'versions_fully_passing': versions_fully_passing,
            'avg_pass_rate': sum(version_pass_rates) / n_versions if n_versions > 0 else 0.0,
            'min_pass_rate': min(version_pass_rates) if version_pass_rates else 0.0,
            'max_pass_rate': max(version_pass_rates) if version_pass_rates else 0.0,
        }
        
        return CorrectnessResult(
            pass_at_k=pass_at_k,
            all_versions_pass=all_versions_pass,
            version_pass_rates=version_pass_rates,
            total_tests=total_tests,
            summary=summary
        )
    
    def compute_for_multiple_sets(
        self,
        sets_results: List[List[List[bool]]],
        k_values: List[int] = None
    ) -> Dict[str, Any]:
        """
        Compute Pass@k across multiple N-version set generations.
        
        This is useful when you generate multiple candidate N-version sets
        and want to estimate the probability of getting at least one
        fully correct set in k attempts.
        
        Args:
            sets_results: List of N-version sets, each containing
                         version test results (List[List[bool]])
            k_values: List of k values for Pass@k
            
        Returns:
            Aggregated Pass@k results
        """
        if k_values is None:
            k_values = [1, 5, 10]
        
        n_sets = len(sets_results)
        correct_sets = 0
        
        for set_results in sets_results:
            result = self.evaluate_n_version_set(set_results)
            if result.all_versions_pass:
                correct_sets += 1
        
        # Compute Pass@k for sets
        pass_at_k = {}
        for k in k_values:
            pass_at_k[k] = self.compute_pass_at_k(
                n_samples=n_sets,
                c_correct=correct_sets,
                k=min(k, n_sets)
            )
        
        return {
            'pass_at_k': pass_at_k,
            'total_sets': n_sets,
            'correct_sets': correct_sets,
            'set_success_rate': correct_sets / n_sets if n_sets > 0 else 0.0
        }
    
    def compute_individual_pass_at_k(
        self,
        test_results: List[List[bool]],
        k_values: List[int] = None
    ) -> Dict[int, float]:
        """
        Compute Pass@k for individual code versions (not N-version sets).
        
        This is the standard Pass@k metric where each version is
        evaluated independently.
        
        Args:
            test_results: List of test results for each version
            k_values: List of k values
            
        Returns:
            Dictionary mapping k to Pass@k value
        """
        if k_values is None:
            k_values = [1, 5, 10]
        
        n_samples = len(test_results)
        c_correct = sum(1 for results in test_results if all(results))
        
        pass_at_k = {}
        for k in k_values:
            pass_at_k[k] = self.compute_pass_at_k(n_samples, c_correct, min(k, n_samples))
        
        return pass_at_k


# Backward compatibility alias
def compute_tpr(version_pass_rates: List[float]) -> float:
    """
    Deprecated: Use CorrectnessMetrics.evaluate_n_version_set() with Pass@k instead.
    
    This function computes the old TPR (Test Pass Rate) for backward compatibility.
    """
    logger.warning("compute_tpr is deprecated. Use Pass@k metric instead.")
    if not version_pass_rates:
        return 0.0
    return sum(version_pass_rates) / len(version_pass_rates)
