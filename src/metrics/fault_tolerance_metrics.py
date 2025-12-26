"""
DeQoG Fault Tolerance Metrics

Metrics for evaluating fault tolerance capabilities of N-version systems.
"""

from typing import Any, Dict, List
from collections import Counter

from ..utils.logger import get_logger

logger = get_logger("fault_tolerance_metrics")


class FaultToleranceMetrics:
    """
    Fault Tolerance Metrics.
    
    Evaluates the fault tolerance of N-version code sets:
    - FR (Failure Rate): System-level failure rate after voting
    - MCR (Majority Consensus Rate): Rate of majority agreement
    - CCR (Complete Consensus Rate): Rate of complete agreement
    """
    
    def compute_failure_rate(
        self,
        voting_results: List[Dict[str, Any]]
    ) -> float:
        """
        Compute Failure Rate (FR).
        
        Rate of test cases where the voted result is incorrect.
        
        Args:
            voting_results: List of voting result dictionaries
                           Each should have 'system_correct' boolean
                           
        Returns:
            Failure rate (0-1)
        """
        if not voting_results:
            return 0.0
        
        failures = sum(
            1 for r in voting_results
            if not r.get('system_correct', True)
        )
        
        return failures / len(voting_results)
    
    def compute_mcr(
        self,
        voting_results: List[Dict[str, Any]]
    ) -> float:
        """
        Compute Majority Consensus Rate (MCR).
        
        Rate of test cases where a majority of versions agree.
        
        Args:
            voting_results: List of voting result dictionaries
                           Each should have 'has_majority_consensus' boolean
                           
        Returns:
            Majority consensus rate (0-1)
        """
        if not voting_results:
            return 0.0
        
        majority_consensus = sum(
            1 for r in voting_results
            if r.get('has_majority_consensus', False)
        )
        
        return majority_consensus / len(voting_results)
    
    def compute_ccr(
        self,
        voting_results: List[Dict[str, Any]]
    ) -> float:
        """
        Compute Complete Consensus Rate (CCR).
        
        Rate of test cases where all versions produce the same result.
        
        Args:
            voting_results: List of voting result dictionaries
                           Each should have 'all_agree' boolean
                           
        Returns:
            Complete consensus rate (0-1)
        """
        if not voting_results:
            return 0.0
        
        complete_consensus = sum(
            1 for r in voting_results
            if r.get('all_agree', False)
        )
        
        return complete_consensus / len(voting_results)
    
    def compute_all(
        self,
        voting_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute all fault tolerance metrics.
        
        Args:
            voting_results: List of voting result dictionaries
            
        Returns:
            Dictionary of all metrics
        """
        return {
            'failure_rate': self.compute_failure_rate(voting_results),
            'mcr': self.compute_mcr(voting_results),
            'ccr': self.compute_ccr(voting_results),
            'success_rate': 1.0 - self.compute_failure_rate(voting_results)
        }
    
    def majority_vote(
        self,
        outputs: List[Any]
    ) -> tuple:
        """
        Perform majority voting on a list of outputs.
        
        Args:
            outputs: List of outputs from N versions
            
        Returns:
            Tuple of (voted_result, voting_info)
        """
        # Convert to string for comparison
        str_outputs = [str(o) for o in outputs]
        
        # Count occurrences
        counter = Counter(str_outputs)
        
        if not counter:
            return None, {
                'has_majority': False,
                'all_agree': False,
                'vote_distribution': {}
            }
        
        # Get most common
        most_common = counter.most_common()
        voted_result_str = most_common[0][0]
        vote_count = most_common[0][1]
        
        n = len(outputs)
        
        # Find original value
        voted_result = None
        for o, s in zip(outputs, str_outputs):
            if s == voted_result_str:
                voted_result = o
                break
        
        return voted_result, {
            'has_majority': vote_count > n // 2,
            'all_agree': vote_count == n,
            'vote_distribution': dict(counter),
            'winning_count': vote_count,
            'total_votes': n
        }
    
    def run_voting_evaluation(
        self,
        version_outputs: List[List[Any]],
        expected_outputs: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Run voting evaluation across all test cases.
        
        Args:
            version_outputs: List of output lists, one per version
            expected_outputs: List of expected outputs
            
        Returns:
            List of voting result dictionaries
        """
        if not version_outputs or not expected_outputs:
            return []
        
        n_tests = len(expected_outputs)
        n_versions = len(version_outputs)
        
        results = []
        
        for test_idx in range(n_tests):
            # Gather outputs for this test case
            outputs = []
            for version_idx in range(n_versions):
                if test_idx < len(version_outputs[version_idx]):
                    outputs.append(version_outputs[version_idx][test_idx])
                else:
                    outputs.append(None)
            
            # Perform voting
            voted_result, vote_info = self.majority_vote(outputs)
            
            # Check correctness
            expected = expected_outputs[test_idx]
            system_correct = self._compare_results(voted_result, expected)
            
            results.append({
                'test_idx': test_idx,
                'outputs': outputs,
                'voted_result': voted_result,
                'expected': expected,
                'system_correct': system_correct,
                'has_majority_consensus': vote_info['has_majority'],
                'all_agree': vote_info['all_agree'],
                'vote_distribution': vote_info['vote_distribution']
            })
        
        return results
    
    def _compare_results(self, result: Any, expected: Any) -> bool:
        """Compare voting result with expected output."""
        if result is None:
            return False
        
        # Direct comparison
        if result == expected:
            return True
        
        # String comparison
        if str(result) == str(expected):
            return True
        
        # Numeric comparison with tolerance
        try:
            if isinstance(result, (int, float)) and isinstance(expected, (int, float)):
                return abs(float(result) - float(expected)) < 1e-9
        except:
            pass
        
        return False
    
    def analyze_fault_patterns(
        self,
        voting_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in voting results.
        
        Args:
            voting_results: Voting results from evaluation
            
        Returns:
            Analysis dictionary
        """
        if not voting_results:
            return {}
        
        total = len(voting_results)
        
        # Categorize results
        correct_majority = 0
        incorrect_majority = 0
        no_majority = 0
        all_wrong = 0
        all_correct = 0
        
        for r in voting_results:
            if r['all_agree']:
                if r['system_correct']:
                    all_correct += 1
                else:
                    all_wrong += 1
            elif r['has_majority_consensus']:
                if r['system_correct']:
                    correct_majority += 1
                else:
                    incorrect_majority += 1
            else:
                no_majority += 1
        
        return {
            'total_tests': total,
            'all_correct': all_correct,
            'all_wrong': all_wrong,
            'correct_majority': correct_majority,
            'incorrect_majority': incorrect_majority,
            'no_majority': no_majority,
            'categories': {
                'unanimous_correct': all_correct / total if total > 0 else 0,
                'unanimous_wrong': all_wrong / total if total > 0 else 0,
                'majority_correct': correct_majority / total if total > 0 else 0,
                'majority_wrong': incorrect_majority / total if total > 0 else 0,
                'no_consensus': no_majority / total if total > 0 else 0
            }
        }

