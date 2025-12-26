"""
DeQoG Fault Tolerance Metrics

Metrics for evaluating the fault tolerance of N-version code systems.

Based on the latest paper: "Automated Fault-Tolerant Code Generation via LLMs:
A Diversity-Enhanced and Quality-Assured Approach"

Key Metrics (Section 4.5):
1. FR (Failure Rate): Percentage of tasks where majority voting fails
2. AFVR (Additional Failure Versions Ratio): System degradation from fault injection
3. Token Cost: Total input and output tokens consumed
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import hashlib

from ..utils.logger import get_logger

logger = get_logger("fault_tolerance_metrics")


@dataclass
class FaultToleranceResult:
    """Container for fault tolerance evaluation results."""
    failure_rate: float              # FR: Percentage of failed majority votes
    success_rate: float              # 1 - FR
    afvr: Optional[float]            # AFVR: Additional Failure Versions Ratio
    voting_details: Dict[str, Any]   # Detailed voting statistics
    token_cost: Optional[int]        # Total token cost (if tracked)


@dataclass
class VotingResult:
    """Result of majority voting on a single test case."""
    voted_output: Any                # The majority-voted output
    is_correct: bool                 # Whether the voted output matches expected
    vote_counts: Dict[Any, int]      # Count of each output value
    agreement_level: str             # "complete", "majority", "tie", "no_majority"
    version_outputs: List[Any]       # Individual version outputs


class FaultToleranceMetrics:
    """
    Fault Tolerance Evaluation Metrics for N-Version Code.
    
    Implements the fault tolerance metrics from the DeQoG paper (Section 4.5):
    
    1. Failure Rate (FR):
       - Percentage of tasks where majority voting produces correct output
         despite injected faults
       - Formula: FR = 1 - (Nc / Nt) × 100%
       - Where Nc = number of tasks with correct majority voting output
       - And Nt = total number of tasks
       - LOWER FR indicates better fault tolerance
    
    2. Additional Failure Versions Ratio (AFVR):
       - Measures system degradation by counting newly introduced failures
       - Formula: AFVR = (N_failpost - N_failpre) / N_vers
       - Where:
         * N_failpost = number of failing versions post fault-injection
         * N_failpre = number of failing versions pre fault-injection
         * N_vers = total number of versions
       - LOWER AFVR indicates better resilience to faults
    
    3. Token Cost:
       - Total input and output tokens consumed during generation
       - Used to evaluate efficiency of the approach
    
    Reference: Section 4.5 of the DeQoG paper
    """
    
    def __init__(self):
        """Initialize fault tolerance metrics calculator."""
        self._token_count = 0
        logger.info("FaultToleranceMetrics initialized with FR and AFVR metrics")
    
    def majority_vote(
        self,
        version_outputs: List[Any]
    ) -> VotingResult:
        """
        Perform majority voting on version outputs.
        
        Args:
            version_outputs: List of outputs from each N-version
            
        Returns:
            VotingResult with voting details
        """
        if not version_outputs:
            return VotingResult(
                voted_output=None,
                is_correct=False,
                vote_counts={},
                agreement_level="no_versions",
                version_outputs=[]
            )
        
        # Normalize outputs for comparison (handle unhashable types)
        normalized = []
        for output in version_outputs:
            try:
                # Try to use the output directly as a hash key
                hash(output)
                normalized.append(output)
            except TypeError:
                # For unhashable types, use a string/hash representation
                normalized.append(self._hash_output(output))
        
        # Count votes
        vote_counts = Counter(normalized)
        
        # Find majority
        n_versions = len(version_outputs)
        majority_threshold = n_versions // 2 + 1
        
        most_common = vote_counts.most_common()
        top_vote_count = most_common[0][1]
        top_vote_output = most_common[0][0]
        
        # Determine agreement level
        if top_vote_count == n_versions:
            agreement_level = "complete"
        elif top_vote_count >= majority_threshold:
            agreement_level = "majority"
        elif len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            agreement_level = "tie"
        else:
            agreement_level = "plurality"  # Most votes but not majority
        
        # Get the original (non-normalized) output for the winning vote
        # Find first occurrence of the winning normalized output
        voted_output = None
        for i, norm_out in enumerate(normalized):
            if norm_out == top_vote_output:
                voted_output = version_outputs[i]
                break
        
        return VotingResult(
            voted_output=voted_output,
            is_correct=False,  # To be set by caller with expected output
            vote_counts=dict(vote_counts),
            agreement_level=agreement_level,
            version_outputs=version_outputs
        )
    
    def _hash_output(self, output: Any) -> str:
        """Create a hashable representation of an output."""
        return hashlib.md5(str(output).encode()).hexdigest()
    
    def compute_failure_rate(
        self,
        version_outputs_per_task: List[List[Any]],
        expected_outputs: List[Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute Failure Rate (FR).
        
        Formula: FR = 1 - (Nc / Nt) × 100%
        
        Where:
        - Nc = number of tasks with correct majority voting output
        - Nt = total number of tasks
        
        Args:
            version_outputs_per_task: For each task, list of outputs from each version
                                     Shape: [n_tasks][n_versions]
            expected_outputs: Expected output for each task
                             Shape: [n_tasks]
        
        Returns:
            Tuple of (failure_rate, details)
        """
        n_tasks = len(version_outputs_per_task)
        
        if n_tasks == 0:
            return 0.0, {'error': 'No tasks provided'}
        
        if len(expected_outputs) != n_tasks:
            logger.error(f"Mismatch: {n_tasks} tasks but {len(expected_outputs)} expected outputs")
            return 1.0, {'error': 'Task/expected output count mismatch'}
        
        correct_tasks = 0
        task_details = []
        
        for i, (version_outputs, expected) in enumerate(zip(version_outputs_per_task, expected_outputs)):
            vote_result = self.majority_vote(version_outputs)
            
            # Check if voted output matches expected
            is_correct = self._outputs_match(vote_result.voted_output, expected)
            vote_result.is_correct = is_correct
            
            if is_correct:
                correct_tasks += 1
            
            task_details.append({
                'task_index': i,
                'voted_output': vote_result.voted_output,
                'expected_output': expected,
                'is_correct': is_correct,
                'agreement_level': vote_result.agreement_level
            })
        
        # FR = 1 - Nc / Nt
        failure_rate = 1.0 - (correct_tasks / n_tasks)
        
        details = {
            'total_tasks': n_tasks,
            'correct_tasks': correct_tasks,
            'failed_tasks': n_tasks - correct_tasks,
            'failure_rate_percent': failure_rate * 100,
            'task_details': task_details
        }
        
        logger.info(f"FR computed: {failure_rate:.4f} ({n_tasks - correct_tasks}/{n_tasks} failed)")
        return failure_rate, details
    
    def _outputs_match(self, output1: Any, output2: Any) -> bool:
        """Check if two outputs match."""
        try:
            # Direct comparison
            if output1 == output2:
                return True
            
            # String comparison (for floating point tolerance)
            if isinstance(output1, (int, float)) and isinstance(output2, (int, float)):
                return abs(output1 - output2) < 1e-9
            
            # String normalization
            return str(output1).strip() == str(output2).strip()
        except Exception:
            return False
    
    def compute_afvr(
        self,
        versions_failing_pre: int,
        versions_failing_post: int,
        total_versions: int
    ) -> float:
        """
        Compute Additional Failure Versions Ratio (AFVR).
        
        Measures system degradation by counting newly introduced failures
        after fault injection.
        
        Formula: AFVR = (N_failpost - N_failpre) / N_vers
        
        Where:
        - N_failpost = number of failing versions post fault-injection
        - N_failpre = number of failing versions pre fault-injection
        - N_vers = total number of versions
        
        Interpretation:
        - AFVR = 0: No additional failures introduced (ideal)
        - AFVR > 0: Some versions started failing after injection
        - AFVR < 0: Some versions recovered (shouldn't happen normally)
        
        Args:
            versions_failing_pre: Number of versions failing before injection
            versions_failing_post: Number of versions failing after injection
            total_versions: Total number of versions
            
        Returns:
            AFVR value
        """
        if total_versions == 0:
            logger.warning("Total versions is 0, returning 0.0 for AFVR")
            return 0.0
        
        afvr = (versions_failing_post - versions_failing_pre) / total_versions
        
        logger.info(f"AFVR computed: {afvr:.4f} "
                   f"(pre: {versions_failing_pre}, post: {versions_failing_post}, total: {total_versions})")
        
        return afvr
    
    def compute_afvr_from_results(
        self,
        pre_injection_results: List[List[bool]],
        post_injection_results: List[List[bool]]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute AFVR from pre and post injection test results.
        
        Args:
            pre_injection_results: Test results before fault injection
                                   Shape: [n_versions][n_tests]
            post_injection_results: Test results after fault injection
                                    Shape: [n_versions][n_tests]
        
        Returns:
            Tuple of (AFVR, details)
        """
        n_versions = len(pre_injection_results)
        
        if n_versions != len(post_injection_results):
            logger.error("Version count mismatch between pre and post injection")
            return 0.0, {'error': 'Version count mismatch'}
        
        # Count failing versions (a version fails if ANY test fails)
        versions_failing_pre = sum(
            1 for results in pre_injection_results 
            if not all(results)
        )
        
        versions_failing_post = sum(
            1 for results in post_injection_results 
            if not all(results)
        )
        
        afvr = self.compute_afvr(
            versions_failing_pre,
            versions_failing_post,
            n_versions
        )
        
        # Identify which versions changed status
        newly_failing = []
        recovered = []
        
        for i, (pre, post) in enumerate(zip(pre_injection_results, post_injection_results)):
            pre_passing = all(pre)
            post_passing = all(post)
            
            if pre_passing and not post_passing:
                newly_failing.append(i)
            elif not pre_passing and post_passing:
                recovered.append(i)
        
        details = {
            'total_versions': n_versions,
            'versions_failing_pre': versions_failing_pre,
            'versions_failing_post': versions_failing_post,
            'newly_failing_versions': newly_failing,
            'recovered_versions': recovered,
            'afvr': afvr
        }
        
        return afvr, details
    
    def track_token_cost(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0
    ):
        """
        Track token consumption for cost evaluation.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        self._token_count += input_tokens + output_tokens
    
    def get_token_cost(self) -> int:
        """Get total token cost tracked so far."""
        return self._token_count
    
    def reset_token_cost(self):
        """Reset token cost counter."""
        self._token_count = 0
    
    def compute_all(
        self,
        version_outputs_per_task: List[List[Any]],
        expected_outputs: List[Any],
        pre_injection_results: Optional[List[List[bool]]] = None,
        post_injection_results: Optional[List[List[bool]]] = None
    ) -> FaultToleranceResult:
        """
        Compute all fault tolerance metrics.
        
        Args:
            version_outputs_per_task: Outputs from each version for each task
            expected_outputs: Expected outputs
            pre_injection_results: Optional pre-injection test results for AFVR
            post_injection_results: Optional post-injection test results for AFVR
            
        Returns:
            FaultToleranceResult with all metrics
        """
        # Compute FR
        failure_rate, voting_details = self.compute_failure_rate(
            version_outputs_per_task,
            expected_outputs
        )
        
        # Compute AFVR if injection results provided
        afvr = None
        if pre_injection_results is not None and post_injection_results is not None:
            afvr, afvr_details = self.compute_afvr_from_results(
                pre_injection_results,
                post_injection_results
            )
            voting_details['afvr_details'] = afvr_details
        
        return FaultToleranceResult(
            failure_rate=failure_rate,
            success_rate=1.0 - failure_rate,
            afvr=afvr,
            voting_details=voting_details,
            token_cost=self._token_count
        )
    
    def evaluate_voting_robustness(
        self,
        version_outputs_per_task: List[List[Any]],
        expected_outputs: List[Any]
    ) -> Dict[str, Any]:
        """
        Evaluate robustness of the voting mechanism.
        
        Analyzes how well the majority voting handles disagreements
        and produces correct outputs.
        
        Args:
            version_outputs_per_task: Outputs from each version for each task
            expected_outputs: Expected outputs
            
        Returns:
            Robustness analysis
        """
        n_tasks = len(version_outputs_per_task)
        
        agreement_counts = {
            'complete': 0,
            'majority': 0,
            'plurality': 0,
            'tie': 0
        }
        
        correct_by_agreement = {
            'complete': 0,
            'majority': 0,
            'plurality': 0,
            'tie': 0
        }
        
        for version_outputs, expected in zip(version_outputs_per_task, expected_outputs):
            vote_result = self.majority_vote(version_outputs)
            is_correct = self._outputs_match(vote_result.voted_output, expected)
            
            level = vote_result.agreement_level
            if level in agreement_counts:
                agreement_counts[level] += 1
                if is_correct:
                    correct_by_agreement[level] += 1
        
        # Calculate accuracy by agreement level
        accuracy_by_agreement = {}
        for level in agreement_counts:
            count = agreement_counts[level]
            correct = correct_by_agreement[level]
            accuracy_by_agreement[level] = correct / count if count > 0 else 0.0
        
        return {
            'total_tasks': n_tasks,
            'agreement_distribution': agreement_counts,
            'correct_by_agreement': correct_by_agreement,
            'accuracy_by_agreement': accuracy_by_agreement,
            'complete_agreement_rate': agreement_counts['complete'] / n_tasks if n_tasks > 0 else 0.0
        }


# Backward compatibility aliases
def compute_mcr(voting_results: List[VotingResult]) -> float:
    """
    Deprecated: MCR is no longer used in the latest paper.
    Use FaultToleranceMetrics.compute_failure_rate() instead.
    """
    logger.warning("compute_mcr is deprecated. Use FR metric instead.")
    if not voting_results:
        return 0.0
    majority_count = sum(1 for r in voting_results if r.agreement_level == 'majority')
    return majority_count / len(voting_results)


def compute_ccr(voting_results: List[VotingResult]) -> float:
    """
    Deprecated: CCR is no longer used in the latest paper.
    Use FaultToleranceMetrics.compute_failure_rate() instead.
    """
    logger.warning("compute_ccr is deprecated. Use FR metric instead.")
    if not voting_results:
        return 0.0
    complete_count = sum(1 for r in voting_results if r.agreement_level == 'complete')
    return complete_count / len(voting_results)
