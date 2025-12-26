"""
DeQoG Diversity Metrics

Metrics for evaluating the diversity of generated N-version code solutions.

Based on the latest paper: "Automated Fault-Tolerant Code Generation via LLMs:
A Diversity-Enhanced and Quality-Assured Approach"

Key Metrics (Section 4.5):
1. LS (Levenshtein Similarity): Syntactic diversity via normalized edit distance
2. SDP (Solutions Difference Probability): Algorithmic diversity via LLM evaluation
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger("diversity_metrics")


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein distance (edit distance) between two strings.
    
    The minimum number of single-character edits (insertions, deletions,
    or substitutions) required to transform s1 into s2.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance (integer)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertions, deletions, substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


@dataclass
class DiversityResult:
    """Container for diversity evaluation results."""
    ls: float                            # Levenshtein Similarity (lower = more diverse)
    sdp: float                           # Solutions Difference Probability (higher = more diverse)
    syntactic_diversity: float           # 1 - LS
    methodological_diversity: float      # Same as SDP
    overall_diversity: float             # Combined metric
    pairwise_details: List[Dict[str, Any]]  # Detailed pairwise comparisons


class DiversityMetrics:
    """
    Diversity Evaluation Metrics for N-Version Code.
    
    Implements the diversity metrics from the DeQoG paper (Section 4.5):
    
    1. LS (Levenshtein Similarity):
       - Measures syntactic diversity via normalized edit distance
       - Formula: LS_norm(ci, cj) = 1 - LD(ci, cj) / max(|ci|, |cj|)
       - Range: [0, 1], where LOWER values indicate HIGHER syntactic diversity
       - We compute the average pairwise LS value for N versions
    
    2. SDP (Solutions Difference Probability):
       - Quantifies algorithmic diversity using LLM-based evaluation
       - Formula: SDP = 1 - Σ_{i<j} S(ci, cj) / C(n,2)
       - Where S(ci, cj) ∈ {0, 1} indicates strategy similarity
       - HIGHER values indicate greater methodological diversity
    
    Reference: Section 4.5 of the DeQoG paper
    """
    
    def __init__(
        self,
        llm_client=None,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize diversity metrics calculator.
        
        Args:
            llm_client: LLM client for SDP strategy comparison
                       If None, SDP uses a heuristic fallback
            similarity_threshold: Threshold for considering codes as "similar"
        """
        self.llm_client = llm_client
        self.similarity_threshold = similarity_threshold
        
        logger.info("DiversityMetrics initialized with LS and SDP metrics")
    
    def compute_ls(self, code_list: List[str]) -> float:
        """
        Compute Levenshtein Similarity (LS).
        
        Measures syntactic diversity via normalized edit distance.
        
        Formula: LS_norm(ci, cj) = 1 - LD(ci, cj) / max(|ci|, |cj|)
        
        We compute the average pairwise LS for all code pairs.
        
        Interpretation:
        - LS close to 1.0: Codes are very similar (low syntactic diversity)
        - LS close to 0.0: Codes are very different (high syntactic diversity)
        
        Args:
            code_list: List of code strings to compare
            
        Returns:
            Mean Levenshtein Similarity (0.0 to 1.0)
        """
        n = len(code_list)
        if n < 2:
            logger.warning("LS requires at least 2 codes, returning 0.0")
            return 0.0
        
        total_similarity = 0.0
        pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                ls_ij = self._normalized_levenshtein_similarity(
                    code_list[i], code_list[j]
                )
                total_similarity += ls_ij
                pairs += 1
        
        ls = total_similarity / pairs if pairs > 0 else 0.0
        
        logger.debug(f"LS computed: {ls:.4f} (from {pairs} pairs)")
        return float(ls)
    
    def _normalized_levenshtein_similarity(self, c1: str, c2: str) -> float:
        """
        Compute normalized Levenshtein Similarity between two codes.
        
        Formula: LS_norm(ci, cj) = 1 - LD(ci, cj) / max(|ci|, |cj|)
        
        Args:
            c1: First code string
            c2: Second code string
            
        Returns:
            Normalized similarity (0.0 to 1.0)
        """
        if not c1 and not c2:
            return 1.0  # Both empty = identical
        
        if not c1 or not c2:
            return 0.0  # One empty = completely different
        
        ld = levenshtein_distance(c1, c2)
        max_len = max(len(c1), len(c2))
        
        # LS = 1 - LD / max_len
        ls = 1.0 - (ld / max_len)
        
        return max(0.0, min(1.0, ls))
    
    def compute_sdp(
        self,
        code_list: List[str],
        llm_client=None
    ) -> float:
        """
        Compute Solutions Difference Probability (SDP).
        
        Quantifies algorithmic diversity using LLM-based evaluation
        of implementation strategies.
        
        Formula: SDP = 1 - Σ_{i<j} S(ci, cj) / C(n,2)
        
        Where S(ci, cj) ∈ {0, 1}:
        - S = 1 if strategies are SIMILAR
        - S = 0 if strategies are DIFFERENT
        
        Interpretation:
        - SDP close to 1.0: All codes use different approaches (high diversity)
        - SDP close to 0.0: All codes use similar approaches (low diversity)
        
        Args:
            code_list: List of code strings to compare
            llm_client: Optional LLM client override
            
        Returns:
            Solutions Difference Probability (0.0 to 1.0)
        """
        client = llm_client or self.llm_client
        n = len(code_list)
        
        if n < 2:
            logger.warning("SDP requires at least 2 codes, returning 1.0")
            return 1.0
        
        # Number of pairs: C(n,2) = n*(n-1)/2
        total_pairs = n * (n - 1) // 2
        
        # Count similar pairs (S = 1)
        similar_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                is_similar = self._are_strategies_similar(
                    code_list[i], code_list[j], client
                )
                if is_similar:
                    similar_pairs += 1
        
        # SDP = 1 - (similar_pairs / total_pairs)
        sdp = 1.0 - (similar_pairs / total_pairs) if total_pairs > 0 else 1.0
        
        logger.debug(f"SDP computed: {sdp:.4f} ({similar_pairs}/{total_pairs} similar)")
        return float(sdp)
    
    def _are_strategies_similar(
        self,
        code1: str,
        code2: str,
        llm_client
    ) -> bool:
        """
        Check if two codes use SIMILAR implementation strategies.
        
        Uses LLM to evaluate whether codes share the same algorithmic approach.
        
        Args:
            code1: First code
            code2: Second code
            llm_client: LLM client for evaluation
            
        Returns:
            True if strategies are SIMILAR (S=1), False if DIFFERENT (S=0)
        """
        # Fallback if no LLM client
        if not llm_client:
            # Use LS as heuristic: if very similar syntactically, likely similar strategy
            ls = self._normalized_levenshtein_similarity(code1, code2)
            return ls > self.similarity_threshold
        
        # Truncate codes to avoid token limits
        max_len = 800
        code1_truncated = code1[:max_len] + "..." if len(code1) > max_len else code1
        code2_truncated = code2[:max_len] + "..." if len(code2) > max_len else code2
        
        prompt = f"""Compare these two code implementations and determine if they use 
the SAME or SIMILAR algorithmic/implementation strategy.

Code 1:
```python
{code1_truncated}
```

Code 2:
```python
{code2_truncated}
```

Consider:
- Core algorithm type (dynamic programming, greedy, divide-and-conquer, etc.)
- Main data structures used
- Overall problem-solving approach

Answer with ONLY 'Similar' if they use the same/similar strategy, or 'Different' if they use different strategies:"""
        
        try:
            response = llm_client.generate(prompt, temperature=0.1)
            content = response.content if hasattr(response, 'content') else str(response)
            # S=1 if similar, S=0 if different
            return 'similar' in content.lower().strip()
        except Exception as e:
            logger.error(f"LLM strategy comparison failed: {e}")
            # Fallback to LS heuristic
            ls = self._normalized_levenshtein_similarity(code1, code2)
            return ls > self.similarity_threshold
    
    def compute_all(
        self,
        code_list: List[str],
        llm_client=None
    ) -> DiversityResult:
        """
        Compute all diversity metrics.
        
        Args:
            code_list: List of code strings
            llm_client: Optional LLM client for SDP
            
        Returns:
            DiversityResult with all metrics
        """
        ls = self.compute_ls(code_list)
        sdp = self.compute_sdp(code_list, llm_client)
        
        # Syntactic diversity = 1 - LS (higher is better)
        syntactic_diversity = 1.0 - ls
        
        # Methodological diversity = SDP (already higher is better)
        methodological_diversity = sdp
        
        # Combined metric: geometric mean of both diversities
        overall_diversity = np.sqrt(syntactic_diversity * methodological_diversity)
        
        # Get pairwise details
        pairwise = self.pairwise_similarities(code_list)
        
        return DiversityResult(
            ls=ls,
            sdp=sdp,
            syntactic_diversity=syntactic_diversity,
            methodological_diversity=methodological_diversity,
            overall_diversity=overall_diversity,
            pairwise_details=pairwise
        )
    
    def pairwise_similarities(
        self,
        code_list: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get all pairwise similarity scores with details.
        
        Args:
            code_list: List of codes
            
        Returns:
            List of pairwise similarity records, sorted by similarity (descending)
        """
        results = []
        n = len(code_list)
        
        for i in range(n):
            for j in range(i + 1, n):
                ls_ij = self._normalized_levenshtein_similarity(
                    code_list[i], code_list[j]
                )
                
                results.append({
                    'pair': (i, j),
                    'levenshtein_similarity': ls_ij,
                    'is_diverse': ls_ij < self.similarity_threshold,
                    'diversity_level': self._classify_diversity(ls_ij)
                })
        
        return sorted(results, key=lambda x: x['levenshtein_similarity'], reverse=True)
    
    def _classify_diversity(self, similarity: float) -> str:
        """Classify diversity level based on similarity."""
        if similarity < 0.3:
            return "high"
        elif similarity < 0.5:
            return "medium-high"
        elif similarity < 0.7:
            return "medium"
        elif similarity < 0.85:
            return "low"
        else:
            return "very_low"
    
    def check_diversity_threshold(
        self,
        code_list: List[str],
        threshold: float = 0.6
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if code diversity meets the required threshold.
        
        Args:
            code_list: List of codes
            threshold: Minimum required diversity (0-1)
            
        Returns:
            Tuple of (meets_threshold, details)
        """
        result = self.compute_all(code_list)
        
        meets_threshold = result.overall_diversity >= threshold
        
        details = {
            'meets_threshold': meets_threshold,
            'required': threshold,
            'achieved': result.overall_diversity,
            'ls': result.ls,
            'sdp': result.sdp,
            'syntactic_diversity': result.syntactic_diversity,
            'methodological_diversity': result.methodological_diversity,
            'recommendation': self._get_recommendation(result, threshold)
        }
        
        return meets_threshold, details
    
    def _get_recommendation(self, result: DiversityResult, threshold: float) -> str:
        """Get recommendation for improving diversity."""
        if result.overall_diversity >= threshold:
            return "Diversity threshold met. No action needed."
        
        if result.syntactic_diversity < 0.5:
            return "Consider using more different coding styles, variable names, and code structures."
        
        if result.methodological_diversity < 0.5:
            return "Consider using more different algorithmic paradigms (DP, Greedy, etc.)."
        
        return "Apply IRQN method to enhance diversity further."


# Backward compatibility: alias for old metric name
def compute_mbcs(code_list: List[str]) -> float:
    """
    Deprecated: Use DiversityMetrics.compute_ls() instead.
    
    This function now returns LS (Levenshtein Similarity) for backward compatibility.
    """
    logger.warning("compute_mbcs is deprecated. Use DiversityMetrics.compute_ls() instead.")
    metrics = DiversityMetrics()
    return metrics.compute_ls(code_list)
