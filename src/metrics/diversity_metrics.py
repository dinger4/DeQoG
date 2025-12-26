"""
DeQoG Diversity Metrics

Metrics for evaluating the diversity of generated N-version code solutions.

Based on the paper: "Automated Fault-Tolerant Code Generation via LLMs:
A Diversity-Enhanced and Quality-Assured Approach"

Key Metrics:
1. MBCS (Mean BERT Cosine Similarity): Semantic similarity using CodeBERT embeddings
2. SDP (Solutions Difference Probability): Methodological diversity via LLM judgment
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger("diversity_metrics")


@dataclass
class DiversityResult:
    """Container for diversity evaluation results."""
    mbcs: float                          # Mean BERT Cosine Similarity (lower = more diverse)
    sdp: float                           # Solutions Difference Probability (higher = more diverse)
    semantic_diversity: float            # 1 - MBCS
    methodological_diversity: float      # Same as SDP
    overall_diversity: float             # Combined metric
    pairwise_details: List[Dict[str, Any]]  # Detailed pairwise comparisons


class DiversityMetrics:
    """
    Diversity Evaluation Metrics for N-Version Code.
    
    Implements the two core diversity metrics from the DeQoG paper:
    
    1. MBCS (Mean BERT Cosine Similarity):
       - Uses CodeBERT embeddings to compute semantic similarity
       - Measures code similarity at the representation level
       - Lower MBCS indicates higher semantic diversity
       - Formula: MBCS = (2 / N(N-1)) * Σ cos(embed(ci), embed(cj))
    
    2. SDP (Solutions Difference Probability):
       - Uses LLM to judge if two codes use different algorithmic approaches
       - Measures methodological/algorithmic diversity
       - Higher SDP indicates higher methodological diversity
       - Formula: SDP = (different_pairs / total_pairs)
    
    Reference: Section 4.1 of the DeQoG paper
    """
    
    def __init__(
        self,
        codebert_model=None,
        llm_client=None,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize diversity metrics calculator.
        
        Args:
            codebert_model: CodeBERT model for generating embeddings
                           If None, uses a simple fallback embedding
            llm_client: LLM client for methodological comparison
                       If None, SDP falls back to 1-MBCS estimate
            similarity_threshold: Threshold for considering codes as "similar"
        """
        self.codebert_model = codebert_model
        self.llm_client = llm_client
        self.similarity_threshold = similarity_threshold
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        
        logger.info(f"DiversityMetrics initialized (CodeBERT: {codebert_model is not None})")
    
    def compute_mbcs(self, code_list: List[str]) -> float:
        """
        Compute Mean BERT Cosine Similarity (MBCS).
        
        Uses CodeBERT embeddings to calculate the average pairwise
        cosine similarity between all code pairs.
        
        Formula: MBCS = (2 / N(N-1)) * Σ_{i<j} cos(embed(c_i), embed(c_j))
        
        Interpretation:
        - MBCS close to 1.0: Codes are very similar (low diversity)
        - MBCS close to 0.0: Codes are very different (high diversity)
        
        Args:
            code_list: List of code strings to compare
            
        Returns:
            Mean cosine similarity (0.0 to 1.0)
        """
        n = len(code_list)
        if n < 2:
            logger.warning("MBCS requires at least 2 codes, returning 0.0")
            return 0.0
        
        # Get embeddings for all codes
        embeddings = [self._get_embedding(code) for code in code_list]
        
        # Compute pairwise similarities
        total_similarity = 0.0
        pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                total_similarity += similarity
                pairs += 1
        
        mbcs = total_similarity / pairs if pairs > 0 else 0.0
        
        logger.debug(f"MBCS computed: {mbcs:.4f} (from {pairs} pairs)")
        return float(mbcs)
    
    def compute_sdp(
        self,
        code_list: List[str],
        llm_client=None
    ) -> float:
        """
        Compute Solutions Difference Probability (SDP).
        
        Uses LLM to evaluate whether each pair of codes uses
        SIGNIFICANTLY different algorithmic approaches or methodologies.
        
        Formula: SDP = (number of different pairs) / (total pairs)
        
        Interpretation:
        - SDP close to 1.0: All codes use different approaches (high diversity)
        - SDP close to 0.0: All codes use similar approaches (low diversity)
        
        Args:
            code_list: List of code strings to compare
            llm_client: Optional LLM client override
            
        Returns:
            Probability of solutions being different (0.0 to 1.0)
        """
        client = llm_client or self.llm_client
        n = len(code_list)
        
        if n < 2:
            logger.warning("SDP requires at least 2 codes, returning 1.0")
            return 1.0
        
        # Fallback if no LLM client
        if not client:
            logger.warning("No LLM client for SDP, using 1-MBCS as fallback")
            return 1.0 - self.compute_mbcs(code_list)
        
        different_pairs = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                is_different = self._are_methodologically_different(
                    code_list[i], code_list[j], client
                )
                if is_different:
                    different_pairs += 1
                total_pairs += 1
        
        sdp = different_pairs / total_pairs if total_pairs > 0 else 1.0
        
        logger.debug(f"SDP computed: {sdp:.4f} ({different_pairs}/{total_pairs} different)")
        return float(sdp)
    
    def compute_all(
        self,
        code_list: List[str],
        llm_client=None
    ) -> DiversityResult:
        """
        Compute all diversity metrics.
        
        Args:
            code_list: List of code strings
            llm_client: Optional LLM client
            
        Returns:
            DiversityResult with all metrics
        """
        mbcs = self.compute_mbcs(code_list)
        sdp = self.compute_sdp(code_list, llm_client)
        
        semantic_diversity = 1.0 - mbcs
        methodological_diversity = sdp
        
        # Combined metric: geometric mean of both diversities
        overall_diversity = np.sqrt(semantic_diversity * methodological_diversity)
        
        # Get pairwise details
        pairwise = self.pairwise_similarities(code_list)
        
        return DiversityResult(
            mbcs=mbcs,
            sdp=sdp,
            semantic_diversity=semantic_diversity,
            methodological_diversity=methodological_diversity,
            overall_diversity=overall_diversity,
            pairwise_details=pairwise
        )
    
    def _get_embedding(self, code: str) -> np.ndarray:
        """
        Get embedding for a code string.
        
        Uses CodeBERT if available, otherwise falls back to a simple
        feature-based embedding.
        """
        # Check cache first
        cache_key = hash(code)
        if cache_key in self._embeddings_cache:
            return self._embeddings_cache[cache_key]
        
        # Try CodeBERT
        if self.codebert_model is not None:
            try:
                embedding = self.codebert_model.encode(code)
                self._embeddings_cache[cache_key] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"CodeBERT embedding failed: {e}")
        
        # Fallback to simple embedding
        embedding = self._simple_embedding(code)
        self._embeddings_cache[cache_key] = embedding
        return embedding
    
    def _simple_embedding(self, code: str, dim: int = 768) -> np.ndarray:
        """
        Create a simple embedding based on code characteristics.
        
        This is a fallback when CodeBERT is not available.
        Uses a combination of:
        - Hash-based random features (for uniqueness)
        - Structural features (for meaningful comparison)
        """
        # Use hash for reproducible randomness
        np.random.seed(hash(code) % (2**32))
        base = np.random.randn(dim)
        
        # Extract structural features
        features = np.zeros(20)
        
        # Length-based features
        features[0] = len(code) / 1000  # Normalized length
        features[1] = code.count('\n') / 50  # Line count
        
        # Keyword counts (normalized)
        keywords = {
            'def ': 2, 'for ': 3, 'while ': 4, 'if ': 5,
            'return ': 6, 'class ': 7, 'import ': 8,
            'try:': 9, 'except': 10, 'lambda': 11,
            'yield': 12, 'async': 13, 'await': 14
        }
        
        for kw, idx in keywords.items():
            features[idx] = code.count(kw) / 10
        
        # Extend features to match dimension
        extended = np.tile(features, dim // 20 + 1)[:dim]
        
        # Combine base and features
        embedding = base * 0.8 + extended * 0.2
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _are_methodologically_different(
        self,
        code1: str,
        code2: str,
        llm_client
    ) -> bool:
        """
        Check if two codes use different methodologies using LLM.
        
        The LLM evaluates whether the codes use significantly different:
        - Algorithmic paradigms (DP vs Greedy vs Divide-Conquer, etc.)
        - Data structures (arrays vs trees vs graphs, etc.)
        - Control flow patterns (iterative vs recursive)
        """
        # Truncate codes to avoid token limits
        max_len = 800
        code1_truncated = code1[:max_len] + "..." if len(code1) > max_len else code1
        code2_truncated = code2[:max_len] + "..." if len(code2) > max_len else code2
        
        prompt = f"""Compare these two code implementations and determine if they use 
SIGNIFICANTLY DIFFERENT algorithmic approaches.

Code 1:
```python
{code1_truncated}
```

Code 2:
```python
{code2_truncated}
```

Consider:
- Algorithm type (dynamic programming, greedy, divide-and-conquer, etc.)
- Data structures used (lists, dicts, trees, graphs, etc.)
- Control flow patterns (iterative vs recursive)
- Problem-solving strategy

Answer with ONLY 'Yes' if they are significantly different, or 'No' if they are similar:"""
        
        try:
            response = llm_client.generate(prompt, temperature=0.1)
            content = response.content if hasattr(response, 'content') else str(response)
            return 'yes' in content.lower().strip()
        except Exception as e:
            logger.error(f"LLM comparison failed: {e}")
            return False
    
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
                emb_i = self._get_embedding(code_list[i])
                emb_j = self._get_embedding(code_list[j])
                sim = self._cosine_similarity(emb_i, emb_j)
                
                results.append({
                    'pair': (i, j),
                    'similarity': sim,
                    'is_diverse': sim < self.similarity_threshold,
                    'diversity_level': self._classify_diversity(sim)
                })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    
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
            'mbcs': result.mbcs,
            'sdp': result.sdp,
            'recommendation': self._get_recommendation(result, threshold)
        }
        
        return meets_threshold, details
    
    def _get_recommendation(self, result: DiversityResult, threshold: float) -> str:
        """Get recommendation for improving diversity."""
        if result.overall_diversity >= threshold:
            return "Diversity threshold met. No action needed."
        
        if result.semantic_diversity < 0.5:
            return "Consider using more different implementation styles and code structures."
        
        if result.methodological_diversity < 0.5:
            return "Consider using more different algorithmic paradigms (DP, Greedy, etc.)."
        
        return "Apply IRQN method to enhance diversity further."
    
    def clear_cache(self):
        """Clear the embeddings cache."""
        self._embeddings_cache.clear()
        logger.debug("Embeddings cache cleared")
