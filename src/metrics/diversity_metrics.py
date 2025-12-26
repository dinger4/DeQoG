"""
DeQoG Diversity Metrics

Metrics for evaluating the diversity of generated code solutions.
"""

import numpy as np
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger("diversity_metrics")


class DiversityMetrics:
    """
    Diversity Evaluation Metrics.
    
    Implements:
    - MBCS (Mean BERT Cosine Similarity): Semantic similarity metric
    - SDP (Solutions Difference Probability): Methodological diversity metric
    """
    
    def __init__(
        self,
        codebert_model=None,
        llm_client=None
    ):
        """
        Initialize diversity metrics.
        
        Args:
            codebert_model: CodeBERT model for embeddings
            llm_client: LLM client for methodological comparison
        """
        self.codebert_model = codebert_model
        self.llm_client = llm_client
        self._embeddings_cache: Dict[str, np.ndarray] = {}
    
    def compute_mbcs(self, code_list: List[str]) -> float:
        """
        Compute Mean BERT Cosine Similarity.
        
        Uses CodeBERT embeddings to calculate semantic similarity
        between all pairs of code solutions.
        
        Lower MBCS indicates higher diversity.
        
        Args:
            code_list: List of code strings
            
        Returns:
            Mean cosine similarity (0-1)
        """
        n = len(code_list)
        if n < 2:
            return 0.0
        
        embeddings = [self._get_embedding(code) for code in code_list]
        
        total_similarity = 0.0
        pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                total_similarity += similarity
                pairs += 1
        
        mbcs = total_similarity / pairs if pairs > 0 else 0.0
        return float(mbcs)
    
    def compute_sdp(
        self,
        code_list: List[str],
        llm_client=None
    ) -> float:
        """
        Compute Solutions Difference Probability.
        
        Uses LLM to evaluate whether code pairs use different
        algorithmic approaches or methodologies.
        
        Higher SDP indicates higher diversity.
        
        Args:
            code_list: List of code strings
            llm_client: Optional LLM client override
            
        Returns:
            Probability of solutions being different (0-1)
        """
        client = llm_client or self.llm_client
        n = len(code_list)
        
        if n < 2:
            return 1.0
        
        if not client:
            logger.warning("No LLM client for SDP, using fallback")
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
        return float(sdp)
    
    def compute_all(
        self,
        code_list: List[str],
        llm_client=None
    ) -> Dict[str, float]:
        """
        Compute all diversity metrics.
        
        Args:
            code_list: List of code strings
            llm_client: Optional LLM client
            
        Returns:
            Dictionary of all metrics
        """
        mbcs = self.compute_mbcs(code_list)
        sdp = self.compute_sdp(code_list, llm_client)
        
        return {
            'mbcs': mbcs,
            'sdp': sdp,
            'semantic_diversity': 1.0 - mbcs,
            'methodological_diversity': sdp,
            'overall_diversity': (1.0 - mbcs + sdp) / 2
        }
    
    def _get_embedding(self, code: str) -> np.ndarray:
        """Get embedding for code."""
        if code in self._embeddings_cache:
            return self._embeddings_cache[code]
        
        if self.codebert_model is not None:
            try:
                embedding = self.codebert_model.encode(code)
                self._embeddings_cache[code] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")
        
        # Fallback: simple TF-IDF-like embedding
        embedding = self._simple_embedding(code)
        self._embeddings_cache[code] = embedding
        return embedding
    
    def _simple_embedding(self, code: str, dim: int = 768) -> np.ndarray:
        """
        Create a simple embedding based on code characteristics.
        
        This is a fallback when CodeBERT is not available.
        """
        # Use hash-based features
        np.random.seed(hash(code) % (2**32))
        base = np.random.randn(dim)
        
        # Add some code-specific features
        features = []
        
        # Length features
        features.append(len(code))
        features.append(code.count('\n'))
        features.append(code.count('def '))
        features.append(code.count('for '))
        features.append(code.count('while '))
        features.append(code.count('if '))
        features.append(code.count('return '))
        
        # Normalize and add to base
        features = np.array(features[:dim] + [0] * (dim - len(features)))
        features = features / (np.linalg.norm(features) + 1e-8)
        
        embedding = base * 0.9 + features * 0.1
        return embedding / (np.linalg.norm(embedding) + 1e-8)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
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
        """Check if two codes use different methodologies."""
        prompt = f"""Compare these two code implementations:

Code 1:
```python
{code1[:800]}
```

Code 2:
```python
{code2[:800]}
```

Are they using SIGNIFICANTLY different algorithmic approaches?
Consider: algorithm type, data structures, control flow patterns.

Answer ONLY 'Yes' or 'No':"""
        
        try:
            response = llm_client.generate(prompt, temperature=0.1)
            content = response.content if hasattr(response, 'content') else str(response)
            return 'yes' in content.lower()
        except Exception as e:
            logger.error(f"LLM comparison failed: {e}")
            return False
    
    def pairwise_similarities(
        self,
        code_list: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get all pairwise similarity scores.
        
        Args:
            code_list: List of codes
            
        Returns:
            List of pairwise similarity records
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
                    'is_diverse': sim < 0.7
                })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    def clear_cache(self):
        """Clear embeddings cache."""
        self._embeddings_cache.clear()

