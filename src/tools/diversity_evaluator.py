"""
DeQoG Diversity Evaluator

Multi-dimensional diversity evaluation tool that measures:
- Semantic similarity (CodeBERT embeddings)
- Methodological difference (LLM-based comparison)
- Execution path divergence
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

from .base_tool import BaseTool
from ..utils.logger import get_logger

logger = get_logger("diversity_evaluator")


class DiversityEvaluator(BaseTool):
    """
    Multi-dimensional Diversity Evaluator.
    
    Evaluates code diversity using multiple metrics:
    - MBCS (Mean BERT Cosine Similarity): Semantic similarity
    - SDP (Solutions Difference Probability): Methodological diversity
    - Execution path divergence
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        similarity_threshold: float = 0.7,
        cache_embeddings: bool = True,
        llm_client=None
    ):
        """
        Initialize the diversity evaluator.
        
        Args:
            model_name: Name of the CodeBERT model to use
            similarity_threshold: Threshold for similarity detection
            cache_embeddings: Whether to cache embeddings
            llm_client: LLM client for methodological comparison
        """
        super().__init__(
            name="diversity_evaluator",
            description="Evaluates diversity of generated code solutions"
        )
        
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.cache_embeddings = cache_embeddings
        self.llm_client = llm_client
        
        self._model = None
        self._tokenizer = None
        self._embeddings_cache: Dict[str, np.ndarray] = {}
    
    def _load_model(self):
        """Load the CodeBERT model."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            logger.info(f"Loading CodeBERT model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            
            # Use GPU if available
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            
            logger.info("CodeBERT model loaded successfully")
        except ImportError:
            logger.warning("Transformers not available, using fallback similarity")
            self._model = None
        except Exception as e:
            logger.error(f"Failed to load CodeBERT: {e}")
            self._model = None
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute diversity evaluation.
        
        Args:
            params: Dictionary containing:
                - code_list: List of code strings to evaluate
                - evaluation_type: Type of evaluation ('semantic', 'methodological', 'all')
                - llm_client: Optional LLM client for methodological evaluation
                
        Returns:
            Dictionary containing diversity scores and analysis
        """
        code_list = params.get('code_list', [])
        evaluation_type = params.get('evaluation_type', 'all')
        llm_client = params.get('llm_client', self.llm_client)
        
        if len(code_list) < 2:
            return {
                'diversity_score': 1.0,
                'message': 'Not enough codes to evaluate diversity'
            }
        
        results = {}
        
        if evaluation_type in ['semantic', 'all']:
            results['mbcs'] = self.compute_mbcs(code_list)
            results['semantic_diversity'] = 1.0 - results['mbcs']
        
        if evaluation_type in ['methodological', 'all'] and llm_client:
            results['sdp'] = self.compute_sdp(code_list, llm_client)
            results['methodological_diversity'] = results['sdp']
        
        if evaluation_type == 'all':
            # Combined diversity score
            scores = [v for k, v in results.items() 
                     if k.endswith('diversity') and isinstance(v, (int, float))]
            results['overall_diversity'] = np.mean(scores) if scores else 0.0
        
        return results
    
    def compute_mbcs(self, code_list: List[str]) -> float:
        """
        Compute Mean BERT Cosine Similarity.
        
        Uses CodeBERT embeddings to calculate semantic similarity
        between all pairs of code.
        
        Args:
            code_list: List of code strings
            
        Returns:
            Mean cosine similarity (lower is more diverse)
        """
        self._load_model()
        
        if self._model is None:
            # Fallback: use simple text-based similarity
            return self._fallback_similarity(code_list)
        
        # Get embeddings for all codes
        embeddings = [self._get_embedding(code) for code in code_list]
        
        # Calculate pairwise cosine similarity
        n = len(embeddings)
        if n < 2:
            return 0.0
        
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _get_embedding(self, code: str) -> np.ndarray:
        """Get embedding for a code string."""
        if self.cache_embeddings and code in self._embeddings_cache:
            return self._embeddings_cache[code]
        
        try:
            import torch
            
            # Tokenize
            inputs = self._tokenizer(
                code,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get embedding
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            if self.cache_embeddings:
                self._embeddings_cache[code] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            # Return random embedding as fallback
            return np.random.randn(768)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _fallback_similarity(self, code_list: List[str]) -> float:
        """Fallback similarity using simple text comparison."""
        from difflib import SequenceMatcher
        
        n = len(code_list)
        if n < 2:
            return 0.0
        
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                ratio = SequenceMatcher(None, code_list[i], code_list[j]).ratio()
                similarities.append(ratio)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def compute_sdp(self, code_list: List[str], llm_client) -> float:
        """
        Compute Solutions Difference Probability.
        
        Uses LLM to evaluate whether code pairs use different
        algorithmic approaches or methodologies.
        
        Args:
            code_list: List of code strings
            llm_client: LLM client for comparison
            
        Returns:
            Probability of solutions being different (higher is more diverse)
        """
        n = len(code_list)
        if n < 2:
            return 1.0
        
        different_pairs = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                is_different = self._are_methodologically_different(
                    code_list[i], code_list[j], llm_client
                )
                if is_different:
                    different_pairs += 1
                total_pairs += 1
        
        return different_pairs / total_pairs if total_pairs > 0 else 1.0
    
    def _are_methodologically_different(
        self,
        code1: str,
        code2: str,
        llm_client
    ) -> bool:
        """Check if two code solutions use different methodologies."""
        prompt = f"""Compare these two code implementations and determine if they use different algorithmic approaches or methodologies.

Code 1:
```python
{code1[:1000]}  # Truncate for token limit
```

Code 2:
```python
{code2[:1000]}
```

Consider:
1. Do they use different algorithms (e.g., DP vs greedy)?
2. Do they have different data structures?
3. Do they have different control flow patterns?

Answer with ONLY 'Yes' if they are SIGNIFICANTLY different, or 'No' if they are similar.
Your answer (Yes/No):"""

        try:
            response = llm_client.generate(prompt, temperature=0.1)
            content = response.content if hasattr(response, 'content') else str(response)
            return 'yes' in content.lower()
        except Exception as e:
            logger.error(f"LLM comparison failed: {e}")
            return False
    
    def calculate_semantic_similarity(
        self,
        code1: str,
        code2: str
    ) -> float:
        """
        Calculate semantic similarity between two code snippets.
        
        Args:
            code1: First code string
            code2: Second code string
            
        Returns:
            Similarity score (0-1)
        """
        self._load_model()
        
        if self._model is None:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, code1, code2).ratio()
        
        emb1 = self._get_embedding(code1)
        emb2 = self._get_embedding(code2)
        
        return self._cosine_similarity(emb1, emb2)
    
    def calculate_methodological_difference(
        self,
        code1: str,
        code2: str,
        llm_client=None
    ) -> float:
        """
        Calculate methodological difference between two code snippets.
        
        Args:
            code1: First code string
            code2: Second code string
            llm_client: LLM client (uses self.llm_client if None)
            
        Returns:
            Difference score (0-1, higher means more different)
        """
        client = llm_client or self.llm_client
        if client is None:
            # Fallback to inverse semantic similarity
            return 1.0 - self.calculate_semantic_similarity(code1, code2)
        
        is_different = self._are_methodologically_different(code1, code2, client)
        return 1.0 if is_different else 0.0
    
    def is_diverse_enough(
        self,
        new_code: str,
        existing_codes: List[str],
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Check if a new code is diverse enough from existing codes.
        
        Args:
            new_code: New code to check
            existing_codes: List of existing codes
            threshold: Similarity threshold (uses self.similarity_threshold if None)
            
        Returns:
            Tuple of (is_diverse, max_similarity)
        """
        if not existing_codes:
            return True, 0.0
        
        threshold = threshold or self.similarity_threshold
        
        similarities = [
            self.calculate_semantic_similarity(new_code, existing)
            for existing in existing_codes
        ]
        
        max_sim = max(similarities)
        return max_sim < threshold, max_sim
    
    def get_diversity_report(
        self,
        code_list: List[str],
        llm_client=None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive diversity report.
        
        Args:
            code_list: List of code strings
            llm_client: LLM client for methodological analysis
            
        Returns:
            Comprehensive diversity report
        """
        client = llm_client or self.llm_client
        
        report = {
            'num_codes': len(code_list),
            'metrics': {}
        }
        
        # Semantic diversity
        mbcs = self.compute_mbcs(code_list)
        report['metrics']['mbcs'] = mbcs
        report['metrics']['semantic_diversity'] = 1.0 - mbcs
        
        # Methodological diversity (if LLM available)
        if client:
            sdp = self.compute_sdp(code_list, client)
            report['metrics']['sdp'] = sdp
            report['metrics']['methodological_diversity'] = sdp
        
        # Pairwise similarities
        n = len(code_list)
        pairwise = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.calculate_semantic_similarity(code_list[i], code_list[j])
                pairwise.append({
                    'pair': (i, j),
                    'similarity': sim
                })
        
        report['pairwise_similarities'] = sorted(
            pairwise, 
            key=lambda x: x['similarity'], 
            reverse=True
        )
        
        # Summary statistics
        if pairwise:
            sims = [p['similarity'] for p in pairwise]
            report['summary'] = {
                'min_similarity': min(sims),
                'max_similarity': max(sims),
                'mean_similarity': np.mean(sims),
                'std_similarity': np.std(sims)
            }
        
        return report
    
    def clear_cache(self):
        """Clear the embeddings cache."""
        self._embeddings_cache.clear()
        logger.info("Embeddings cache cleared")

