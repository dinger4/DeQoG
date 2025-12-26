"""
DeQoG IRQN Method

Iterative Retention, Questioning and Negation method for
enhancing diversity through iterative refinement.
"""

import random
from typing import Any, Dict, List, Optional, Callable

from ..utils.logger import get_logger

logger = get_logger("irqn")


class IRQNMethod:
    """
    Iterative Retention, Questioning and Negation (IRQN) Method.
    
    Enhances diversity through iterative processing:
    - Retain: Keep outputs that are sufficiently different
    - Question: Refine outputs that are partially similar
    - Negate: Regenerate outputs that are too similar
    
    Parameters:
    - p_qn1: Probability of triggering deep judgment (vs direct accept)
    - p_qn2: Probability of negating even retained outputs (for extra diversity)
    - theta_diff: Similarity threshold for "completely different"
    - theta_ident: Similarity threshold for "too similar"
    """
    
    def __init__(
        self,
        llm_client,
        diversity_evaluator=None,
        p_qn1: float = 0.7,
        p_qn2: float = 0.3,
        theta_diff: float = 0.3,
        theta_ident: float = 0.7,
        max_iterations: int = 5
    ):
        """
        Initialize the IRQN method.
        
        Args:
            llm_client: LLM client for regeneration
            diversity_evaluator: Tool for evaluating similarity
            p_qn1: Probability of triggering judgment
            p_qn2: Probability of negating retained outputs
            theta_diff: Threshold for "different enough"
            theta_ident: Threshold for "too similar"
            max_iterations: Maximum number of iterations
        """
        self.llm_client = llm_client
        self.diversity_evaluator = diversity_evaluator
        
        self.p_qn1 = p_qn1
        self.p_qn2 = p_qn2
        self.theta_diff = theta_diff
        self.theta_ident = theta_ident
        self.max_iterations = max_iterations
        
        # Statistics
        self._stats = {
            'retained': 0,
            'questioned': 0,
            'negated': 0,
            'direct_accept': 0
        }
    
    def execute(
        self,
        initial_outputs: List[Dict[str, Any]],
        knowledge_base: Optional[Dict[str, Any]] = None,
        level: str = 'thought'
    ) -> List[Dict[str, Any]]:
        """
        Execute the IRQN method on a set of outputs.
        
        Args:
            initial_outputs: Initial outputs to process
            knowledge_base: Optional knowledge base for regeneration
            level: Level of outputs ('thought', 'solution', 'implementation')
            
        Returns:
            Processed outputs with enhanced diversity
        """
        logger.info(f"Starting IRQN on {len(initial_outputs)} {level}-level outputs")
        
        # Reset statistics
        self._stats = {
            'retained': 0,
            'questioned': 0,
            'negated': 0,
            'direct_accept': 0
        }
        
        final_outputs = []
        pending_outputs = initial_outputs.copy()
        history = []
        
        for iteration in range(self.max_iterations):
            logger.debug(f"IRQN iteration {iteration + 1}/{self.max_iterations}")
            
            if not pending_outputs:
                logger.info(f"IRQN converged at iteration {iteration + 1}")
                break
            
            current_batch = []
            
            for output in pending_outputs:
                # Probabilistic trigger for deep judgment
                if random.random() > self.p_qn1:
                    # Direct accept without judgment
                    final_outputs.append(output)
                    self._stats['direct_accept'] += 1
                    logger.debug(f"Direct accept: {output.get('id', 'unknown')}")
                    continue
                
                # Evaluate similarity with existing outputs
                reference_set = history + final_outputs
                similarity = self._evaluate_similarity(output, reference_set)
                
                logger.debug(
                    f"Output {output.get('id', 'unknown')}: "
                    f"similarity={similarity:.3f}"
                )
                
                # Decide action based on similarity thresholds
                action, result = self._decide_action(
                    output, similarity, knowledge_base, level
                )
                
                if action == 'retain':
                    final_outputs.append(result)
                    self._stats['retained'] += 1
                elif action == 'question':
                    current_batch.append(result)
                    self._stats['questioned'] += 1
                elif action == 'negate':
                    current_batch.append(result)
                    self._stats['negated'] += 1
            
            # Update for next iteration
            pending_outputs = current_batch
            history = final_outputs.copy()
        
        # Add any remaining pending outputs
        final_outputs.extend(pending_outputs)
        
        logger.info(
            f"IRQN completed: {len(final_outputs)} outputs "
            f"(stats: {self._stats})"
        )
        
        return final_outputs
    
    def _evaluate_similarity(
        self,
        output: Dict[str, Any],
        reference_set: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate similarity between output and reference set.
        
        Args:
            output: Output to evaluate
            reference_set: Reference outputs to compare against
            
        Returns:
            Maximum similarity score (0-1)
        """
        if not reference_set:
            return 0.0
        
        content = output.get('content', '')
        
        if self.diversity_evaluator:
            similarities = [
                self.diversity_evaluator.calculate_semantic_similarity(
                    content,
                    ref.get('content', '')
                )
                for ref in reference_set
            ]
            return max(similarities) if similarities else 0.0
        
        # Fallback: simple text similarity
        from difflib import SequenceMatcher
        
        similarities = [
            SequenceMatcher(None, content, ref.get('content', '')).ratio()
            for ref in reference_set
        ]
        
        return max(similarities) if similarities else 0.0
    
    def _decide_action(
        self,
        output: Dict[str, Any],
        similarity: float,
        knowledge_base: Optional[Dict[str, Any]],
        level: str
    ) -> tuple:
        """
        Decide what action to take based on similarity.
        
        Args:
            output: Output being processed
            similarity: Similarity score
            knowledge_base: Knowledge base for regeneration
            level: Output level
            
        Returns:
            Tuple of (action, processed_output)
        """
        if similarity < self.theta_diff:
            # Completely different - retain
            if random.random() < self.p_qn2:
                # Extra diversity: negate even retained outputs
                result = self._negate_and_regenerate(output, knowledge_base, level)
                return 'negate', result
            else:
                return 'retain', output
        
        elif similarity <= self.theta_ident:
            # Partially similar - question and refine
            result = self._question_and_refine(output, knowledge_base, level)
            return 'question', result
        
        else:
            # Too similar - negate and regenerate
            result = self._negate_and_regenerate(output, knowledge_base, level)
            return 'negate', result
    
    def _question_and_refine(
        self,
        output: Dict[str, Any],
        knowledge_base: Optional[Dict[str, Any]],
        level: str
    ) -> Dict[str, Any]:
        """
        Question and refine a partially similar output.
        
        Args:
            output: Output to refine
            knowledge_base: Knowledge base
            level: Output level
            
        Returns:
            Refined output
        """
        content = output.get('content', '')
        
        prompt = f"""The following {level}-level solution shows partial similarity with existing solutions.

Current solution:
{content}

Please refine it to be more distinctive by:
1. Identifying what makes it similar to other approaches
2. Modifying the approach to be more unique
3. Maintaining feasibility while increasing distinctiveness

Provide the refined solution:"""
        
        refined_content = self._generate(prompt, temperature=0.7)
        
        return {
            **output,
            'id': f"{output.get('id', 'unknown')}_refined",
            'content': refined_content,
            'meta': {
                **output.get('meta', {}),
                'refined': True,
                'original_id': output.get('id')
            }
        }
    
    def _negate_and_regenerate(
        self,
        output: Dict[str, Any],
        knowledge_base: Optional[Dict[str, Any]],
        level: str
    ) -> Dict[str, Any]:
        """
        Negate and regenerate a too-similar output.
        
        Args:
            output: Output to regenerate
            knowledge_base: Knowledge base
            level: Output level
            
        Returns:
            Regenerated output
        """
        content = output.get('content', '')
        
        level_specifics = {
            'thought': 'algorithmic paradigm',
            'solution': 'solution strategy',
            'implementation': 'implementation approach'
        }
        
        prompt = f"""The following {level}-level solution is too similar to existing solutions.

Current solution:
{content}

Generate a COMPLETELY DIFFERENT {level_specifics.get(level, 'approach')} that:
1. Uses a fundamentally different method
2. Employs different data structures or algorithms
3. Takes an opposite or contrasting perspective

IMPORTANT: The new solution must be substantially different.

Provide the new solution:"""
        
        regenerated_content = self._generate(prompt, temperature=0.9)
        
        return {
            **output,
            'id': f"{output.get('id', 'unknown')}_negated",
            'content': regenerated_content,
            'meta': {
                **output.get('meta', {}),
                'negated': True,
                'original_id': output.get('id'),
                'original_rejected': True
            }
        }
    
    def _generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using LLM."""
        response = self.llm_client.generate(prompt, temperature=temperature)
        return response.content if hasattr(response, 'content') else str(response)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get execution statistics."""
        return self._stats.copy()
    
    def set_thresholds(
        self,
        theta_diff: Optional[float] = None,
        theta_ident: Optional[float] = None
    ):
        """
        Update similarity thresholds.
        
        Args:
            theta_diff: New difference threshold
            theta_ident: New identity threshold
        """
        if theta_diff is not None:
            self.theta_diff = theta_diff
        if theta_ident is not None:
            self.theta_ident = theta_ident
    
    def set_probabilities(
        self,
        p_qn1: Optional[float] = None,
        p_qn2: Optional[float] = None
    ):
        """
        Update probability parameters.
        
        Args:
            p_qn1: New p_qn1 value
            p_qn2: New p_qn2 value
        """
        if p_qn1 is not None:
            self.p_qn1 = p_qn1
        if p_qn2 is not None:
            self.p_qn2 = p_qn2

