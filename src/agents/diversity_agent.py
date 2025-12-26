"""
DeQoG Diversity Enhancing Agent

State 2 Agent: Multi-level diversity generation using HILE and IRQN algorithms.
"""

import random
import json
from typing import Any, Dict, List, Optional

from .base_agent import BaseLLMAgent
from ..utils.logger import get_logger

logger = get_logger("diversity_agent")


class DiversityEnhancingAgent(BaseLLMAgent):
    """
    Diversity Enhancing Agent (State 2).
    
    Implements the HILE (Hierarchical Isolation and Local Expansion)
    and IRQN (Iterative Retention, Questioning and Negation) algorithms
    for generating diverse solution approaches.
    
    Three-level diversity:
    1. Thought Level: Different algorithmic approaches (natural language)
    2. Solution Level: Different pseudocode strategies
    3. Implementation Level: Different concrete implementation schemes
    """
    
    def __init__(
        self,
        llm_client,
        diversity_evaluator=None,
        knowledge_search=None,
        dynamic_prompt_generator=None,
        config=None
    ):
        """
        Initialize the diversity enhancing agent.
        
        Args:
            llm_client: LLM client for generation
            diversity_evaluator: Tool for evaluating diversity
            knowledge_search: Tool for searching knowledge
            dynamic_prompt_generator: Tool for generating prompts
            config: Configuration object
        """
        role_prompt = """You are an expert in algorithm design and software engineering.
Your specialty is generating diverse solutions to programming problems.
You think creatively and explore multiple approaches with different:
- Algorithmic paradigms (DP, greedy, divide-conquer, etc.)
- Data structures (arrays, trees, graphs, hash tables)
- Implementation strategies (iterative, recursive, functional)"""
        
        available_tools = {}
        if diversity_evaluator:
            available_tools['diversity_evaluator'] = diversity_evaluator
        if knowledge_search:
            available_tools['knowledge_search'] = knowledge_search
        if dynamic_prompt_generator:
            available_tools['dynamic_prompt_generator'] = dynamic_prompt_generator
        
        super().__init__(
            llm_client=llm_client,
            role_prompt=role_prompt,
            available_tools=available_tools
        )
        
        self.config = config
        
        # HILE parameters
        self.num_thoughts = getattr(config, 'num_thoughts', 5) if config else 5
        self.num_solutions = getattr(config, 'num_solutions', 3) if config else 3
        self.num_implementations = getattr(config, 'num_implementations', 2) if config else 2
        
        # IRQN parameters
        self.p_qn1 = getattr(config, 'p_qn1', 0.7) if config else 0.7
        self.p_qn2 = getattr(config, 'p_qn2', 0.3) if config else 0.3
        self.max_iterations = getattr(config, 'max_iterations', 5) if config else 5
        self.theta_diff = getattr(config, 'theta_diff', 0.3) if config else 0.3
        self.theta_ident = getattr(config, 'theta_ident', 0.7) if config else 0.7
    
    def process(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process understanding result and generate diverse ideas.
        
        Args:
            input_data: Understanding result from State 1
            context: Current context
            
        Returns:
            Dictionary with multi-level diverse outputs
        """
        self._execution_count += 1
        
        logger.info("Starting diversity ideation (State 2)")
        
        task_understanding = input_data.get('task_understanding', {})
        knowledge = input_data.get('collected_knowledge', {})
        
        # Level 1: Thought-level diversity (algorithmic approaches)
        logger.info("Generating thought-level diversity...")
        thoughts = self.explore_thought_level(task_understanding, knowledge)
        
        # Apply IRQN to enhance thought diversity
        thoughts = self.apply_irqn(thoughts, 'thought', knowledge)
        
        # Level 2: Solution-level diversity (pseudocode strategies)
        logger.info("Generating solution-level diversity...")
        solutions = self.explore_solution_level(thoughts, knowledge)
        
        # Apply IRQN to enhance solution diversity
        solutions = self.apply_irqn(solutions, 'solution', knowledge)
        
        # Level 3: Implementation-level diversity (concrete schemes)
        logger.info("Generating implementation-level diversity...")
        implementations = self.explore_implementation_level(solutions, knowledge)
        
        # Apply IRQN to enhance implementation diversity
        implementations = self.apply_irqn(implementations, 'implementation', knowledge)
        
        # Calculate diversity scores
        diversity_scores = self._calculate_diversity_scores({
            'thoughts': thoughts,
            'solutions': solutions,
            'implementations': implementations
        })
        
        logger.info(f"Diversity scores: {diversity_scores}")
        
        result = {
            'success': True,
            'thought_level': thoughts,
            'solution_level': solutions,
            'implementation_level': implementations,
            'diversity_scores': diversity_scores
        }
        
        self._last_result = result
        return result
    
    def explore_thought_level(
        self,
        task_understanding: Dict[str, Any],
        knowledge: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Explore thought-level diversity.
        
        Generate different algorithmic approaches in natural language.
        """
        thoughts = []
        
        # Get problem summary
        problem_summary = task_understanding.get('problem_summary', '')
        known_approaches = task_understanding.get('algorithmic_approaches', [])
        
        # Get algorithmic patterns from knowledge base
        algo_patterns = knowledge.get('algorithmic', [])
        
        for i in range(self.num_thoughts):
            prompt = f"""{self.role_prompt}

## Problem
{problem_summary}

## Known Algorithmic Approaches
{json.dumps(known_approaches, indent=2) if known_approaches else "None identified yet"}

## Available Algorithmic Patterns
{self._format_patterns(algo_patterns)}

## Already Generated Approaches (AVOID SIMILARITY)
{self._format_existing_thoughts(thoughts)}

## Your Task
Generate a NEW and UNIQUE algorithmic approach for this problem.
This is approach #{i+1} of {self.num_thoughts}.

Your approach should:
1. Use a DIFFERENT algorithmic paradigm from existing approaches
2. Have distinct time/space complexity tradeoffs
3. Be clearly different in methodology

Describe your approach in natural language:
- Core algorithmic idea
- Why it's different from existing approaches
- Expected time and space complexity
- Any tradeoffs or limitations

Provide your response:"""
            
            response = self.generate(prompt, temperature=0.8)
            
            thoughts.append({
                'id': f'thought_{i}',
                'content': response,
                'type': 'algorithmic_approach',
                'meta': {
                    'iteration': i,
                    'paradigm': self._extract_paradigm(response)
                }
            })
        
        return thoughts
    
    def explore_solution_level(
        self,
        thoughts: List[Dict[str, Any]],
        knowledge: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Explore solution-level diversity.
        
        Generate pseudocode strategies for each thought.
        """
        solutions = []
        
        for thought in thoughts:
            for j in range(self.num_solutions):
                prompt = f"""{self.role_prompt}

## Algorithmic Approach
{thought['content']}

## Your Task
Convert this algorithmic approach into a detailed solution strategy.
This is variant #{j+1} of {self.num_solutions} for this approach.

Provide:
1. **Data Structures**: What data structures will be used and why
2. **Pseudocode**: High-level pseudocode outlining the solution
3. **Key Steps**: Numbered steps explaining the algorithm flow
4. **Complexity Analysis**: Time and space complexity
5. **Implementation Notes**: Important considerations

Make this variant DISTINCT from other variants by varying:
- Data structure choices
- Algorithm flow organization
- Edge case handling approach

Provide your solution strategy:"""
                
                response = self.generate(prompt, temperature=0.7)
                
                solutions.append({
                    'id': f'solution_{thought["id"]}_{j}',
                    'parent_thought': thought['id'],
                    'content': response,
                    'type': 'pseudocode_strategy',
                    'meta': {
                        'iteration': j,
                        'data_structures': self._extract_data_structures(response)
                    }
                })
        
        return solutions
    
    def explore_implementation_level(
        self,
        solutions: List[Dict[str, Any]],
        knowledge: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Explore implementation-level diversity.
        
        Generate concrete implementation schemes for each solution.
        """
        implementations = []
        
        variation_hints_list = [
            ['Use iterative approach', 'Prefer list comprehension'],
            ['Use recursive approach', 'Optimize with memoization'],
            ['Use built-in functions', 'Minimize custom code'],
            ['Use explicit loops', 'Avoid built-ins for control'],
        ]
        
        for solution in solutions:
            for k in range(self.num_implementations):
                variation_hints = variation_hints_list[k % len(variation_hints_list)]
                
                prompt = f"""{self.role_prompt}

## Solution Strategy
{solution['content']}

## Variation Guidelines
{', '.join(variation_hints)}

## Your Task
Plan the concrete Python implementation details.
This is implementation variant #{k+1}.

Specify:
1. **Control Flow**: Recursion vs iteration, loop types
2. **Data Structures**: Specific Python types (list, dict, set, etc.)
3. **Built-in Functions**: Which to use or avoid
4. **Variable Naming**: Key variable names and purposes
5. **Code Organization**: Overall structure

Following the variation guidelines:
{chr(10).join(f'- {hint}' for hint in variation_hints)}

Provide your implementation plan:"""
                
                response = self.generate(prompt, temperature=0.6)
                
                implementations.append({
                    'id': f'impl_{solution["id"]}_{k}',
                    'parent_solution': solution['id'],
                    'content': response,
                    'type': 'implementation_scheme',
                    'meta': {
                        'iteration': k,
                        'variant': self._classify_variant(response),
                        'variation_hints': variation_hints
                    }
                })
        
        return implementations
    
    def apply_irqn(
        self,
        outputs: List[Dict[str, Any]],
        level: str,
        knowledge: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply IRQN method to enhance diversity.
        
        Iterative Retention, Questioning and Negation:
        - Retain: Keep completely different outputs
        - Question: Refine partially similar outputs
        - Negate: Regenerate overly similar outputs
        """
        logger.info(f"Applying IRQN to {level} level outputs")
        
        diversity_evaluator = self.get_tool('diversity_evaluator')
        
        final_outputs = []
        pending_outputs = outputs.copy()
        history = []
        
        for iteration in range(self.max_iterations):
            logger.debug(f"IRQN iteration {iteration + 1}")
            
            current_batch = []
            
            for output in pending_outputs:
                # Probabilistic trigger for judgment
                if random.random() > self.p_qn1:
                    final_outputs.append(output)
                    logger.debug(f"Direct accept: {output['id']}")
                    continue
                
                # Evaluate similarity
                similarity = self._evaluate_similarity(
                    output,
                    history + final_outputs,
                    level,
                    diversity_evaluator
                )
                
                logger.debug(f"Similarity for {output['id']}: {similarity:.3f}")
                
                # Decide action based on similarity
                if similarity < self.theta_diff:
                    # Retain: Completely different
                    if random.random() < self.p_qn2:
                        # Further negate to produce more diversity
                        negated = self._negate_and_regenerate(output, knowledge, level)
                        current_batch.append(negated)
                        logger.debug(f"Retain + Negate: {output['id']}")
                    else:
                        final_outputs.append(output)
                        logger.debug(f"Retain: {output['id']}")
                
                elif similarity <= self.theta_ident:
                    # Question: Partially similar
                    questioned = self._question_and_refine(output, knowledge, level)
                    current_batch.append(questioned)
                    logger.debug(f"Question: {output['id']}")
                
                else:
                    # Negate: Too similar
                    negated = self._negate_and_regenerate(output, knowledge, level)
                    current_batch.append(negated)
                    logger.debug(f"Negate: {output['id']}")
            
            pending_outputs = current_batch
            history.extend(final_outputs)
            
            if not pending_outputs:
                logger.info(f"IRQN converged at iteration {iteration + 1}")
                break
        
        # Add any remaining pending outputs
        final_outputs.extend(pending_outputs)
        
        logger.info(f"IRQN completed: {len(final_outputs)} outputs retained")
        return final_outputs
    
    def _evaluate_similarity(
        self,
        output: Dict[str, Any],
        reference_set: List[Dict[str, Any]],
        level: str,
        diversity_evaluator
    ) -> float:
        """Evaluate output similarity with reference set."""
        if not reference_set:
            return 0.0
        
        if diversity_evaluator:
            similarities = []
            for ref in reference_set:
                sim = diversity_evaluator.calculate_semantic_similarity(
                    output['content'],
                    ref['content']
                )
                similarities.append(sim)
            return max(similarities)
        
        # Fallback: simple text comparison
        from difflib import SequenceMatcher
        similarities = [
            SequenceMatcher(None, output['content'], ref['content']).ratio()
            for ref in reference_set
        ]
        return max(similarities) if similarities else 0.0
    
    def _question_and_refine(
        self,
        output: Dict[str, Any],
        knowledge: Dict[str, Any],
        level: str
    ) -> Dict[str, Any]:
        """Question and refine a partially similar output."""
        prompt = f"""Current {level}-level output:
{output['content']}

This output shows partial similarity with existing solutions.

Please refine it by:
1. Identifying what makes it similar to existing approaches
2. Proposing alternative approaches that are more distinctive
3. Enhancing its uniqueness while maintaining feasibility

Provide the refined {level}-level solution:"""
        
        refined_content = self.generate(prompt, temperature=0.7)
        
        return {
            'id': f'{output["id"]}_refined',
            'parent': output['id'],
            'content': refined_content,
            'type': output['type'],
            'meta': {
                **output.get('meta', {}),
                'refined': True
            }
        }
    
    def _negate_and_regenerate(
        self,
        output: Dict[str, Any],
        knowledge: Dict[str, Any],
        level: str
    ) -> Dict[str, Any]:
        """Negate and regenerate an overly similar output."""
        level_specifics = {
            'thought': 'algorithmic paradigm',
            'solution': 'implementation strategy',
            'implementation': 'coding approach'
        }
        
        prompt = f"""Current {level}-level output:
{output['content']}

This approach has been used or is too similar to existing solutions.

Please generate a COMPLETELY DIFFERENT solution that:
1. Uses a different {level_specifics.get(level, 'approach')}
2. Employs different data structures and control flow
3. Takes a contrasting perspective to solve the problem

IMPORTANT: Avoid any similarity to the current output.

Provide the new {level}-level solution:"""
        
        regenerated_content = self.generate(prompt, temperature=0.9)
        
        return {
            'id': f'{output["id"]}_negated',
            'parent': output['id'],
            'content': regenerated_content,
            'type': output['type'],
            'meta': {
                **output.get('meta', {}),
                'negated': True,
                'original_rejected': True
            }
        }
    
    def _calculate_diversity_scores(
        self,
        all_outputs: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Calculate diversity scores for all levels."""
        diversity_evaluator = self.get_tool('diversity_evaluator')
        scores = {}
        
        for level, outputs in all_outputs.items():
            if not outputs:
                scores[level] = 0.0
                continue
            
            contents = [o['content'] for o in outputs]
            
            if diversity_evaluator:
                # Calculate MBCS
                mbcs = diversity_evaluator.compute_mbcs(contents)
                scores[f'{level}_mbcs'] = mbcs
                scores[f'{level}_semantic_diversity'] = 1.0 - mbcs
            else:
                # Fallback: simple average similarity
                from difflib import SequenceMatcher
                n = len(contents)
                if n < 2:
                    scores[f'{level}_semantic_diversity'] = 1.0
                else:
                    total_sim = 0
                    pairs = 0
                    for i in range(n):
                        for j in range(i + 1, n):
                            total_sim += SequenceMatcher(None, contents[i], contents[j]).ratio()
                            pairs += 1
                    avg_sim = total_sim / pairs if pairs > 0 else 0
                    scores[f'{level}_semantic_diversity'] = 1.0 - avg_sim
        
        return scores
    
    def _format_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Format algorithmic patterns for prompt."""
        if not patterns:
            return "No patterns available"
        
        lines = []
        for p in patterns[:5]:
            name = p.get('name', 'Unknown')
            desc = p.get('description', '')
            lines.append(f"- **{name}**: {desc}")
        
        return "\n".join(lines)
    
    def _format_existing_thoughts(self, thoughts: List[Dict[str, Any]]) -> str:
        """Format existing thoughts for prompt."""
        if not thoughts:
            return "None yet"
        
        lines = []
        for t in thoughts:
            content = t['content'][:200] + "..." if len(t['content']) > 200 else t['content']
            lines.append(f"- Approach {t['id']}: {content}")
        
        return "\n".join(lines)
    
    def _extract_paradigm(self, thought: str) -> str:
        """Extract algorithmic paradigm from thought."""
        paradigms = {
            'dynamic_programming': ['dynamic programming', 'dp', 'memoization', 'tabulation'],
            'greedy': ['greedy', 'locally optimal'],
            'divide_conquer': ['divide and conquer', 'divide-and-conquer', 'split'],
            'backtracking': ['backtracking', 'backtrack'],
            'graph': ['graph', 'bfs', 'dfs', 'dijkstra'],
            'two_pointers': ['two pointers', 'two-pointer'],
            'sliding_window': ['sliding window'],
            'binary_search': ['binary search'],
            'sorting': ['sort', 'sorting'],
        }
        
        thought_lower = thought.lower()
        for paradigm, keywords in paradigms.items():
            if any(kw in thought_lower for kw in keywords):
                return paradigm
        
        return 'other'
    
    def _extract_data_structures(self, solution: str) -> List[str]:
        """Extract data structures from solution."""
        ds_keywords = {
            'array': ['array', 'list'],
            'hash_table': ['dict', 'hash', 'map', 'dictionary'],
            'tree': ['tree', 'bst', 'binary tree'],
            'graph': ['graph', 'node', 'edge'],
            'queue': ['queue', 'deque'],
            'stack': ['stack'],
            'heap': ['heap', 'priority queue'],
            'set': ['set']
        }
        
        found = []
        solution_lower = solution.lower()
        
        for ds_type, keywords in ds_keywords.items():
            if any(kw in solution_lower for kw in keywords):
                found.append(ds_type)
        
        return found
    
    def _classify_variant(self, implementation: str) -> str:
        """Classify implementation variant."""
        impl_lower = implementation.lower()
        
        if 'recursive' in impl_lower or 'recursion' in impl_lower:
            return 'recursive'
        elif 'iterative' in impl_lower or 'loop' in impl_lower or 'for ' in impl_lower:
            return 'iterative'
        else:
            return 'mixed'

