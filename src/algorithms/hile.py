"""
DeQoG HILE Algorithm

Hierarchical Isolation and Local Expansion algorithm for
generating diverse solutions at multiple levels.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from ..utils.logger import get_logger

logger = get_logger("hile")


@dataclass
class LevelOutput:
    """Output from a single level of HILE."""
    level: str
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HILEAlgorithm:
    """
    Hierarchical Isolation and Local Expansion (HILE) Algorithm.
    
    Generates diverse solutions through three hierarchical levels:
    1. Thought Level (L_Thought): Algorithmic approaches in natural language
    2. Solution Level (L_Solution): Pseudocode-level strategies
    3. Implementation Level (L_Implementation): Concrete implementation schemes
    
    Key principles:
    - Hierarchical Isolation: Each level explores independently
    - Local Expansion: Deep exploration within each level
    - Cross-level Independence: Different levels produce orthogonal diversity
    """
    
    def __init__(
        self,
        llm_client,
        knowledge_bases: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the HILE algorithm.
        
        Args:
            llm_client: LLM client for generation
            knowledge_bases: Dictionary of knowledge bases
            config: Algorithm configuration
        """
        self.llm_client = llm_client
        self.knowledge_bases = knowledge_bases or {}
        
        # Configuration
        config = config or {}
        self.num_thoughts = config.get('num_thoughts', 5)
        self.num_solutions = config.get('num_solutions', 3)
        self.num_implementations = config.get('num_implementations', 2)
        
        # Level definitions
        self.levels = {
            'thought': 'L_Thought',
            'solution': 'L_Solution',
            'implementation': 'L_Implementation'
        }
    
    def execute(
        self,
        task_info: Dict[str, Any],
        n_versions: int = 5
    ) -> Dict[str, LevelOutput]:
        """
        Execute the HILE algorithm.
        
        Args:
            task_info: Task information from understanding phase
            n_versions: Target number of final versions
            
        Returns:
            Dictionary of level outputs
        """
        logger.info("Starting HILE algorithm execution")
        
        results = {}
        
        # Level 1: Thought-level diversity
        logger.info(f"Exploring thought level ({self.num_thoughts} targets)")
        results['thought'] = self.explore_thought_level(task_info, n_versions)
        
        # Level 2: Solution-level diversity
        logger.info(f"Exploring solution level ({self.num_solutions} per thought)")
        results['solution'] = self.explore_solution_level(
            results['thought'].outputs, n_versions
        )
        
        # Level 3: Implementation-level diversity
        logger.info(f"Exploring implementation level ({self.num_implementations} per solution)")
        results['implementation'] = self.explore_implementation_level(
            results['solution'].outputs, n_versions
        )
        
        logger.info("HILE algorithm execution completed")
        return results
    
    def explore_thought_level(
        self,
        task_info: Dict[str, Any],
        n: int
    ) -> LevelOutput:
        """
        Explore thought-level diversity.
        
        Generate different algorithmic approaches as natural language descriptions.
        This level focuses on high-level strategy diversity.
        
        Args:
            task_info: Task information
            n: Target number of thoughts
            
        Returns:
            LevelOutput with thought-level outputs
        """
        outputs = []
        problem_summary = task_info.get('problem_summary', '')
        
        # Get algorithmic patterns from knowledge base
        algo_patterns = self.knowledge_bases.get('algorithmic', {})
        
        # Define paradigm categories to ensure diversity
        paradigm_categories = [
            ('dynamic_programming', 'Use dynamic programming with optimal substructure'),
            ('greedy', 'Use a greedy approach making locally optimal choices'),
            ('divide_conquer', 'Use divide and conquer strategy'),
            ('backtracking', 'Use backtracking with constraint checking'),
            ('two_pointers', 'Use two-pointer technique'),
            ('sliding_window', 'Use sliding window approach'),
            ('graph_based', 'Model as a graph problem'),
            ('mathematical', 'Use mathematical formula or pattern'),
        ]
        
        for i in range(min(n, self.num_thoughts)):
            # Select paradigm hint for diversity
            paradigm = paradigm_categories[i % len(paradigm_categories)]
            
            prompt = f"""Generate an algorithmic approach for the following problem:

Problem: {problem_summary}

Paradigm hint: {paradigm[1]}

Create a detailed algorithmic approach that:
1. Explains the core idea in natural language
2. Describes the main steps of the algorithm
3. Analyzes time and space complexity
4. Identifies key data structures needed

Provide your approach:"""
            
            response = self._generate(prompt, temperature=0.8)
            
            outputs.append({
                'id': f'thought_{i}',
                'content': response,
                'paradigm': paradigm[0],
                'level': 'thought'
            })
        
        return LevelOutput(
            level='thought',
            outputs=outputs,
            metadata={
                'total_generated': len(outputs),
                'paradigms_used': list(set(o['paradigm'] for o in outputs))
            }
        )
    
    def explore_solution_level(
        self,
        thoughts: List[Dict[str, Any]],
        n: int
    ) -> LevelOutput:
        """
        Explore solution-level diversity.
        
        Generate pseudocode-level strategies for each thought.
        This level focuses on structural and organizational diversity.
        
        Args:
            thoughts: Thought-level outputs
            n: Target number of solutions
            
        Returns:
            LevelOutput with solution-level outputs
        """
        outputs = []
        
        for thought in thoughts:
            for j in range(self.num_solutions):
                # Variation strategies for solutions
                variation_strategies = [
                    'Optimize for time complexity over space',
                    'Optimize for space complexity over time',
                    'Prioritize code readability and maintainability',
                ]
                variation = variation_strategies[j % len(variation_strategies)]
                
                prompt = f"""Convert this algorithmic approach into a detailed solution strategy:

Approach:
{thought['content']}

Variation focus: {variation}

Provide:
1. Data structures with justification
2. Detailed pseudocode
3. Step-by-step algorithm flow
4. Complexity analysis
5. Edge case handling strategy

Solution strategy:"""
                
                response = self._generate(prompt, temperature=0.7)
                
                outputs.append({
                    'id': f'solution_{thought["id"]}_{j}',
                    'parent_id': thought['id'],
                    'content': response,
                    'variation': variation,
                    'level': 'solution'
                })
        
        return LevelOutput(
            level='solution',
            outputs=outputs,
            metadata={
                'total_generated': len(outputs),
                'parent_thoughts': len(thoughts)
            }
        )
    
    def explore_implementation_level(
        self,
        solutions: List[Dict[str, Any]],
        n: int
    ) -> LevelOutput:
        """
        Explore implementation-level diversity.
        
        Generate concrete implementation schemes for each solution.
        This level focuses on coding style and technique diversity.
        
        Args:
            solutions: Solution-level outputs
            n: Target number of implementations
            
        Returns:
            LevelOutput with implementation-level outputs
        """
        outputs = []
        
        for solution in solutions:
            for k in range(self.num_implementations):
                # Implementation style hints
                style_hints = [
                    ('iterative', ['Use for/while loops', 'Avoid recursion']),
                    ('recursive', ['Use recursion', 'Add memoization if needed']),
                    ('functional', ['Use list comprehensions', 'Prefer map/filter/reduce']),
                    ('pythonic', ['Use Python idioms', 'Leverage built-in functions']),
                ]
                style = style_hints[k % len(style_hints)]
                
                prompt = f"""Create an implementation plan for this solution:

Solution:
{solution['content']}

Style: {style[0]}
Guidelines: {', '.join(style[1])}

Specify:
1. Control flow pattern (loops, recursion)
2. Specific Python data types
3. Built-in functions to use
4. Variable naming conventions
5. Code organization

Implementation plan:"""
                
                response = self._generate(prompt, temperature=0.6)
                
                outputs.append({
                    'id': f'impl_{solution["id"]}_{k}',
                    'parent_id': solution['id'],
                    'content': response,
                    'style': style[0],
                    'guidelines': style[1],
                    'level': 'implementation'
                })
        
        return LevelOutput(
            level='implementation',
            outputs=outputs,
            metadata={
                'total_generated': len(outputs),
                'parent_solutions': len(solutions)
            }
        )
    
    def _generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using LLM."""
        response = self.llm_client.generate(prompt, temperature=temperature)
        return response.content if hasattr(response, 'content') else str(response)
    
    def get_all_outputs(
        self,
        results: Dict[str, LevelOutput]
    ) -> List[Dict[str, Any]]:
        """
        Get all outputs from all levels.
        
        Args:
            results: HILE execution results
            
        Returns:
            Flat list of all outputs
        """
        all_outputs = []
        for level_name, level_output in results.items():
            all_outputs.extend(level_output.outputs)
        return all_outputs
    
    def get_implementation_count(
        self,
        results: Dict[str, LevelOutput]
    ) -> int:
        """Get total number of implementation plans."""
        return len(results.get('implementation', LevelOutput('implementation')).outputs)

