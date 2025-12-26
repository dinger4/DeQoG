"""
DeQoG Knowledge Search

Searches and retrieves relevant knowledge from predefined knowledge bases
for algorithmic patterns, implementation techniques, and fault tolerance strategies.
"""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from difflib import SequenceMatcher

from .base_tool import BaseTool
from ..utils.logger import get_logger

logger = get_logger("knowledge_search")


class KnowledgeSearch(BaseTool):
    """
    Knowledge Search Tool.
    
    Retrieves relevant information from knowledge bases:
    - K_algo: Algorithmic patterns
    - K_impl: Implementation techniques
    - K_f-t: Fault tolerance strategies
    """
    
    def __init__(
        self,
        knowledge_base_dir: Optional[str] = None,
        llm_client=None
    ):
        """
        Initialize the knowledge search tool.
        
        Args:
            knowledge_base_dir: Directory containing knowledge base files
            llm_client: LLM client for semantic search
        """
        super().__init__(
            name="knowledge_search",
            description="Searches knowledge bases for relevant information"
        )
        
        self.knowledge_base_dir = Path(knowledge_base_dir) if knowledge_base_dir else None
        self.llm_client = llm_client
        
        # Load knowledge bases
        self.knowledge_bases = {
            'algorithmic': self._load_algorithmic_patterns(),
            'implementation': self._load_implementation_techniques(),
            'fault_tolerance': self._load_fault_tolerance_strategies()
        }
    
    def _load_algorithmic_patterns(self) -> Dict[str, Any]:
        """Load algorithmic patterns knowledge base."""
        if self.knowledge_base_dir:
            kb_file = self.knowledge_base_dir / "algorithmic_patterns.json"
            if kb_file.exists():
                with open(kb_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # Default algorithmic patterns
        return {
            "dynamic_programming": {
                "description": "Break problem into overlapping subproblems with optimal substructure",
                "keywords": ["optimal", "subproblem", "memoization", "tabulation", "sequence"],
                "use_cases": ["fibonacci", "knapsack", "longest common subsequence", "edit distance"],
                "complexity": {"time": "O(n*m) typically", "space": "O(n*m) or O(n)"},
                "variants": ["top-down with memoization", "bottom-up tabulation", "space-optimized"]
            },
            "divide_and_conquer": {
                "description": "Divide problem into independent subproblems, solve, and combine",
                "keywords": ["divide", "merge", "recursive", "binary", "split"],
                "use_cases": ["merge sort", "quick sort", "binary search", "closest pair"],
                "complexity": {"time": "O(n log n) typically", "space": "O(log n) to O(n)"},
                "variants": ["recursive", "iterative", "parallel"]
            },
            "greedy": {
                "description": "Make locally optimal choices at each step",
                "keywords": ["optimal", "choice", "selection", "minimum", "maximum"],
                "use_cases": ["activity selection", "huffman coding", "minimum spanning tree"],
                "complexity": {"time": "O(n log n) typically", "space": "O(1) to O(n)"},
                "variants": ["sorting-based", "heap-based", "priority queue"]
            },
            "backtracking": {
                "description": "Explore all possibilities with pruning for constraint satisfaction",
                "keywords": ["constraint", "permutation", "combination", "search", "prune"],
                "use_cases": ["n-queens", "sudoku", "subset sum", "graph coloring"],
                "complexity": {"time": "Exponential worst case", "space": "O(n) for recursion"},
                "variants": ["recursive", "iterative with stack", "with constraint propagation"]
            },
            "two_pointers": {
                "description": "Use two pointers to iterate through data structure",
                "keywords": ["pair", "sum", "window", "sorted", "linear"],
                "use_cases": ["two sum", "container with most water", "remove duplicates"],
                "complexity": {"time": "O(n)", "space": "O(1)"},
                "variants": ["opposite ends", "same direction", "fast-slow pointers"]
            },
            "sliding_window": {
                "description": "Maintain a window over data for streaming computations",
                "keywords": ["window", "substring", "subarray", "continuous", "stream"],
                "use_cases": ["maximum sum subarray", "longest substring", "minimum window"],
                "complexity": {"time": "O(n)", "space": "O(k) for window size k"},
                "variants": ["fixed size", "variable size", "with auxiliary data structure"]
            },
            "graph_traversal": {
                "description": "Systematically visit nodes in a graph",
                "keywords": ["graph", "node", "edge", "path", "connected"],
                "use_cases": ["shortest path", "cycle detection", "topological sort"],
                "complexity": {"time": "O(V+E)", "space": "O(V)"},
                "variants": ["BFS", "DFS", "Dijkstra", "Bellman-Ford", "A*"]
            },
            "binary_search": {
                "description": "Search in sorted data by repeatedly halving search space",
                "keywords": ["sorted", "search", "half", "logarithmic", "monotonic"],
                "use_cases": ["find element", "find boundary", "minimize/maximize"],
                "complexity": {"time": "O(log n)", "space": "O(1)"},
                "variants": ["standard", "lower bound", "upper bound", "on answer"]
            }
        }
    
    def _load_implementation_techniques(self) -> Dict[str, Any]:
        """Load implementation techniques knowledge base."""
        if self.knowledge_base_dir:
            kb_file = self.knowledge_base_dir / "implementation_techniques.json"
            if kb_file.exists():
                with open(kb_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        return {
            "data_structures": {
                "list": {
                    "description": "Dynamic array with O(1) append",
                    "operations": {"append": "O(1)", "access": "O(1)", "insert": "O(n)"},
                    "use_when": "Need ordered collection with fast append and random access"
                },
                "dict": {
                    "description": "Hash table with O(1) average operations",
                    "operations": {"get": "O(1)", "set": "O(1)", "delete": "O(1)"},
                    "use_when": "Need fast key-value lookups"
                },
                "set": {
                    "description": "Hash set for unique elements",
                    "operations": {"add": "O(1)", "remove": "O(1)", "check": "O(1)"},
                    "use_when": "Need unique elements with fast membership testing"
                },
                "deque": {
                    "description": "Double-ended queue",
                    "operations": {"append": "O(1)", "popleft": "O(1)", "appendleft": "O(1)"},
                    "use_when": "Need fast operations at both ends"
                },
                "heap": {
                    "description": "Priority queue implemented as heap",
                    "operations": {"push": "O(log n)", "pop": "O(log n)", "peek": "O(1)"},
                    "use_when": "Need to repeatedly get min/max element"
                }
            },
            "control_patterns": {
                "iterative": {
                    "description": "Use loops for repetition",
                    "pros": ["No stack overflow", "Usually faster", "Less memory"],
                    "cons": ["May be less intuitive", "State management can be complex"]
                },
                "recursive": {
                    "description": "Function calls itself",
                    "pros": ["Often more intuitive", "Natural for tree structures"],
                    "cons": ["Stack overflow risk", "Function call overhead"]
                },
                "memoization": {
                    "description": "Cache function results",
                    "pros": ["Avoids redundant computation"],
                    "cons": ["Memory overhead", "Cache management"]
                }
            },
            "optimization_techniques": {
                "early_termination": "Return as soon as answer is found",
                "lazy_evaluation": "Compute values only when needed",
                "preprocessing": "Precompute data to speed up queries",
                "space_time_tradeoff": "Use more memory for faster time"
            }
        }
    
    def _load_fault_tolerance_strategies(self) -> Dict[str, Any]:
        """Load fault tolerance strategies knowledge base."""
        if self.knowledge_base_dir:
            kb_file = self.knowledge_base_dir / "fault_tolerance_strategies.json"
            if kb_file.exists():
                with open(kb_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        return {
            "diversity_strategies": {
                "algorithmic_diversity": {
                    "description": "Use different algorithms for same problem",
                    "examples": ["DP vs greedy", "BFS vs DFS", "iterative vs recursive"],
                    "effectiveness": "High - different algorithms have different failure modes"
                },
                "implementation_diversity": {
                    "description": "Different code implementations of same algorithm",
                    "examples": ["Different data structures", "Different loop constructs"],
                    "effectiveness": "Medium - some implementation bugs may be shared"
                },
                "data_representation_diversity": {
                    "description": "Different ways to represent same data",
                    "examples": ["Array vs linked list", "Adjacency matrix vs list"],
                    "effectiveness": "Medium - affects performance-related failures"
                }
            },
            "voting_mechanisms": {
                "majority_voting": {
                    "description": "Accept result agreed by majority of versions",
                    "requirement": "N >= 3 versions, tolerates (N-1)/2 failures"
                },
                "consensus_voting": {
                    "description": "All versions must agree",
                    "requirement": "Higher reliability but stricter"
                },
                "weighted_voting": {
                    "description": "Votes weighted by version reliability",
                    "requirement": "Historical reliability data needed"
                }
            },
            "error_handling": {
                "input_validation": "Validate all inputs before processing",
                "boundary_checking": "Check array bounds and numeric limits",
                "exception_handling": "Catch and handle expected exceptions",
                "graceful_degradation": "Provide partial results when full computation fails"
            }
        }
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search knowledge bases.
        
        Args:
            params: Dictionary containing:
                - query: Search query string
                - knowledge_type: Type of knowledge ('algorithmic', 'implementation', 'fault_tolerance', 'all')
                - top_k: Number of results to return
                
        Returns:
            Dictionary with search results
        """
        query = params.get('query', '')
        knowledge_type = params.get('knowledge_type', 'all')
        top_k = params.get('top_k', 5)
        
        results = {}
        
        if knowledge_type in ['algorithmic', 'all']:
            results['algorithmic'] = self.search_algorithmic_patterns(query, top_k)
        
        if knowledge_type in ['implementation', 'all']:
            results['implementation'] = self.search_implementation_techniques(query, top_k)
        
        if knowledge_type in ['fault_tolerance', 'all']:
            results['fault_tolerance'] = self.search_fault_tolerance(query, top_k)
        
        return results
    
    def search_algorithmic_patterns(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search algorithmic patterns knowledge base.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of matching patterns
        """
        kb = self.knowledge_bases['algorithmic']
        results = []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for pattern_name, pattern_info in kb.items():
            score = 0
            
            # Check name match
            if pattern_name.replace('_', ' ') in query_lower:
                score += 10
            
            # Check keyword matches
            keywords = pattern_info.get('keywords', [])
            matched_keywords = [kw for kw in keywords if kw in query_lower]
            score += len(matched_keywords) * 2
            
            # Check use case matches
            use_cases = pattern_info.get('use_cases', [])
            for use_case in use_cases:
                if any(word in use_case.lower() for word in query_words):
                    score += 1
            
            # Text similarity
            description = pattern_info.get('description', '')
            sim = SequenceMatcher(None, query_lower, description.lower()).ratio()
            score += sim * 5
            
            if score > 0:
                results.append({
                    'name': pattern_name,
                    'score': score,
                    'matched_keywords': matched_keywords,
                    **pattern_info
                })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def search_implementation_techniques(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search implementation techniques."""
        kb = self.knowledge_bases['implementation']
        results = []
        
        query_lower = query.lower()
        
        # Search data structures
        for ds_name, ds_info in kb.get('data_structures', {}).items():
            if ds_name in query_lower or any(
                word in query_lower for word in ds_info.get('use_when', '').lower().split()
            ):
                results.append({
                    'category': 'data_structure',
                    'name': ds_name,
                    **ds_info
                })
        
        # Search control patterns
        for pattern_name, pattern_info in kb.get('control_patterns', {}).items():
            if pattern_name in query_lower:
                results.append({
                    'category': 'control_pattern',
                    'name': pattern_name,
                    **pattern_info
                })
        
        return results[:top_k]
    
    def search_fault_tolerance(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search fault tolerance strategies."""
        kb = self.knowledge_bases['fault_tolerance']
        results = []
        
        query_lower = query.lower()
        
        for category, strategies in kb.items():
            for strategy_name, strategy_info in strategies.items():
                if isinstance(strategy_info, dict):
                    desc = strategy_info.get('description', '')
                    if strategy_name in query_lower or query_lower in desc.lower():
                        results.append({
                            'category': category,
                            'name': strategy_name,
                            **strategy_info
                        })
        
        return results[:top_k]
    
    def get_pattern_by_name(self, pattern_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific algorithmic pattern by name.
        
        Args:
            pattern_name: Name of the pattern
            
        Returns:
            Pattern information or None
        """
        return self.knowledge_bases['algorithmic'].get(pattern_name)
    
    def get_all_patterns(self) -> List[str]:
        """Get all available algorithmic pattern names."""
        return list(self.knowledge_bases['algorithmic'].keys())
    
    def suggest_diverse_approaches(
        self,
        query: str,
        n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Suggest diverse algorithmic approaches for a problem.
        
        Args:
            query: Problem description
            n: Number of approaches to suggest
            
        Returns:
            List of diverse approach suggestions
        """
        patterns = self.search_algorithmic_patterns(query, top_k=n * 2)
        
        # Ensure diversity by selecting patterns from different categories
        selected = []
        used_categories = set()
        
        for pattern in patterns:
            # Determine category based on pattern characteristics
            category = self._categorize_pattern(pattern['name'])
            
            if category not in used_categories or len(selected) < n:
                selected.append({
                    'pattern': pattern['name'],
                    'description': pattern.get('description', ''),
                    'suggested_approach': self._generate_approach_hint(pattern)
                })
                used_categories.add(category)
            
            if len(selected) >= n:
                break
        
        return selected
    
    def _categorize_pattern(self, pattern_name: str) -> str:
        """Categorize a pattern."""
        categories = {
            'optimization': ['dynamic_programming', 'greedy'],
            'search': ['binary_search', 'graph_traversal', 'backtracking'],
            'divide': ['divide_and_conquer'],
            'pointer': ['two_pointers', 'sliding_window']
        }
        
        for category, patterns in categories.items():
            if pattern_name in patterns:
                return category
        
        return 'other'
    
    def _generate_approach_hint(self, pattern: Dict[str, Any]) -> str:
        """Generate a hint for using the pattern."""
        name = pattern.get('name', '')
        variants = pattern.get('variants', [])
        
        hint = f"Consider using {name.replace('_', ' ')}"
        if variants:
            hint += f" with {variants[0]} approach"
        
        return hint

