"""
Tests for Diversity Enhancing Agent
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.diversity_agent import DiversityEnhancingAgent


class MockLLMClient:
    """Mock LLM client."""
    
    def generate(self, prompt, temperature=0.7):
        mock_response = MagicMock()
        mock_response.content = "Mock generated response for diversity"
        return mock_response


class MockDiversityEvaluator:
    """Mock diversity evaluator."""
    
    def calculate_semantic_similarity(self, text1, text2):
        return 0.3
    
    def compute_mbcs(self, code_list):
        return 0.4
    
    def compute_sdp(self, code_list, llm_client=None):
        return 0.7


class MockConfig:
    """Mock configuration."""
    num_thoughts = 3
    num_solutions = 2
    num_implementations = 2
    p_qn1 = 0.7
    p_qn2 = 0.3
    max_iterations = 3
    theta_diff = 0.3
    theta_ident = 0.7


class TestDiversityEnhancingAgent(unittest.TestCase):
    """Test cases for DiversityEnhancingAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMClient()
        self.mock_evaluator = MockDiversityEvaluator()
        self.mock_knowledge = Mock()
        self.mock_prompt_gen = Mock()
        
        self.agent = DiversityEnhancingAgent(
            llm_client=self.mock_llm,
            diversity_evaluator=self.mock_evaluator,
            knowledge_search=self.mock_knowledge,
            dynamic_prompt_generator=self.mock_prompt_gen,
            config=MockConfig()
        )
    
    def test_explore_thought_level(self):
        """Test thought-level exploration."""
        task_info = {
            'problem_summary': 'Find the longest palindromic substring'
        }
        knowledge = {'algorithmic': []}
        
        thoughts = self.agent.explore_thought_level(task_info, knowledge)
        
        self.assertEqual(len(thoughts), 3)  # num_thoughts = 3
        self.assertEqual(thoughts[0]['type'], 'algorithmic_approach')
        self.assertIn('paradigm', thoughts[0]['meta'])
    
    def test_explore_solution_level(self):
        """Test solution-level exploration."""
        thoughts = [
            {'id': 'thought_0', 'content': 'Use dynamic programming approach'},
            {'id': 'thought_1', 'content': 'Use two pointers approach'},
        ]
        knowledge = {}
        
        solutions = self.agent.explore_solution_level(thoughts, knowledge)
        
        # 2 thoughts * 2 solutions each = 4
        self.assertEqual(len(solutions), 4)
        self.assertEqual(solutions[0]['type'], 'pseudocode_strategy')
    
    def test_explore_implementation_level(self):
        """Test implementation-level exploration."""
        solutions = [
            {'id': 'solution_0', 'content': 'Solution 1'},
            {'id': 'solution_1', 'content': 'Solution 2'},
        ]
        knowledge = {}
        
        implementations = self.agent.explore_implementation_level(solutions, knowledge)
        
        # 2 solutions * 2 implementations each = 4
        self.assertEqual(len(implementations), 4)
        self.assertEqual(implementations[0]['type'], 'implementation_scheme')
    
    def test_extract_paradigm(self):
        """Test paradigm extraction from thought."""
        dp_thought = "Use dynamic programming with memoization"
        greedy_thought = "Use a greedy approach to select elements"
        
        self.assertEqual(
            self.agent._extract_paradigm(dp_thought),
            'dynamic_programming'
        )
        self.assertEqual(
            self.agent._extract_paradigm(greedy_thought),
            'greedy'
        )
    
    def test_extract_data_structures(self):
        """Test data structure extraction."""
        solution = "Use a hash table to store values and an array for results"
        
        ds = self.agent._extract_data_structures(solution)
        
        self.assertIn('hash_table', ds)
        self.assertIn('array', ds)
    
    def test_classify_variant(self):
        """Test variant classification."""
        recursive = "Use recursive approach with base case"
        iterative = "Use iterative loop through the array"
        
        self.assertEqual(
            self.agent._classify_variant(recursive),
            'recursive'
        )
        self.assertEqual(
            self.agent._classify_variant(iterative),
            'iterative'
        )
    
    def test_apply_irqn_retain(self):
        """Test IRQN retain logic."""
        outputs = [
            {'id': 'test_1', 'content': 'Approach A', 'type': 'thought'},
            {'id': 'test_2', 'content': 'Approach B', 'type': 'thought'},
        ]
        knowledge = {}
        
        with patch('random.random', return_value=0.9):  # > p_qn1, direct accept
            result = self.agent.apply_irqn(outputs, 'thought', knowledge)
        
        # Should directly accept all
        self.assertEqual(len(result), 2)


class TestIRQN(unittest.TestCase):
    """Test cases for IRQN method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMClient()
        self.agent = DiversityEnhancingAgent(
            llm_client=self.mock_llm,
            config=MockConfig()
        )
    
    def test_question_and_refine(self):
        """Test question and refine operation."""
        output = {
            'id': 'test',
            'content': 'Original content',
            'type': 'thought',
            'meta': {}
        }
        
        refined = self.agent._question_and_refine(output, {}, 'thought')
        
        self.assertIn('_refined', refined['id'])
        self.assertTrue(refined['meta'].get('refined'))
    
    def test_negate_and_regenerate(self):
        """Test negate and regenerate operation."""
        output = {
            'id': 'test',
            'content': 'Original content',
            'type': 'thought',
            'meta': {}
        }
        
        negated = self.agent._negate_and_regenerate(output, {}, 'thought')
        
        self.assertIn('_negated', negated['id'])
        self.assertTrue(negated['meta'].get('negated'))


if __name__ == '__main__':
    unittest.main()

