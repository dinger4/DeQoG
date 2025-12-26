"""
Tests for DeQoG Tools
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools.code_interpreter import CodeInterpreter
from tools.test_executor import TestExecutor
from tools.diversity_evaluator import DiversityEvaluator


class TestCodeInterpreter(unittest.TestCase):
    """Test cases for CodeInterpreter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interpreter = CodeInterpreter(timeout=5, sandbox_enabled=True)
    
    def test_validate_syntax_valid(self):
        """Test syntax validation with valid code."""
        code = """
def add(a, b):
    return a + b
"""
        result = self.interpreter.validate_syntax(code)
        
        self.assertTrue(result['valid'])
        self.assertIsNone(result['error'])
    
    def test_validate_syntax_invalid(self):
        """Test syntax validation with invalid code."""
        code = """
def add(a, b)
    return a + b
"""
        result = self.interpreter.validate_syntax(code)
        
        self.assertFalse(result['valid'])
        self.assertIsNotNone(result['error'])
    
    def test_execute_simple_code(self):
        """Test execution of simple code."""
        code = "print(2 + 2)"
        
        result = self.interpreter.execute({'code': code})
        
        self.assertTrue(result['success'])
        self.assertEqual(result['output'], '4')
    
    def test_execute_with_error(self):
        """Test execution with runtime error."""
        code = "print(1/0)"
        
        result = self.interpreter.execute({'code': code})
        
        self.assertFalse(result['success'])
        self.assertIn('ZeroDivision', result.get('error', ''))
    
    def test_check_code_safety(self):
        """Test safety checking."""
        unsafe_code = "import os; os.system('rm -rf /')"
        safe_code = "x = 1 + 2"
        
        unsafe_result = self.interpreter.check_code_safety(unsafe_code)
        safe_result = self.interpreter.check_code_safety(safe_code)
        
        self.assertFalse(unsafe_result['safe'])
        self.assertTrue(safe_result['safe'])


class TestTestExecutor(unittest.TestCase):
    """Test cases for TestExecutor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.executor = TestExecutor(parallel=False, timeout_per_test=5)
    
    def test_execute_passing_tests(self):
        """Test execution with passing test cases."""
        code = """
def add(a, b):
    return a + b

result = add(*test_input)
print(repr(result))
"""
        # This is a simplified test - actual implementation would need proper setup
        test_cases = [
            {'input': (1, 2), 'expected_output': 3},
        ]
        
        # Test the executor structure
        self.assertIsNotNone(self.executor)
        self.assertEqual(self.executor.timeout_per_test, 5)
    
    def test_result_aggregation(self):
        """Test result aggregation."""
        # Mock results for testing
        from tools.test_executor import TestResult
        
        results = [
            TestResult(
                test_case={'input': 1, 'expected_output': 1},
                passed=True,
                actual_output=1,
                expected_output=1
            ),
            TestResult(
                test_case={'input': 2, 'expected_output': 2},
                passed=True,
                actual_output=2,
                expected_output=2
            ),
            TestResult(
                test_case={'input': 3, 'expected_output': 4},
                passed=False,
                actual_output=3,
                expected_output=4
            ),
        ]
        
        aggregated = self.executor._aggregate_results(results)
        
        self.assertEqual(aggregated['total'], 3)
        self.assertEqual(aggregated['passed'], 2)
        self.assertEqual(aggregated['failed'], 1)
        self.assertAlmostEqual(aggregated['pass_rate'], 2/3)


class TestDiversityEvaluator(unittest.TestCase):
    """Test cases for DiversityEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = DiversityEvaluator(cache_embeddings=True)
    
    def test_compute_mbcs_single(self):
        """Test MBCS with single code."""
        code_list = ["def foo(): return 1"]
        
        mbcs = self.evaluator.compute_mbcs(code_list)
        
        self.assertEqual(mbcs, 0.0)
    
    def test_compute_mbcs_identical(self):
        """Test MBCS with identical codes."""
        code = "def foo(): return 1"
        code_list = [code, code]
        
        mbcs = self.evaluator.compute_mbcs(code_list)
        
        # Identical codes should have high similarity
        self.assertGreater(mbcs, 0.9)
    
    def test_compute_mbcs_different(self):
        """Test MBCS with different codes."""
        code_list = [
            "def iterative_sum(n): return sum(range(n))",
            "def recursive_factorial(n): return 1 if n <= 1 else n * recursive_factorial(n-1)"
        ]
        
        mbcs = self.evaluator.compute_mbcs(code_list)
        
        # Different codes should have lower similarity
        self.assertLess(mbcs, 0.95)
    
    def test_is_diverse_enough(self):
        """Test diversity threshold checking."""
        new_code = "def bar(): return 2"
        existing = ["def foo(): return 1"]
        
        is_diverse, similarity = self.evaluator.is_diverse_enough(
            new_code,
            existing,
            threshold=0.95
        )
        
        self.assertIsInstance(is_diverse, bool)
        self.assertIsInstance(similarity, float)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add something to cache
        self.evaluator._get_embedding("test code")
        self.assertGreater(len(self.evaluator._embeddings_cache), 0)
        
        # Clear cache
        self.evaluator.clear_cache()
        self.assertEqual(len(self.evaluator._embeddings_cache), 0)


if __name__ == '__main__':
    unittest.main()

