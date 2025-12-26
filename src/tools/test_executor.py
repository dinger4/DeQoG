"""
DeQoG Test Executor

Executes test cases against generated code and validates functional correctness.
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .base_tool import BaseTool
from .code_interpreter import CodeInterpreter
from ..utils.logger import get_logger

logger = get_logger("test_executor")


@dataclass
class TestResult:
    """Result of a single test case execution."""
    test_case: Dict[str, Any]
    passed: bool
    actual_output: Any
    expected_output: Any
    error: Optional[str] = None
    execution_time: float = 0.0


class TestExecutor(BaseTool):
    """
    Test Cases Executor Tool.
    
    Runs comprehensive test suites against generated code
    to verify functional correctness.
    """
    
    def __init__(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        timeout_per_test: int = 5
    ):
        """
        Initialize the test executor.
        
        Args:
            parallel: Whether to run tests in parallel
            max_workers: Maximum number of parallel workers
            timeout_per_test: Timeout for each test case
        """
        super().__init__(
            name="test_executor",
            description="Executes test cases to verify code correctness"
        )
        
        self.parallel = parallel
        self.max_workers = max_workers
        self.timeout_per_test = timeout_per_test
        self.code_interpreter = CodeInterpreter(timeout=timeout_per_test)
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute test cases against the provided code.
        
        Args:
            params: Dictionary containing:
                - code: Code string to test
                - test_cases: List of test cases
                - function_name: Name of the function to test (optional)
                
        Returns:
            Dictionary with test results
        """
        code = params.get('code', '')
        test_cases = params.get('test_cases', [])
        function_name = params.get('function_name', None)
        
        if not test_cases:
            return {
                'pass_rate': 1.0,
                'passed': 0,
                'failed': 0,
                'total': 0,
                'results': [],
                'message': 'No test cases provided'
            }
        
        # Extract function name from code if not provided
        if function_name is None:
            function_name = self._extract_function_name(code)
        
        # Run tests
        if self.parallel and len(test_cases) > 1:
            results = self._run_tests_parallel(code, test_cases, function_name)
        else:
            results = self._run_tests_sequential(code, test_cases, function_name)
        
        # Aggregate results
        return self._aggregate_results(results)
    
    def _extract_function_name(self, code: str) -> Optional[str]:
        """Extract the main function name from code."""
        import ast
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except:
            pass
        
        return None
    
    def _run_tests_sequential(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
        function_name: Optional[str]
    ) -> List[TestResult]:
        """Run tests sequentially."""
        results = []
        
        for test_case in test_cases:
            result = self._run_single_test(code, test_case, function_name)
            results.append(result)
        
        return results
    
    def _run_tests_parallel(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
        function_name: Optional[str]
    ) -> List[TestResult]:
        """Run tests in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_single_test, code, test_case, function_name
                ): test_case
                for test_case in test_cases
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    test_case = futures[future]
                    results.append(TestResult(
                        test_case=test_case,
                        passed=False,
                        actual_output=None,
                        expected_output=test_case.get('expected_output'),
                        error=str(e)
                    ))
        
        return results
    
    def _run_single_test(
        self,
        code: str,
        test_case: Dict[str, Any],
        function_name: Optional[str]
    ) -> TestResult:
        """
        Run a single test case.
        
        Args:
            code: Code to test
            test_case: Test case dictionary with 'input' and 'expected_output'
            function_name: Name of the function to call
            
        Returns:
            TestResult object
        """
        import time
        
        start_time = time.time()
        
        test_input = test_case.get('input')
        expected_output = test_case.get('expected_output')
        
        try:
            # Build test execution code
            exec_code = self._build_test_code(code, function_name, test_input)
            
            # Execute
            result = self.code_interpreter.execute({
                'code': exec_code,
                'timeout': self.timeout_per_test
            })
            
            execution_time = time.time() - start_time
            
            if not result['success']:
                return TestResult(
                    test_case=test_case,
                    passed=False,
                    actual_output=None,
                    expected_output=expected_output,
                    error=result.get('error', 'Unknown error'),
                    execution_time=execution_time
                )
            
            # Parse actual output
            actual_output = self._parse_output(result.get('output', ''))
            
            # Compare outputs
            passed = self._compare_outputs(actual_output, expected_output)
            
            return TestResult(
                test_case=test_case,
                passed=passed,
                actual_output=actual_output,
                expected_output=expected_output,
                execution_time=execution_time
            )
            
        except Exception as e:
            return TestResult(
                test_case=test_case,
                passed=False,
                actual_output=None,
                expected_output=expected_output,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _build_test_code(
        self,
        code: str,
        function_name: Optional[str],
        test_input: Any
    ) -> str:
        """Build code for test execution."""
        # Format test input
        if isinstance(test_input, dict):
            args_str = ", ".join(f"{k}={v!r}" for k, v in test_input.items())
        elif isinstance(test_input, (list, tuple)):
            args_str = ", ".join(repr(arg) for arg in test_input)
        else:
            args_str = repr(test_input)
        
        # Build execution code
        if function_name:
            exec_code = f"""
{code}

result = {function_name}({args_str})
print(repr(result))
"""
        else:
            # If no function name, assume the code directly produces output
            exec_code = code
        
        return exec_code
    
    def _parse_output(self, output: str) -> Any:
        """Parse the output string to a Python value."""
        output = output.strip()
        
        if not output:
            return None
        
        try:
            # Try to evaluate as Python literal
            return eval(output)
        except:
            # Return as string if evaluation fails
            return output
    
    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """
        Compare actual and expected outputs.
        
        Handles various edge cases and type conversions.
        """
        # Direct equality
        if actual == expected:
            return True
        
        # String representation comparison
        if str(actual) == str(expected):
            return True
        
        # Float comparison with tolerance
        try:
            if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                return abs(float(actual) - float(expected)) < 1e-9
        except:
            pass
        
        # List/tuple comparison (order matters)
        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                return False
            return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))
        
        # Set comparison (order doesn't matter)
        if isinstance(actual, set) and isinstance(expected, set):
            return actual == expected
        
        # Convert to set for comparison if both are lists/tuples
        # (for problems where order doesn't matter)
        try:
            if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
                if set(map(str, actual)) == set(map(str, expected)):
                    return True
        except:
            pass
        
        return False
    
    def _aggregate_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """Aggregate test results into a summary."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        
        pass_rate = passed / total if total > 0 else 0.0
        
        failed_cases = [
            {
                'test_case': r.test_case,
                'actual': r.actual_output,
                'expected': r.expected_output,
                'error': r.error
            }
            for r in results if not r.passed
        ]
        
        return {
            'pass_rate': pass_rate,
            'passed': passed,
            'failed': failed,
            'total': total,
            'failed_cases': failed_cases,
            'results': [
                {
                    'test_case': r.test_case,
                    'passed': r.passed,
                    'actual': r.actual_output,
                    'expected': r.expected_output,
                    'error': r.error,
                    'execution_time': r.execution_time
                }
                for r in results
            ],
            'total_execution_time': sum(r.execution_time for r in results)
        }
    
    def run_single_test(
        self,
        code: str,
        test_case: Dict[str, Any],
        function_name: Optional[str] = None
    ) -> TestResult:
        """
        Public interface to run a single test.
        
        Args:
            code: Code to test
            test_case: Test case with 'input' and 'expected_output'
            function_name: Optional function name
            
        Returns:
            TestResult object
        """
        if function_name is None:
            function_name = self._extract_function_name(code)
        
        return self._run_single_test(code, test_case, function_name)

