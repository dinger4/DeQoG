"""
DeQoG Quality Assurance Engine

Implements feedback-based iterative optimization for code quality.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from ..utils.logger import get_logger

logger = get_logger("quality_assurance")


@dataclass
class RefinementResult:
    """Result of a refinement iteration."""
    iteration: int
    code: str
    pass_rate: float
    failed_tests: List[Dict[str, Any]] = field(default_factory=list)
    fix_applied: bool = False


class QualityAssuranceEngine:
    """
    Quality Assurance Engine.
    
    Implements iterative code refinement based on test feedback.
    Coordinates test execution, bug analysis, and code fixing.
    """
    
    def __init__(
        self,
        test_executor=None,
        debugger=None,
        code_interpreter=None,
        llm_client=None,
        max_iterations: int = 5,
        quality_threshold: float = 0.9
    ):
        """
        Initialize the quality assurance engine.
        
        Args:
            test_executor: Test execution tool
            debugger: Debugging tool
            code_interpreter: Code interpreter tool
            llm_client: LLM client for fix generation
            max_iterations: Maximum refinement iterations
            quality_threshold: Target pass rate
        """
        self.test_executor = test_executor
        self.debugger = debugger
        self.code_interpreter = code_interpreter
        self.llm_client = llm_client
        
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
    
    def validate_and_refine(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate code and iteratively refine until quality threshold is met.
        
        Args:
            code: Code to validate and refine
            test_cases: Test cases for validation
            context: Optional context information
            
        Returns:
            Dictionary with refined code and metrics
        """
        logger.info(f"Starting quality assurance (max {self.max_iterations} iterations)")
        
        current_code = code
        refinement_history: List[RefinementResult] = []
        
        for iteration in range(self.max_iterations):
            logger.debug(f"QA iteration {iteration + 1}")
            
            # Execute tests
            test_result = self._execute_tests(current_code, test_cases)
            
            # Record iteration
            refinement_history.append(RefinementResult(
                iteration=iteration,
                code=current_code,
                pass_rate=test_result['pass_rate'],
                failed_tests=test_result.get('failed_cases', [])
            ))
            
            # Check if quality threshold is met
            if test_result['pass_rate'] >= self.quality_threshold:
                logger.info(
                    f"Quality threshold met at iteration {iteration + 1} "
                    f"(pass rate: {test_result['pass_rate']:.2%})"
                )
                return self._create_result(
                    current_code, test_result, refinement_history, success=True
                )
            
            # If all tests pass, we're done
            if test_result['pass_rate'] == 1.0:
                return self._create_result(
                    current_code, test_result, refinement_history, success=True
                )
            
            # Attempt to fix
            logger.debug(f"Pass rate {test_result['pass_rate']:.2%}, attempting fix...")
            
            fixed_code = self._generate_fix(
                current_code,
                test_result.get('failed_cases', []),
                context
            )
            
            if fixed_code and fixed_code != current_code:
                current_code = fixed_code
                refinement_history[-1].fix_applied = True
            else:
                logger.debug("No fix generated, stopping refinement")
                break
        
        # Return best result
        final_test = self._execute_tests(current_code, test_cases)
        return self._create_result(
            current_code, final_test, refinement_history,
            success=final_test['pass_rate'] >= self.quality_threshold
        )
    
    def _execute_tests(
        self,
        code: str,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute tests against code."""
        if self.test_executor:
            return self.test_executor.execute({
                'code': code,
                'test_cases': test_cases
            })
        
        # Fallback: basic execution
        if not self.code_interpreter:
            logger.warning("No test executor or code interpreter available")
            return {
                'pass_rate': 0.0,
                'passed': 0,
                'failed': len(test_cases),
                'total': len(test_cases),
                'failed_cases': test_cases
            }
        
        passed = 0
        failed_cases = []
        
        for tc in test_cases:
            try:
                # This is a simplified fallback
                result = self.code_interpreter.execute({
                    'code': code,
                    'test_input': tc.get('input')
                })
                
                if result['success']:
                    actual = result.get('output', '').strip()
                    expected = str(tc.get('expected_output', ''))
                    
                    if actual == expected:
                        passed += 1
                    else:
                        failed_cases.append({
                            'test_case': tc,
                            'expected': expected,
                            'actual': actual
                        })
                else:
                    failed_cases.append({
                        'test_case': tc,
                        'expected': tc.get('expected_output'),
                        'actual': None,
                        'error': result.get('error')
                    })
            except Exception as e:
                failed_cases.append({
                    'test_case': tc,
                    'expected': tc.get('expected_output'),
                    'actual': None,
                    'error': str(e)
                })
        
        total = len(test_cases)
        return {
            'pass_rate': passed / total if total > 0 else 0.0,
            'passed': passed,
            'failed': total - passed,
            'total': total,
            'failed_cases': failed_cases
        }
    
    def _generate_fix(
        self,
        code: str,
        failed_cases: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Generate a fix for the code based on failed tests."""
        # Try using debugger first
        if self.debugger:
            debug_result = self.debugger.execute({
                'code': code,
                'test_failures': failed_cases,
                'error_messages': [f.get('error', '') for f in failed_cases if f.get('error')]
            })
            
            # Check for generated fix
            for fix in debug_result.get('suggested_fixes', []):
                if fix.get('fixed_code'):
                    return fix['fixed_code']
        
        # Fall back to LLM-based fix
        if not self.llm_client:
            return None
        
        # Format failed cases for prompt
        failure_summary = self._format_failures(failed_cases[:3])
        
        prompt = f"""Fix the following Python code based on test failures:

## Code
```python
{code}
```

## Failed Tests
{failure_summary}

## Instructions
1. Identify the bug causing the failures
2. Fix the code while maintaining the algorithm
3. Provide ONLY the corrected code

```python"""
        
        try:
            response = self.llm_client.generate(prompt, temperature=0.2)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract code
            import re
            match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                return match.group(1).strip()
            
            if 'def ' in content:
                return content.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Fix generation failed: {e}")
            return None
    
    def _format_failures(self, failures: List[Dict[str, Any]]) -> str:
        """Format failures for prompt."""
        lines = []
        for i, f in enumerate(failures, 1):
            tc = f.get('test_case', {})
            lines.append(f"Test {i}:")
            lines.append(f"  Input: {tc.get('input')}")
            lines.append(f"  Expected: {f.get('expected')}")
            lines.append(f"  Actual: {f.get('actual')}")
            if f.get('error'):
                lines.append(f"  Error: {f['error']}")
        return "\n".join(lines)
    
    def _create_result(
        self,
        code: str,
        test_result: Dict[str, Any],
        history: List[RefinementResult],
        success: bool
    ) -> Dict[str, Any]:
        """Create the final result dictionary."""
        return {
            'success': success,
            'code': code,
            'pass_rate': test_result['pass_rate'],
            'iterations': len(history),
            'refinement_history': [
                {
                    'iteration': r.iteration,
                    'pass_rate': r.pass_rate,
                    'failed_count': len(r.failed_tests),
                    'fix_applied': r.fix_applied
                }
                for r in history
            ],
            'final_test_result': {
                'passed': test_result['passed'],
                'failed': test_result['failed'],
                'total': test_result['total']
            }
        }
    
    def validate_single(
        self,
        code: str,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate code without refinement.
        
        Args:
            code: Code to validate
            test_cases: Test cases
            
        Returns:
            Validation result
        """
        test_result = self._execute_tests(code, test_cases)
        return {
            'valid': test_result['pass_rate'] >= self.quality_threshold,
            'pass_rate': test_result['pass_rate'],
            'passed': test_result['passed'],
            'failed': test_result['failed'],
            'total': test_result['total'],
            'failed_cases': test_result.get('failed_cases', [])
        }

