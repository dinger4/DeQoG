"""
DeQoG Evaluating Agent

State 4 Agent: Quality validation and iterative debugging.
"""

from typing import Any, Dict, List, Optional

from .base_agent import BaseLLMAgent
from ..utils.logger import get_logger

logger = get_logger("evaluating_agent")


class EvaluatingAgent(BaseLLMAgent):
    """
    Decision-making & Evaluating Agent (State 4).
    
    Responsible for:
    1. Executing test cases
    2. Identifying bugs
    3. Integrating feedback into refinement
    4. Iterative code fixing
    
    Uses tools:
    - Test Cases Executor
    - Debugger
    - Code Interpreter
    """
    
    def __init__(
        self,
        llm_client,
        available_tools: Optional[Dict[str, Any]] = None,
        config=None
    ):
        """
        Initialize the evaluating agent.
        
        Args:
            llm_client: LLM client for generation
            available_tools: Available tools
            config: Configuration object
        """
        role_prompt = """You are an expert software engineer specializing in testing and debugging.
Your task is to identify bugs, understand their causes, and fix them.
You are methodical and thorough in your analysis."""
        
        super().__init__(
            llm_client=llm_client,
            role_prompt=role_prompt,
            available_tools=available_tools
        )
        
        self.config = config
        self.quality_threshold = 0.9
        self.max_refinement_iterations = 5
    
    def process(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and refine generated codes.
        
        Args:
            input_data: Generated codes from State 3
            context: Current context with test cases
            
        Returns:
            Dictionary with validated codes
        """
        self._execution_count += 1
        
        logger.info("Starting quality validation (State 4)")
        
        generated_codes = input_data.get('generated_codes', [])
        test_cases = context.get('test_cases', [])
        
        if not generated_codes:
            return {
                'success': False,
                'error': 'No codes to validate'
            }
        
        if not test_cases:
            logger.warning("No test cases provided, skipping validation")
            return {
                'success': True,
                'validated_codes': generated_codes,
                'warning': 'No test cases - validation skipped'
            }
        
        validated_codes = []
        failed_codes = []
        
        for code_info in generated_codes:
            logger.debug(f"Validating code: {code_info.get('impl_id', 'unknown')}")
            
            # Validate and refine
            result = self._validate_and_refine(
                code_info['code'],
                test_cases,
                code_info
            )
            
            if result['success']:
                validated_codes.append({
                    'code': result['code'],
                    'impl_id': code_info.get('impl_id', ''),
                    'algorithm': code_info.get('algorithm', ''),
                    'metrics': {
                        'pass_rate': result['pass_rate'],
                        'iterations': result['iterations']
                    },
                    'meta': {
                        **code_info.get('meta', {}),
                        'refinement_history': result.get('refinement_history', [])
                    }
                })
            else:
                failed_codes.append({
                    'code': code_info['code'],
                    'impl_id': code_info.get('impl_id', ''),
                    'error': result.get('error', 'Validation failed'),
                    'final_pass_rate': result.get('pass_rate', 0)
                })
        
        logger.info(
            f"Validation completed: {len(validated_codes)} validated, "
            f"{len(failed_codes)} failed"
        )
        
        # Calculate overall quality metrics
        quality_metrics = self._calculate_quality_metrics(
            validated_codes, failed_codes, test_cases
        )
        
        return {
            'success': len(validated_codes) > 0,
            'validated_codes': validated_codes,
            'failed_codes': failed_codes,
            'quality_metrics': quality_metrics,
            'stats': {
                'total': len(generated_codes),
                'validated': len(validated_codes),
                'failed': len(failed_codes)
            }
        }
    
    def _validate_and_refine(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
        code_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate code and iteratively refine if needed.
        
        Args:
            code: Code to validate
            test_cases: Test cases
            code_info: Additional code information
            
        Returns:
            Validation/refinement result
        """
        current_code = code
        refinement_history = []
        
        for iteration in range(self.max_refinement_iterations):
            # Execute tests
            test_result = self._execute_tests(current_code, test_cases)
            
            # Record iteration
            refinement_history.append({
                'iteration': iteration,
                'pass_rate': test_result['pass_rate'],
                'failed_count': test_result['failed']
            })
            
            # Check if quality threshold is met
            if test_result['pass_rate'] >= self.quality_threshold:
                logger.debug(
                    f"Code passed with {test_result['pass_rate']:.2%} "
                    f"after {iteration + 1} iterations"
                )
                return {
                    'success': True,
                    'code': current_code,
                    'pass_rate': test_result['pass_rate'],
                    'iterations': iteration + 1,
                    'refinement_history': refinement_history
                }
            
            # If all tests pass, we're done
            if test_result['pass_rate'] == 1.0:
                return {
                    'success': True,
                    'code': current_code,
                    'pass_rate': 1.0,
                    'iterations': iteration + 1,
                    'refinement_history': refinement_history
                }
            
            # Attempt to fix
            logger.debug(
                f"Iteration {iteration + 1}: pass rate {test_result['pass_rate']:.2%}, "
                f"attempting fix..."
            )
            
            fixed_code = self._fix_code(
                current_code,
                test_result['failed_cases'],
                code_info
            )
            
            if fixed_code and fixed_code != current_code:
                current_code = fixed_code
            else:
                # No fix possible, break
                logger.debug("No further fix possible")
                break
        
        # Return best result achieved
        final_test = self._execute_tests(current_code, test_cases)
        
        return {
            'success': final_test['pass_rate'] >= self.quality_threshold,
            'code': current_code,
            'pass_rate': final_test['pass_rate'],
            'iterations': self.max_refinement_iterations,
            'refinement_history': refinement_history,
            'error': 'Failed to meet quality threshold' 
                    if final_test['pass_rate'] < self.quality_threshold else None
        }
    
    def _execute_tests(
        self,
        code: str,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute test cases against code.
        
        Args:
            code: Code to test
            test_cases: Test cases
            
        Returns:
            Test execution results
        """
        test_executor = self.get_tool('test_executor')
        
        if test_executor:
            return test_executor.execute({
                'code': code,
                'test_cases': test_cases
            })
        
        # Fallback: simple execution
        code_interpreter = self.get_tool('code_interpreter')
        
        if not code_interpreter:
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
            # Build execution code
            test_input = tc.get('input')
            expected = tc.get('expected_output')
            
            # Try to execute
            try:
                result = code_interpreter.execute({
                    'code': code,
                    'test_input': test_input
                })
                
                if result['success']:
                    actual = result.get('output', '').strip()
                    if str(actual) == str(expected):
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
                        'expected': expected,
                        'actual': None,
                        'error': result.get('error', 'Execution failed')
                    })
            except Exception as e:
                failed_cases.append({
                    'test_case': tc,
                    'expected': expected,
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
    
    def _fix_code(
        self,
        code: str,
        failed_cases: List[Dict[str, Any]],
        code_info: Dict[str, Any]
    ) -> Optional[str]:
        """
        Attempt to fix code based on failed test cases.
        
        Args:
            code: Code to fix
            failed_cases: Failed test cases
            code_info: Additional code information
            
        Returns:
            Fixed code or None
        """
        debugger = self.get_tool('debugger')
        
        # Get debug analysis
        if debugger:
            debug_result = debugger.execute({
                'code': code,
                'test_failures': failed_cases,
                'error_messages': [f.get('error', '') for f in failed_cases if f.get('error')]
            })
            
            # Check if we got a fixed code from debugger
            if debug_result.get('suggested_fixes'):
                for fix in debug_result['suggested_fixes']:
                    if fix.get('fixed_code'):
                        return fix['fixed_code']
        
        # Use LLM to fix
        failed_summary = self._format_failed_cases(failed_cases[:3])
        
        prompt = f"""{self.role_prompt}

## Original Code
```python
{code}
```

## Failed Test Cases
{failed_summary}

## Task
Fix the code to pass these test cases.
Maintain the original algorithm approach.
Focus on the specific bugs that cause the failures.

Provide ONLY the fixed Python code:

```python"""
        
        try:
            response = self.generate(prompt, temperature=0.2)
            
            # Extract code
            import re
            code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            
            if 'def ' in response:
                return response.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Fix generation failed: {e}")
            return None
    
    def _format_failed_cases(
        self,
        failed_cases: List[Dict[str, Any]]
    ) -> str:
        """Format failed test cases for prompt."""
        lines = []
        
        for i, case in enumerate(failed_cases, 1):
            tc = case.get('test_case', {})
            lines.append(f"Case {i}:")
            lines.append(f"  Input: {tc.get('input', 'N/A')}")
            lines.append(f"  Expected: {case.get('expected', 'N/A')}")
            lines.append(f"  Actual: {case.get('actual', 'N/A')}")
            if case.get('error'):
                lines.append(f"  Error: {case['error']}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _calculate_quality_metrics(
        self,
        validated_codes: List[Dict[str, Any]],
        failed_codes: List[Dict[str, Any]],
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate overall quality metrics."""
        total_codes = len(validated_codes) + len(failed_codes)
        
        if total_codes == 0:
            return {
                'validation_rate': 0.0,
                'average_pass_rate': 0.0,
                'average_iterations': 0.0
            }
        
        # Validation rate
        validation_rate = len(validated_codes) / total_codes
        
        # Average pass rate
        pass_rates = [
            c.get('metrics', {}).get('pass_rate', 0)
            for c in validated_codes
        ]
        avg_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0
        
        # Average iterations
        iterations = [
            c.get('metrics', {}).get('iterations', 0)
            for c in validated_codes
        ]
        avg_iterations = sum(iterations) / len(iterations) if iterations else 0
        
        return {
            'validation_rate': validation_rate,
            'average_pass_rate': avg_pass_rate,
            'average_iterations': avg_iterations,
            'total_test_cases': len(test_cases),
            'validated_count': len(validated_codes),
            'failed_count': len(failed_codes)
        }
    
    def iterative_refinement(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
        max_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Public interface for iterative refinement.
        
        Args:
            code: Code to refine
            test_cases: Test cases
            max_iterations: Maximum iterations (optional)
            
        Returns:
            Refinement result
        """
        if max_iterations:
            original = self.max_refinement_iterations
            self.max_refinement_iterations = max_iterations
        
        result = self._validate_and_refine(code, test_cases, {})
        
        if max_iterations:
            self.max_refinement_iterations = original
        
        return result

