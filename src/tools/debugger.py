"""
DeQoG Debugger

Identifies and suggests fixes for bugs based on test failures and error messages.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .base_tool import BaseTool
from ..utils.logger import get_logger

logger = get_logger("debugger")


class Debugger(BaseTool):
    """
    Debugger Tool.
    
    Analyzes code failures, identifies bugs, and generates fix suggestions
    using LLM-based analysis.
    """
    
    def __init__(
        self,
        llm_client=None,
        max_analysis_depth: int = 3,
        include_stack_trace: bool = True
    ):
        """
        Initialize the debugger.
        
        Args:
            llm_client: LLM client for analysis
            max_analysis_depth: Maximum depth for bug analysis
            include_stack_trace: Whether to include stack traces in analysis
        """
        super().__init__(
            name="debugger",
            description="Analyzes bugs and suggests fixes"
        )
        
        self.llm_client = llm_client
        self.max_analysis_depth = max_analysis_depth
        self.include_stack_trace = include_stack_trace
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze bugs and generate fix suggestions.
        
        Args:
            params: Dictionary containing:
                - code: Buggy code
                - test_failures: List of failed test cases
                - error_messages: Error messages from execution
                - llm_client: Optional LLM client override
                
        Returns:
            Dictionary with bug analysis and fix suggestions
        """
        code = params.get('code', '')
        test_failures = params.get('test_failures', [])
        error_messages = params.get('error_messages', [])
        llm_client = params.get('llm_client', self.llm_client)
        
        # Analyze each failure
        analyses = []
        for failure in test_failures[:self.max_analysis_depth]:
            analysis = self.analyze_failure(code, failure)
            analyses.append(analysis)
        
        # Generate overall bug analysis
        bug_analysis = self._synthesize_analyses(analyses, error_messages)
        
        # Generate fix suggestions
        if llm_client:
            fix_suggestions = self.generate_fix_suggestions(
                code, bug_analysis, test_failures, llm_client
            )
        else:
            fix_suggestions = self._generate_heuristic_fixes(code, bug_analysis)
        
        return {
            'bug_analysis': bug_analysis,
            'suggested_fixes': fix_suggestions,
            'individual_analyses': analyses
        }
    
    def analyze_failure(
        self,
        code: str,
        test_failure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a single test failure.
        
        Args:
            code: The buggy code
            test_failure: Test failure information
            
        Returns:
            Analysis dictionary
        """
        analysis = {
            'test_case': test_failure.get('test_case', {}),
            'expected': test_failure.get('expected'),
            'actual': test_failure.get('actual'),
            'error': test_failure.get('error'),
            'probable_causes': [],
            'affected_lines': []
        }
        
        error = test_failure.get('error', '')
        
        # Identify error type
        if error:
            analysis['error_type'] = self._identify_error_type(error)
            analysis['probable_causes'] = self._get_probable_causes(
                analysis['error_type'], error
            )
            
            # Try to identify affected lines
            line_numbers = self._extract_line_numbers(error)
            if line_numbers:
                analysis['affected_lines'] = self._get_code_lines(code, line_numbers)
        else:
            # Logic error - output doesn't match expected
            analysis['error_type'] = 'LogicError'
            analysis['probable_causes'] = self._analyze_output_mismatch(
                test_failure.get('expected'),
                test_failure.get('actual')
            )
        
        return analysis
    
    def _identify_error_type(self, error: str) -> str:
        """Identify the type of error from error message."""
        error_patterns = {
            'IndexError': r'IndexError',
            'KeyError': r'KeyError',
            'TypeError': r'TypeError',
            'ValueError': r'ValueError',
            'ZeroDivisionError': r'ZeroDivisionError',
            'AttributeError': r'AttributeError',
            'NameError': r'NameError',
            'RecursionError': r'RecursionError|maximum recursion depth',
            'TimeoutError': r'TimeoutError|timeout',
            'SyntaxError': r'SyntaxError',
        }
        
        for error_type, pattern in error_patterns.items():
            if re.search(pattern, error, re.IGNORECASE):
                return error_type
        
        return 'UnknownError'
    
    def _get_probable_causes(
        self,
        error_type: str,
        error_message: str
    ) -> List[str]:
        """Get probable causes based on error type."""
        causes = {
            'IndexError': [
                'Array index out of bounds',
                'Empty list access',
                'Off-by-one error in loop bounds'
            ],
            'KeyError': [
                'Dictionary key not found',
                'Missing key initialization',
                'Typo in key name'
            ],
            'TypeError': [
                'Wrong argument types passed',
                'Operation on incompatible types',
                'None value where object expected'
            ],
            'ValueError': [
                'Invalid value for operation',
                'Conversion error',
                'Empty or invalid input'
            ],
            'ZeroDivisionError': [
                'Division by zero',
                'Missing zero check',
                'Modulo by zero'
            ],
            'RecursionError': [
                'Missing or incorrect base case',
                'Infinite recursion',
                'Recursive call with unchanged parameters'
            ],
            'LogicError': [
                'Incorrect algorithm implementation',
                'Wrong condition in if/while statement',
                'Off-by-one error',
                'Wrong return value'
            ]
        }
        
        return causes.get(error_type, ['Unknown error cause'])
    
    def _extract_line_numbers(self, error: str) -> List[int]:
        """Extract line numbers from error message."""
        line_numbers = []
        
        # Pattern: "line X" or "Line X"
        matches = re.findall(r'line\s+(\d+)', error, re.IGNORECASE)
        line_numbers.extend(int(m) for m in matches)
        
        return sorted(set(line_numbers))
    
    def _get_code_lines(
        self,
        code: str,
        line_numbers: List[int]
    ) -> List[Dict[str, Any]]:
        """Get specific code lines."""
        lines = code.split('\n')
        result = []
        
        for ln in line_numbers:
            if 1 <= ln <= len(lines):
                result.append({
                    'line_number': ln,
                    'content': lines[ln - 1]
                })
        
        return result
    
    def _analyze_output_mismatch(
        self,
        expected: Any,
        actual: Any
    ) -> List[str]:
        """Analyze why output doesn't match expected."""
        causes = []
        
        if actual is None and expected is not None:
            causes.append('Function may not return a value (returns None)')
        
        if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            if len(expected) != len(actual):
                causes.append(f'Output length mismatch: expected {len(expected)}, got {len(actual)}')
            else:
                causes.append('Elements in the output differ from expected')
        
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if expected > 0 and actual <= 0:
                causes.append('Sign error - expected positive, got non-positive')
            elif expected < 0 and actual >= 0:
                causes.append('Sign error - expected negative, got non-negative')
            else:
                causes.append('Numerical calculation error')
        
        if isinstance(expected, str) and isinstance(actual, str):
            if expected.lower() == actual.lower():
                causes.append('Case sensitivity issue')
            if expected.strip() == actual.strip():
                causes.append('Whitespace handling issue')
        
        if not causes:
            causes.append('Logic error in algorithm implementation')
        
        return causes
    
    def _synthesize_analyses(
        self,
        analyses: List[Dict[str, Any]],
        error_messages: List[str]
    ) -> Dict[str, Any]:
        """Synthesize individual analyses into overall analysis."""
        error_types = [a.get('error_type', 'Unknown') for a in analyses]
        all_causes = []
        for a in analyses:
            all_causes.extend(a.get('probable_causes', []))
        
        # Find most common error type
        from collections import Counter
        error_counter = Counter(error_types)
        primary_error = error_counter.most_common(1)[0][0] if error_counter else 'Unknown'
        
        # Deduplicate causes
        unique_causes = list(dict.fromkeys(all_causes))
        
        return {
            'primary_error_type': primary_error,
            'error_distribution': dict(error_counter),
            'probable_causes': unique_causes[:5],  # Top 5 causes
            'total_failures_analyzed': len(analyses),
            'additional_errors': error_messages
        }
    
    def generate_fix_suggestions(
        self,
        code: str,
        bug_analysis: Dict[str, Any],
        test_failures: List[Dict[str, Any]],
        llm_client
    ) -> List[Dict[str, Any]]:
        """
        Generate fix suggestions using LLM.
        
        Args:
            code: Original buggy code
            bug_analysis: Bug analysis results
            test_failures: Test failure details
            llm_client: LLM client for generating fixes
            
        Returns:
            List of fix suggestions
        """
        # Prepare failure summary
        failure_summary = []
        for f in test_failures[:3]:
            failure_summary.append(
                f"Input: {f.get('test_case', {}).get('input')}, "
                f"Expected: {f.get('expected')}, "
                f"Actual: {f.get('actual')}, "
                f"Error: {f.get('error', 'None')}"
            )
        
        prompt = f"""Analyze and fix the following buggy Python code.

## Code
```python
{code}
```

## Bug Analysis
- Primary Error Type: {bug_analysis.get('primary_error_type')}
- Probable Causes: {', '.join(bug_analysis.get('probable_causes', []))}

## Test Failures
{chr(10).join(failure_summary)}

## Task
1. Identify the exact bug(s) in the code
2. Explain why the bug causes the observed failures
3. Provide the corrected code

Please provide your response in the following format:
BUG EXPLANATION:
[Your explanation]

FIXED CODE:
```python
[Your corrected code]
```
"""
        
        try:
            response = llm_client.generate(prompt, temperature=0.3)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the response
            explanation = self._extract_section(content, 'BUG EXPLANATION:')
            fixed_code = self._extract_code_block(content)
            
            return [{
                'type': 'llm_generated',
                'explanation': explanation,
                'fixed_code': fixed_code,
                'confidence': 'high' if fixed_code else 'low'
            }]
            
        except Exception as e:
            logger.error(f"LLM fix generation failed: {e}")
            return self._generate_heuristic_fixes(code, bug_analysis)
    
    def _extract_section(self, text: str, header: str) -> str:
        """Extract a section from the response."""
        start = text.find(header)
        if start == -1:
            return ''
        
        start += len(header)
        end = text.find('\n\n', start)
        
        if end == -1:
            end = text.find('```', start)
        
        if end == -1:
            return text[start:].strip()
        
        return text[start:end].strip()
    
    def _extract_code_block(self, text: str) -> str:
        """Extract code block from response."""
        # Look for ```python ... ``` pattern
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[-1].strip()  # Return the last code block
        
        # Try without language specifier
        pattern = r'```\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[-1].strip()
        
        return ''
    
    def _generate_heuristic_fixes(
        self,
        code: str,
        bug_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate heuristic-based fix suggestions."""
        fixes = []
        error_type = bug_analysis.get('primary_error_type', '')
        
        heuristic_fixes = {
            'IndexError': {
                'explanation': 'Add bounds checking before array access',
                'pattern': r'(\w+)\[(\w+)\]',
                'suggestion': 'Add: if 0 <= index < len(array)'
            },
            'ZeroDivisionError': {
                'explanation': 'Add zero check before division',
                'pattern': r'/\s*(\w+)',
                'suggestion': 'Add: if divisor != 0'
            },
            'RecursionError': {
                'explanation': 'Check base case and recursive call',
                'pattern': r'def\s+(\w+)\s*\(',
                'suggestion': 'Ensure base case is correct and parameters change in recursive call'
            }
        }
        
        if error_type in heuristic_fixes:
            fix = heuristic_fixes[error_type]
            fixes.append({
                'type': 'heuristic',
                'explanation': fix['explanation'],
                'suggestion': fix['suggestion'],
                'confidence': 'medium'
            })
        
        # Generic suggestions
        fixes.append({
            'type': 'generic',
            'explanation': 'Review the algorithm logic',
            'suggestion': 'Check conditions, loop bounds, and return statements',
            'confidence': 'low'
        })
        
        return fixes

