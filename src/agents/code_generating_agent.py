"""
DeQoG Code Generating Agent

State 3 Agent: Code synthesis from diverse implementation plans.
"""

import re
from typing import Any, Dict, List, Optional

from .base_agent import BaseLLMAgent
from ..utils.logger import get_logger

logger = get_logger("code_generating_agent")


class CodeGeneratingAgent(BaseLLMAgent):
    """
    Code Generating Agent (State 3).
    
    Responsible for:
    1. Converting implementation plans to executable code
    2. Validating code syntax
    3. Ensuring implementation diversity
    4. Maintaining code quality
    
    Uses tools:
    - Code Interpreter
    - Diversity Evaluator
    """
    
    def __init__(
        self,
        llm_client,
        available_tools: Optional[Dict[str, Any]] = None,
        config=None
    ):
        """
        Initialize the code generating agent.
        
        Args:
            llm_client: LLM client for generation
            available_tools: Available tools
            config: Configuration object
        """
        role_prompt = """You are an expert Python programmer.
Your task is to write clean, efficient, and correct Python code.
You follow best practices:
- Clear variable naming
- Appropriate comments
- Proper error handling
- Efficient algorithms"""
        
        super().__init__(
            llm_client=llm_client,
            role_prompt=role_prompt,
            available_tools=available_tools
        )
        
        self.config = config
        self.diversity_threshold = 0.7
    
    def process(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate code from implementation plans.
        
        Args:
            input_data: Diverse implementation plans from State 2
            context: Current context
            
        Returns:
            Dictionary with generated codes
        """
        self._execution_count += 1
        
        logger.info("Starting code synthesis (State 3)")
        
        implementations = input_data.get('implementation_level', [])
        task_description = context.get('task_description', '')
        function_signature = context.get('function_signature', '')
        
        if not implementations:
            return {
                'success': False,
                'error': 'No implementation plans provided'
            }
        
        generated_codes = []
        failed_generations = []
        
        for impl in implementations:
            logger.debug(f"Generating code for {impl['id']}")
            
            # Generate code
            code_result = self._generate_code(
                impl, task_description, function_signature, context
            )
            
            if not code_result['success']:
                failed_generations.append({
                    'impl_id': impl['id'],
                    'error': code_result.get('error', 'Unknown error')
                })
                continue
            
            code = code_result['code']
            
            # Validate syntax
            syntax_result = self._validate_syntax(code)
            
            if not syntax_result['valid']:
                # Try to fix syntax errors
                fixed_code = self._fix_syntax_error(
                    code, syntax_result['error'], impl, context
                )
                
                if fixed_code:
                    code = fixed_code
                else:
                    failed_generations.append({
                        'impl_id': impl['id'],
                        'error': f"Syntax error: {syntax_result['error']}"
                    })
                    continue
            
            # Check diversity against existing codes
            is_diverse, similarity = self._check_diversity(
                code, [c['code'] for c in generated_codes]
            )
            
            if not is_diverse:
                logger.warning(
                    f"Code for {impl['id']} too similar (similarity: {similarity:.2f})"
                )
                # Still include but mark as potentially similar
            
            generated_codes.append({
                'code': code,
                'impl_id': impl['id'],
                'algorithm': impl.get('meta', {}).get('variant', 'unknown'),
                'meta': {
                    'parent_solution': impl.get('parent_solution', ''),
                    'variation_hints': impl.get('meta', {}).get('variation_hints', []),
                    'diversity_check': {
                        'is_diverse': is_diverse,
                        'max_similarity': similarity
                    }
                }
            })
        
        logger.info(
            f"Code synthesis completed: {len(generated_codes)} codes generated, "
            f"{len(failed_generations)} failed"
        )
        
        return {
            'success': len(generated_codes) > 0,
            'generated_codes': generated_codes,
            'failed_generations': failed_generations,
            'stats': {
                'total_attempts': len(implementations),
                'successful': len(generated_codes),
                'failed': len(failed_generations)
            }
        }
    
    def _generate_code(
        self,
        implementation: Dict[str, Any],
        task_description: str,
        function_signature: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate code from an implementation plan.
        
        Args:
            implementation: Implementation plan
            task_description: Original task description
            function_signature: Expected function signature
            context: Current context
            
        Returns:
            Dictionary with code or error
        """
        # Build rollback warnings if any
        rollback_warnings = ""
        if context.get('rollback_warnings'):
            warnings = context['rollback_warnings']
            if isinstance(warnings, list):
                rollback_warnings = "\n⚠️ Avoid these previous mistakes:\n"
                rollback_warnings += "\n".join(f"- {w}" for w in warnings)
        
        prompt = f"""{self.role_prompt}

## Task Description
{task_description}

## Implementation Plan
{implementation['content']}

## Function Signature
{function_signature if function_signature else "Define an appropriate function"}

## Requirements
1. Write complete, executable Python code
2. Follow the implementation plan precisely
3. Include appropriate comments
4. Handle edge cases properly
5. Use meaningful variable names
{rollback_warnings}

## Output
Provide ONLY the Python code, no explanations.
Start with the function definition.

```python"""
        
        try:
            response = self.generate(prompt, temperature=0.3, max_tokens=1500)
            
            # Extract code from response
            code = self._extract_code(response)
            
            if not code:
                return {
                    'success': False,
                    'error': 'Failed to extract code from response'
                }
            
            return {
                'success': True,
                'code': code
            }
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_code(self, response: str) -> str:
        """
        Extract Python code from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted code string
        """
        # Try to find code block
        code_patterns = [
            r'```python\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
        ]
        
        for pattern in code_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no code block, try to find function definition
        if 'def ' in response:
            lines = response.split('\n')
            code_lines = []
            in_function = False
            
            for line in lines:
                if line.strip().startswith('def '):
                    in_function = True
                
                if in_function:
                    code_lines.append(line)
            
            if code_lines:
                return '\n'.join(code_lines)
        
        # Last resort: return the whole response
        return response.strip()
    
    def _validate_syntax(self, code: str) -> Dict[str, Any]:
        """
        Validate code syntax.
        
        Args:
            code: Code to validate
            
        Returns:
            Validation result dictionary
        """
        code_interpreter = self.get_tool('code_interpreter')
        
        if code_interpreter:
            return code_interpreter.validate_syntax(code)
        
        # Fallback: use ast
        import ast
        try:
            ast.parse(code)
            return {'valid': True, 'error': None}
        except SyntaxError as e:
            return {
                'valid': False,
                'error': f"Line {e.lineno}: {e.msg}",
                'line': e.lineno
            }
    
    def _fix_syntax_error(
        self,
        code: str,
        error: str,
        implementation: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Attempt to fix syntax errors in code.
        
        Args:
            code: Code with syntax error
            error: Error message
            implementation: Original implementation plan
            context: Current context
            
        Returns:
            Fixed code or None if fix failed
        """
        prompt = f"""The following Python code has a syntax error:

```python
{code}
```

Error: {error}

Please fix the syntax error and provide the corrected code.
Maintain the original algorithm and logic.

```python"""
        
        try:
            response = self.generate(prompt, temperature=0.1)
            fixed_code = self._extract_code(response)
            
            # Validate the fix
            if fixed_code:
                validation = self._validate_syntax(fixed_code)
                if validation['valid']:
                    logger.info("Successfully fixed syntax error")
                    return fixed_code
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to fix syntax error: {e}")
            return None
    
    def _check_diversity(
        self,
        new_code: str,
        existing_codes: List[str]
    ) -> tuple:
        """
        Check if new code is diverse enough from existing codes.
        
        Args:
            new_code: New code to check
            existing_codes: List of existing codes
            
        Returns:
            Tuple of (is_diverse, max_similarity)
        """
        if not existing_codes:
            return True, 0.0
        
        diversity_evaluator = self.get_tool('diversity_evaluator')
        
        if diversity_evaluator:
            return diversity_evaluator.is_diverse_enough(
                new_code,
                existing_codes,
                self.diversity_threshold
            )
        
        # Fallback: simple text similarity
        from difflib import SequenceMatcher
        
        similarities = [
            SequenceMatcher(None, new_code, existing).ratio()
            for existing in existing_codes
        ]
        
        max_sim = max(similarities) if similarities else 0.0
        return max_sim < self.diversity_threshold, max_sim
    
    def generate_single_code(
        self,
        implementation_plan: str,
        task_description: str,
        function_signature: str = ""
    ) -> Dict[str, Any]:
        """
        Generate a single code from an implementation plan.
        
        Public interface for generating one code.
        
        Args:
            implementation_plan: Natural language implementation plan
            task_description: Task description
            function_signature: Expected function signature
            
        Returns:
            Dictionary with generated code or error
        """
        impl = {
            'id': 'single',
            'content': implementation_plan,
            'meta': {}
        }
        
        result = self._generate_code(
            impl, task_description, function_signature, {}
        )
        
        if result['success']:
            # Validate
            syntax_result = self._validate_syntax(result['code'])
            if syntax_result['valid']:
                return result
            else:
                # Try to fix
                fixed = self._fix_syntax_error(
                    result['code'],
                    syntax_result['error'],
                    impl,
                    {}
                )
                if fixed:
                    return {'success': True, 'code': fixed}
                else:
                    return {
                        'success': False,
                        'error': f"Syntax error: {syntax_result['error']}",
                        'code': result['code']
                    }
        
        return result

