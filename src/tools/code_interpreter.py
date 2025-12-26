"""
DeQoG Code Interpreter

Executes code in a sandboxed environment, validates syntax,
and captures runtime behavior and error information.
"""

import ast
import sys
import os
import subprocess
import tempfile
import traceback
from typing import Any, Dict, Optional, Tuple
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from .base_tool import BaseTool
from ..utils.logger import get_logger

logger = get_logger("code_interpreter")


class CodeInterpreter(BaseTool):
    """
    Code Interpreter Tool.
    
    Executes Python code in a sandboxed environment with:
    - Syntax validation
    - Safe execution
    - Timeout protection
    - Output capture
    """
    
    def __init__(
        self,
        timeout: int = 5,
        sandbox_enabled: bool = True,
        max_memory_mb: int = 256
    ):
        """
        Initialize the code interpreter.
        
        Args:
            timeout: Execution timeout in seconds
            sandbox_enabled: Whether to use sandboxed execution
            max_memory_mb: Maximum memory limit in MB
        """
        super().__init__(
            name="code_interpreter",
            description="Executes Python code in a sandboxed environment"
        )
        
        self.timeout = timeout
        self.sandbox_enabled = sandbox_enabled
        self.max_memory_mb = max_memory_mb
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute code and return results.
        
        Args:
            params: Dictionary containing:
                - code: Code string to execute
                - test_input: Optional test input
                - timeout: Optional timeout override
                
        Returns:
            Dictionary with execution results
        """
        code = params.get('code', '')
        test_input = params.get('test_input', None)
        timeout = params.get('timeout', self.timeout)
        
        # First validate syntax
        syntax_result = self.validate_syntax(code)
        if not syntax_result['valid']:
            return {
                'success': False,
                'output': None,
                'error': syntax_result['error'],
                'error_type': 'SyntaxError'
            }
        
        # Execute the code
        if self.sandbox_enabled:
            return self.execute_in_sandbox(code, test_input, timeout)
        else:
            return self.execute_directly(code, test_input, timeout)
    
    def validate_syntax(self, code: str) -> Dict[str, Any]:
        """
        Validate Python syntax.
        
        Args:
            code: Code string to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            ast.parse(code)
            return {
                'valid': True,
                'error': None
            }
        except SyntaxError as e:
            return {
                'valid': False,
                'error': f"Syntax error at line {e.lineno}: {e.msg}",
                'line': e.lineno,
                'offset': e.offset
            }
    
    def execute_in_sandbox(
        self,
        code: str,
        test_input: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute code in a sandboxed subprocess.
        
        Args:
            code: Code to execute
            test_input: Optional input to provide
            timeout: Execution timeout
            
        Returns:
            Execution results
        """
        timeout = timeout or self.timeout
        
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute in subprocess
            result = subprocess.run(
                [sys.executable, temp_file],
                input=str(test_input) if test_input is not None else None,
                capture_output=True,
                timeout=timeout,
                text=True,
                env={**os.environ, 'PYTHONPATH': ''}
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'output': result.stdout.strip(),
                    'error': None,
                    'return_code': result.returncode
                }
            else:
                return {
                    'success': False,
                    'output': result.stdout.strip() if result.stdout else None,
                    'error': result.stderr.strip(),
                    'error_type': 'RuntimeError',
                    'return_code': result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': None,
                'error': f'Execution timeout after {timeout} seconds',
                'error_type': 'TimeoutError'
            }
        except Exception as e:
            return {
                'success': False,
                'output': None,
                'error': str(e),
                'error_type': type(e).__name__
            }
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def execute_directly(
        self,
        code: str,
        test_input: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute code directly in current process (less safe).
        
        Args:
            code: Code to execute
            test_input: Optional input
            timeout: Timeout (not fully supported in direct execution)
            
        Returns:
            Execution results
        """
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        # Create a restricted global namespace
        restricted_globals = {
            '__builtins__': {
                'print': print,
                'range': range,
                'len': len,
                'int': int,
                'float': float,
                'str': str,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'bool': bool,
                'max': max,
                'min': min,
                'sum': sum,
                'abs': abs,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'any': any,
                'all': all,
                'isinstance': isinstance,
                'type': type,
                'None': None,
                'True': True,
                'False': False,
            }
        }
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, restricted_globals)
            
            return {
                'success': True,
                'output': stdout_capture.getvalue().strip(),
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'output': stdout_capture.getvalue().strip() if stdout_capture.getvalue() else None,
                'error': f"{type(e).__name__}: {str(e)}",
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
    
    def execute_function(
        self,
        code: str,
        function_name: str,
        args: Tuple = (),
        kwargs: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute a specific function from the code.
        
        Args:
            code: Code containing the function
            function_name: Name of the function to call
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Execution results with function return value
        """
        kwargs = kwargs or {}
        
        # Wrap the code to call the function
        wrapper_code = f"""
{code}

__result__ = {function_name}(*{args!r}, **{kwargs!r})
print(__result__)
"""
        
        result = self.execute_in_sandbox(wrapper_code)
        
        if result['success']:
            # Try to parse the output as the return value
            try:
                result['return_value'] = eval(result['output'])
            except:
                result['return_value'] = result['output']
        
        return result
    
    def check_code_safety(self, code: str) -> Dict[str, Any]:
        """
        Check code for potentially unsafe operations.
        
        Args:
            code: Code to check
            
        Returns:
            Safety check results
        """
        unsafe_patterns = [
            ('import os', 'OS module import'),
            ('import sys', 'Sys module import'),
            ('import subprocess', 'Subprocess module import'),
            ('open(', 'File operation'),
            ('exec(', 'Dynamic execution'),
            ('eval(', 'Dynamic evaluation'),
            ('__import__', 'Dynamic import'),
            ('globals()', 'Global namespace access'),
            ('locals()', 'Local namespace access'),
        ]
        
        warnings = []
        for pattern, description in unsafe_patterns:
            if pattern in code:
                warnings.append({
                    'pattern': pattern,
                    'description': description
                })
        
        return {
            'safe': len(warnings) == 0,
            'warnings': warnings
        }
    
    def get_function_signature(self, code: str, function_name: str) -> Optional[str]:
        """
        Extract function signature from code.
        
        Args:
            code: Code containing the function
            function_name: Name of the function
            
        Returns:
            Function signature string or None
        """
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Build signature
                    args = []
                    
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        args.append(arg_str)
                    
                    returns = ""
                    if node.returns:
                        returns = f" -> {ast.unparse(node.returns)}"
                    
                    return f"def {function_name}({', '.join(args)}){returns}"
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract signature: {e}")
            return None

