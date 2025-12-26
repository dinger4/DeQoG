"""
DeQoG Dynamic Prompt Generator

Generates state-specific prompts by adapting to current context,
history, and feedback information.
"""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path

from .base_tool import BaseTool
from ..utils.logger import get_logger

logger = get_logger("prompt_generator")


class DynamicPromptGenerator(BaseTool):
    """
    Dynamic Prompt Generator Tool.
    
    Generates context-aware prompts for each state in the DeQoG FSM.
    Integrates task information, generation history, and execution feedback
    into prompts for optimal LLM performance.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the prompt generator.
        
        Args:
            template_dir: Directory containing prompt templates
        """
        super().__init__(
            name="dynamic_prompt_generator",
            description="Generates state-specific prompts for LLM agents"
        )
        self.template_dir = Path(template_dir) if template_dir else None
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Load prompt templates from files or use defaults."""
        if self.template_dir and self.template_dir.exists():
            templates = {}
            for template_file in self.template_dir.glob("*.json"):
                with open(template_file, 'r', encoding='utf-8') as f:
                    templates[template_file.stem] = json.load(f)
            return templates
        
        # Return default templates
        return self._get_default_templates()
    
    def _get_default_templates(self) -> Dict[str, Dict[str, str]]:
        """Get default prompt templates."""
        return {
            "STATE_1_UNDERSTANDING": {
                "main": """You are an expert software engineer analyzing a programming task.

## Task Description
{task_description}

## Your Objectives
1. Fully understand the problem requirements
2. Identify input/output specifications
3. Recognize edge cases and constraints
4. Collect relevant algorithmic knowledge

## Instructions
Analyze the task and provide:
1. **Problem Summary**: Clear, concise description of what needs to be solved
2. **Input/Output Format**: Expected input format and output format
3. **Constraints**: Any constraints or limitations mentioned
4. **Edge Cases**: Potential edge cases to consider
5. **Algorithmic Approaches**: Possible algorithms or approaches to solve this

{context_info}

Please provide your analysis in a structured format.""",
                
                "with_feedback": """Based on the previous analysis that had issues:
{feedback}

Please provide an improved analysis addressing the above concerns."""
            },
            
            "STATE_2_THOUGHT_GENERATION": {
                "main": """You are generating diverse algorithmic approaches for a programming problem.

## Problem Understanding
{problem_understanding}

## Knowledge Base
{knowledge}

## Already Generated Approaches
{existing_thoughts}

## Your Task
Generate a NEW and DIFFERENT algorithmic approach that:
1. Uses a distinct algorithmic paradigm from existing approaches
2. Has different time/space complexity tradeoffs
3. Takes a unique perspective on solving the problem

Iteration: {iteration}

Describe your approach in natural language, explaining:
- The core algorithmic idea
- Why this approach is different from existing ones
- Expected time and space complexity
- Any trade-offs or limitations""",
                
                "diversity_prompt": """The previous approaches were too similar. 
Generate a COMPLETELY DIFFERENT approach that:
- Uses a different data structure
- Employs a different algorithmic paradigm
- Has distinct characteristics from: {similar_approaches}"""
            },
            
            "STATE_2_SOLUTION_GENERATION": {
                "main": """Convert the following algorithmic approach into a detailed solution strategy.

## Algorithmic Approach
{thought}

## Knowledge Context
{knowledge}

## Your Task
Create a detailed solution strategy that includes:
1. **Data Structures**: What data structures will be used and why
2. **Pseudocode**: High-level pseudocode outlining the solution
3. **Key Steps**: Numbered steps explaining the algorithm flow
4. **Complexity Analysis**: Time and space complexity
5. **Implementation Notes**: Important considerations for implementation

Iteration: {iteration}

Provide the solution strategy:"""
            },
            
            "STATE_2_IMPLEMENTATION_PLANNING": {
                "main": """Plan the concrete implementation details for the following solution.

## Solution Strategy
{solution}

## Variation Hints
{variation_hints}

## Your Task
Create an implementation plan that specifies:
1. **Language-specific constructs**: List, dict, set usage
2. **Control flow**: Recursion vs iteration, loop types
3. **Built-in functions**: Which standard library functions to use
4. **Variable naming**: Key variable names and their purposes
5. **Function signature**: Expected function signature
6. **Code structure**: Overall code organization

Iteration: {iteration}

Provide the implementation plan:"""
            },
            
            "STATE_3_CODE_SYNTHESIS": {
                "main": """Generate executable Python code based on the implementation plan.

## Task Description
{task_description}

## Implementation Plan
{implementation_plan}

## Function Signature
{function_signature}

## Requirements
1. Code must be syntactically correct Python
2. Follow the implementation plan precisely
3. Include appropriate comments
4. Handle edge cases as identified
5. Use meaningful variable names

{rollback_warnings}

Generate the complete, executable Python code:

```python""",
                
                "with_error": """The previous code had the following issues:
{error_info}

Please fix the code while maintaining the same algorithmic approach:

```python"""
            },
            
            "STATE_4_QUALITY_VALIDATION": {
                "main": """Analyze the test failure and provide a fix.

## Original Code
```python
{code}
```

## Test Failures
{test_failures}

## Error Messages
{error_messages}

## Your Task
1. Identify the root cause of each failure
2. Explain why the code failed
3. Provide the corrected code

Provide your analysis and the fixed code:""",
                
                "debug": """Debug the following code based on execution feedback.

## Code
```python
{code}
```

## Execution Result
{execution_result}

## Expected Behavior
{expected_behavior}

Identify the bug and provide the fix:"""
            }
        }
    
    def execute(self, params: Dict[str, Any]) -> str:
        """
        Generate a prompt based on the current state and context.
        
        Args:
            params: Dictionary containing:
                - state: Current state name
                - task_info: Task information
                - context: Additional context
                - template_type: Type of template to use (optional)
                
        Returns:
            Generated prompt string
        """
        state = params.get('state', '')
        task_info = params.get('task_info', {})
        context = params.get('context', {})
        template_type = params.get('template_type', 'main')
        
        # Get the appropriate template
        state_templates = self.templates.get(state, {})
        template = state_templates.get(template_type, state_templates.get('main', ''))
        
        if not template:
            logger.warning(f"No template found for state: {state}")
            template = self._create_fallback_template(state, task_info)
        
        # Build the prompt
        prompt = self._fill_template(template, task_info, context)
        
        # Add context information
        prompt = self._add_context_info(prompt, context)
        
        # Add rollback warnings if any
        if context.get('rollback_warnings'):
            prompt = self._add_rollback_warnings(prompt, context['rollback_warnings'])
        
        return prompt
    
    def _fill_template(
        self,
        template: str,
        task_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Fill template with provided values."""
        # Merge task_info and context for substitution
        all_vars = {**task_info, **context}
        
        # Convert complex objects to strings
        for key, value in all_vars.items():
            if isinstance(value, (dict, list)):
                all_vars[key] = json.dumps(value, indent=2, default=str)
            elif value is None:
                all_vars[key] = ''
        
        # Safe format - don't fail on missing keys
        try:
            return template.format_map(SafeDict(all_vars))
        except Exception as e:
            logger.warning(f"Template formatting error: {e}")
            return template
    
    def _add_context_info(self, prompt: str, context: Dict[str, Any]) -> str:
        """Add context information to the prompt."""
        context_parts = []
        
        if context.get('history'):
            context_parts.append("## Previous Attempts")
            for i, h in enumerate(context['history'][-3:], 1):  # Last 3 entries
                context_parts.append(f"{i}. From {h.get('from', 'unknown')} to {h.get('to', 'unknown')}")
        
        if context.get('feedback'):
            context_parts.append("\n## Feedback from Previous Attempts")
            for fb in context['feedback'][-3:]:  # Last 3 feedback entries
                context_parts.append(f"- {fb.get('feedback', {}).get('message', 'No message')}")
        
        if context_parts:
            context_str = "\n".join(context_parts)
            prompt = prompt.replace("{context_info}", context_str)
        else:
            prompt = prompt.replace("{context_info}", "")
        
        return prompt
    
    def _add_rollback_warnings(self, prompt: str, warnings: List[str]) -> str:
        """Add rollback warnings to the prompt."""
        if not warnings:
            return prompt.replace("{rollback_warnings}", "")
        
        warning_str = "\n## ⚠️ Previous Attempts Failed\n"
        warning_str += "Avoid these mistakes:\n"
        for warning in warnings:
            warning_str += f"- {warning}\n"
        
        return prompt.replace("{rollback_warnings}", warning_str)
    
    def _create_fallback_template(
        self,
        state: str,
        task_info: Dict[str, Any]
    ) -> str:
        """Create a fallback template for unknown states."""
        return f"""You are working on state: {state}

Task Information:
{json.dumps(task_info, indent=2, default=str)}

Please proceed with the appropriate action for this state."""
    
    def get_template(self, state: str, template_type: str = 'main') -> Optional[str]:
        """
        Get a specific template.
        
        Args:
            state: State name
            template_type: Template type
            
        Returns:
            Template string or None
        """
        return self.templates.get(state, {}).get(template_type)
    
    def set_template(self, state: str, template_type: str, template: str):
        """
        Set a custom template.
        
        Args:
            state: State name
            template_type: Template type
            template: Template string
        """
        if state not in self.templates:
            self.templates[state] = {}
        self.templates[state][template_type] = template


class SafeDict(dict):
    """Dictionary that returns placeholder for missing keys."""
    
    def __missing__(self, key):
        return f"{{{key}}}"

