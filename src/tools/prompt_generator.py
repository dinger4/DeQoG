"""
DeQoG Dynamic Prompt Generator

Generates adaptive prompts with output format templates for deterministic
LLM output control. This is a core component of the deterministic workflow
orchestration approach.

Key Features:
1. Stage-specific prompt templates
2. Output format specifications for structured responses
3. Context injection from previous stages
4. Feedback integration for iterative refinement
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .base_tool import BaseTool
from ..utils.logger import get_logger

logger = get_logger("prompt_generator")


@dataclass
class PromptTemplate:
    """Template for generating stage-specific prompts."""
    system_role: str
    task_template: str
    output_format_template: str
    examples: Optional[List[Dict[str, str]]] = None


class DynamicPromptGenerator(BaseTool):
    """
    Dynamic Prompt Generator Tool.
    
    Generates stage-specific prompts with:
    1. Structured output format specifications
    2. Context from previous stages
    3. Feedback integration for refinement
    4. Knowledge base integration
    
    This tool is central to the deterministic workflow orchestration,
    converting unpredictable LLM outputs into deterministic templated responses.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the prompt generator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.templates = self._load_templates()
    
    def execute(self, params: Dict[str, Any]) -> str:
        """
        Execute prompt generation.
        
        Args:
            params: {
                'stage': workflow stage name,
                'task_info': task description and context,
                'history': previous generation history,
                'feedback': execution feedback for refinement,
                'output_format': desired output format template
            }
            
        Returns:
            Generated prompt string
        """
        self.validate_params(params)
        
        stage = params.get('stage', 'UNDERSTANDING')
        task_info = params.get('task_info', {})
        history = params.get('history', [])
        feedback = params.get('feedback', {})
        output_format = params.get('output_format')
        
        prompt = self._build_prompt(stage, task_info, history, feedback, output_format)
        
        logger.debug(f"Generated prompt for stage {stage}")
        return prompt
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary")
        return True
    
    def generate_stage_prompt(
        self,
        stage,
        context: Dict[str, Any],
        output_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate prompt for a specific workflow stage.
        
        This method creates prompts that enforce deterministic output formats,
        a key feature of the workflow orchestration approach.
        
        Args:
            stage: Workflow stage (enum or string)
            context: Accumulated context from previous stages
            output_format: Output format template to enforce
            
        Returns:
            Complete prompt with output format specification
        """
        stage_name = stage.name if hasattr(stage, 'name') else str(stage)
        
        # Get stage-specific template
        template = self.templates.get(stage_name, self._get_default_template(stage_name))
        
        # Build the prompt
        prompt_parts = []
        
        # 1. System role
        prompt_parts.append(f"# Role\n{template.system_role}\n")
        
        # 2. Task description
        task_prompt = self._fill_template(template.task_template, context)
        prompt_parts.append(f"# Task\n{task_prompt}\n")
        
        # 3. Context from previous stages
        if context.get('completed_stages'):
            context_section = self._format_context(context['completed_stages'])
            prompt_parts.append(f"# Previous Stage Outputs\n{context_section}\n")
        
        # 4. Output format specification (CRITICAL for deterministic output)
        if output_format:
            format_spec = self._format_output_specification(output_format)
            prompt_parts.append(f"# Output Format (MUST FOLLOW EXACTLY)\n{format_spec}\n")
        
        # 5. Examples if available
        if template.examples:
            examples_section = self._format_examples(template.examples)
            prompt_parts.append(f"# Examples\n{examples_section}\n")
        
        return "\n".join(prompt_parts)
    
    def _build_prompt(
        self,
        stage: str,
        task_info: Dict[str, Any],
        history: List[Dict[str, Any]],
        feedback: Dict[str, Any],
        output_format: Optional[Dict[str, Any]]
    ) -> str:
        """Build a complete prompt for the given stage."""
        
        # Get stage-specific template
        template = self.templates.get(stage, self._get_default_template(stage))
        
        prompt_parts = []
        
        # System role
        prompt_parts.append(template.system_role)
        
        # Task context
        prompt_parts.append(f"\n## Task Information\n{json.dumps(task_info, indent=2)}")
        
        # History if available
        if history:
            prompt_parts.append(f"\n## Previous Attempts\n{self._format_history(history)}")
        
        # Feedback if available
        if feedback:
            prompt_parts.append(f"\n## Feedback for Improvement\n{json.dumps(feedback, indent=2)}")
        
        # Output format specification
        if output_format:
            prompt_parts.append(f"\n## Required Output Format\n{self._format_output_specification(output_format)}")
        
        # Final instruction
        prompt_parts.append(template.output_format_template)
        
        return "\n".join(prompt_parts)
    
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load stage-specific prompt templates."""
        return {
            "UNDERSTANDING": PromptTemplate(
                system_role="""You are an expert problem analyst specializing in understanding 
programming tasks for fault-tolerant N-version code generation.

Your goal is to thoroughly understand the problem and identify key information 
needed for generating diverse implementations.""",
                
                task_template="""Analyze the following programming task:

{task_description}

Provide a comprehensive understanding including:
1. Problem summary - what the code should do
2. Input format and constraints
3. Output format and requirements
4. Edge cases and corner cases to handle
5. Relevant algorithmic patterns that could be applied""",
                
                output_format_template="""
Respond in the following JSON format:
```json
{
    "problem_summary": "Clear description of what the function should do",
    "input_format": "Description of input parameters and types",
    "output_format": "Description of return value and type",
    "constraints": ["constraint 1", "constraint 2", ...],
    "edge_cases": ["edge case 1", "edge case 2", ...],
    "suggested_paradigms": ["paradigm 1", "paradigm 2", ...]
}
```"""
            ),
            
            "DIVERSITY_IDEATION": PromptTemplate(
                system_role="""You are an expert algorithm designer specializing in generating 
diverse solutions using multiple algorithmic paradigms.

Your goal is to generate MAXIMALLY DIVERSE approaches to solve the problem,
covering different algorithmic paradigms, data structures, and implementation strategies.""",
                
                task_template="""Based on the problem understanding:

{problem_context}

Generate diverse algorithmic approaches at three levels:
1. THOUGHT LEVEL: Different algorithmic paradigms (DP, Greedy, Divide-Conquer, etc.)
2. SOLUTION LEVEL: Different pseudocode strategies for each thought
3. IMPLEMENTATION LEVEL: Different coding styles and techniques

Ensure maximum diversity - each approach should be fundamentally different.""",
                
                output_format_template="""
Respond in the following JSON format:
```json
{
    "thoughts": [
        {
            "id": "thought_0",
            "paradigm": "dynamic_programming",
            "description": "Detailed description of the approach",
            "complexity": {"time": "O(n^2)", "space": "O(n)"}
        },
        ...
    ],
    "solutions": [
        {
            "id": "solution_thought_0_0",
            "parent_thought": "thought_0",
            "pseudocode": "Step-by-step pseudocode",
            "data_structures": ["list", "dict"],
            "variation_focus": "time_optimized"
        },
        ...
    ],
    "implementations": [
        {
            "id": "impl_solution_thought_0_0_0",
            "parent_solution": "solution_thought_0_0",
            "style": "iterative",
            "plan": "Detailed implementation plan with specific Python constructs"
        },
        ...
    ]
}
```"""
            ),
            
            "CODE_SYNTHESIS": PromptTemplate(
                system_role="""You are an expert Python programmer specializing in 
implementing diverse algorithmic solutions.

Your goal is to translate implementation plans into working Python code
while maintaining the diversity of approaches.""",
                
                task_template="""Based on the implementation plan:

{implementation_plan}

Generate complete, working Python code that:
1. Follows the specified algorithmic approach exactly
2. Handles all edge cases identified
3. Is syntactically correct and executable
4. Maintains the intended implementation style (iterative/recursive/functional)""",
                
                output_format_template="""
Respond in the following JSON format:
```json
{
    "codes": [
        {
            "id": "code_impl_0",
            "implementation_id": "impl_solution_thought_0_0_0",
            "code": "def function_name(params):\\n    # Complete implementation\\n    ...",
            "language": "python",
            "approach_summary": "Brief description of the approach used"
        },
        ...
    ]
}
```"""
            ),
            
            "QUALITY_VALIDATION": PromptTemplate(
                system_role="""You are an expert code reviewer and debugger specializing in 
analyzing and fixing code based on test feedback.

Your goal is to identify bugs and suggest fixes to make the code pass all tests
while preserving the original algorithmic approach.""",
                
                task_template="""Code under review:

{code}

Test Results:
{test_results}

Analyze the failures and provide fixes that:
1. Address the root cause of each failure
2. Preserve the original algorithmic approach
3. Handle the edge cases properly
4. Do not change the fundamental implementation strategy""",
                
                output_format_template="""
Respond in the following JSON format:
```json
{
    "analysis": "Root cause analysis of the failures",
    "fixes": [
        {
            "issue": "Description of the issue",
            "location": "Line or section where the issue is",
            "fix": "Suggested code fix"
        },
        ...
    ],
    "fixed_code": "Complete fixed code preserving original approach"
}
```"""
            ),
            
            "COLLECTION": PromptTemplate(
                system_role="""You are a code collection specialist preparing the final
N-version code set with diversity and quality metrics.""",
                
                task_template="""Finalize the N-version code collection from:

{validated_codes}

Ensure:
1. All versions are functionally correct
2. Diversity is maximized across versions
3. Metadata is complete for each version""",
                
                output_format_template="""
Respond in the following JSON format:
```json
{
    "n_version_codes": [
        {
            "version_id": "v1",
            "code": "Complete code",
            "paradigm": "algorithm paradigm used",
            "style": "implementation style",
            "pass_rate": 1.0
        },
        ...
    ],
    "collection_summary": {
        "total_versions": 5,
        "paradigms_covered": ["dp", "greedy", ...],
        "styles_covered": ["iterative", "recursive", ...]
    }
}
```"""
            )
        }
    
    def _get_default_template(self, stage: str) -> PromptTemplate:
        """Get a default template for unknown stages."""
        return PromptTemplate(
            system_role="You are an AI assistant helping with code generation.",
            task_template="Process the following: {context}",
            output_format_template="Provide your response in JSON format."
        )
    
    def _fill_template(self, template: str, context: Dict[str, Any]) -> str:
        """Fill template placeholders with context values."""
        result = template
        
        # Simple placeholder replacement
        for key, value in context.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                if isinstance(value, (dict, list)):
                    result = result.replace(placeholder, json.dumps(value, indent=2))
                else:
                    result = result.replace(placeholder, str(value))
        
        return result
    
    def _format_output_specification(self, output_format: Dict[str, Any]) -> str:
        """Format output specification for the prompt."""
        spec = """You MUST respond with a valid JSON object following this exact structure:

```json
"""
        spec += json.dumps(output_format, indent=2)
        spec += """
```

IMPORTANT:
- Follow the structure exactly
- All required fields must be present
- Use proper JSON syntax (quoted strings, no trailing commas)
- Do not include any text outside the JSON block
"""
        return spec
    
    def _format_context(self, completed_stages: Dict[str, Any]) -> str:
        """Format context from completed stages."""
        sections = []
        for stage, output in completed_stages.items():
            sections.append(f"## {stage}\n```json\n{json.dumps(output, indent=2)}\n```")
        return "\n\n".join(sections)
    
    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format generation history."""
        if not history:
            return "No previous attempts."
        
        sections = []
        for i, entry in enumerate(history[-3:], 1):  # Show last 3 entries
            sections.append(f"Attempt {i}: {json.dumps(entry, indent=2)}")
        return "\n".join(sections)
    
    def _format_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format examples for the prompt."""
        sections = []
        for i, example in enumerate(examples, 1):
            sections.append(f"Example {i}:\nInput: {example.get('input', 'N/A')}\nOutput: {example.get('output', 'N/A')}")
        return "\n\n".join(sections)
    
    def generate_refinement_prompt(
        self,
        code: str,
        test_failures: List[Dict[str, Any]],
        iteration: int
    ) -> str:
        """
        Generate prompt for FBIR (Feedback-Based Iterative Repair).
        
        Args:
            code: Current code to refine
            test_failures: List of test failure information
            iteration: Current refinement iteration
            
        Returns:
            Refinement prompt
        """
        prompt = f"""# Feedback-Based Iterative Repair (FBIR) - Iteration {iteration}

You are fixing code based on test feedback. Your goal is to make the code pass all tests
while preserving the original algorithmic approach.

## Current Code
```python
{code}
```

## Test Failures
{json.dumps(test_failures, indent=2)}

## Instructions
1. Analyze why each test failed
2. Identify the minimal fix needed
3. Preserve the original algorithm structure
4. Do NOT change the fundamental approach

## Required Output Format
```json
{{
    "analysis": "Brief analysis of the issues",
    "fixed_code": "Complete fixed code as a single string",
    "changes_made": ["list of specific changes made"]
}}
```
"""
        return prompt
    
    def generate_diversity_enhancement_prompt(
        self,
        existing_outputs: List[Dict[str, Any]],
        level: str,
        action: str
    ) -> str:
        """
        Generate prompt for IRQN diversity enhancement.
        
        Args:
            existing_outputs: Currently generated outputs
            level: HILE level (thought/solution/implementation)
            action: IRQN action (retain/question/negate)
            
        Returns:
            Diversity enhancement prompt
        """
        if action == "question":
            instruction = """This output shows partial similarity with existing solutions.
Please REFINE it by:
1. Identifying what makes it similar
2. Proposing alternative approaches that are more distinctive
3. Enhancing its uniqueness while maintaining feasibility"""
        elif action == "negate":
            instruction = """This approach is too similar to existing solutions.
Please generate a COMPLETELY DIFFERENT solution that:
1. Uses a different algorithmic paradigm
2. Employs different data structures and control flow
3. Takes a contrasting perspective to solve the problem"""
        else:  # retain
            return None
        
        prompt = f"""# IRQN Diversity Enhancement - {action.upper()} at {level} level

## Existing Outputs
{json.dumps(existing_outputs, indent=2)}

## Instruction
{instruction}

## Required Output Format
```json
{{
    "id": "unique_id",
    "content": "The refined/new {level}-level output",
    "paradigm": "algorithmic paradigm if applicable",
    "justification": "Why this is sufficiently different"
}}
```
"""
        return prompt
