"""
DeQoG Understanding Agent

State 1 Agent: Problem understanding and information collection.
"""

import json
import re
from typing import Any, Dict, List, Optional

from .base_agent import BaseLLMAgent
from ..utils.logger import get_logger

logger = get_logger("understanding_agent")


class UnderstandingAgent(BaseLLMAgent):
    """
    Understanding & Observing Agent (State 1).
    
    Responsible for:
    1. Parsing task descriptions
    2. Collecting relevant knowledge
    3. Generating problem understanding reports
    4. Identifying edge cases and constraints
    
    Uses tools:
    - Knowledge Search
    - Dynamic Prompt Generator
    """
    
    def __init__(
        self,
        llm_client,
        available_tools: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the understanding agent.
        
        Args:
            llm_client: LLM client for generation
            available_tools: Available tools
        """
        role_prompt = """You are an expert software engineer specializing in problem analysis and requirement understanding.
Your task is to thoroughly analyze programming problems and extract all relevant information needed for implementation.
Focus on:
1. Understanding the core problem
2. Identifying input/output specifications
3. Recognizing constraints and edge cases
4. Identifying applicable algorithmic approaches"""
        
        super().__init__(
            llm_client=llm_client,
            role_prompt=role_prompt,
            available_tools=available_tools
        )
    
    def process(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process the task description and generate understanding.
        
        Args:
            input_data: Dictionary with 'task_description'
            context: Current context
            
        Returns:
            Dictionary with task understanding and collected knowledge
        """
        self._execution_count += 1
        
        task_description = input_data.get('task_description', '')
        
        if not task_description:
            return {
                'success': False,
                'error': 'No task description provided'
            }
        
        logger.info("Starting problem understanding...")
        
        # Step 1: Collect relevant knowledge
        collected_knowledge = self._collect_knowledge(task_description)
        
        # Step 2: Analyze the problem
        task_understanding = self._analyze_problem(
            task_description, collected_knowledge, context
        )
        
        # Step 3: Extract function signature
        function_signature = self._extract_function_signature(task_description)
        
        # Step 4: Identify test case patterns
        test_patterns = self._identify_test_patterns(task_description)
        
        result = {
            'success': True,
            'task_understanding': task_understanding,
            'collected_knowledge': collected_knowledge,
            'function_signature': function_signature,
            'test_patterns': test_patterns,
            'original_task': task_description
        }
        
        self._last_result = result
        logger.info("Problem understanding completed")
        
        return result
    
    def _collect_knowledge(self, task_description: str) -> Dict[str, Any]:
        """
        Collect relevant knowledge for the task.
        
        Args:
            task_description: Task description
            
        Returns:
            Collected knowledge dictionary
        """
        knowledge = {
            'algorithmic': [],
            'implementation': [],
            'fault_tolerance': []
        }
        
        knowledge_search = self.get_tool('knowledge_search')
        
        if knowledge_search:
            try:
                search_result = knowledge_search.execute({
                    'query': task_description,
                    'knowledge_type': 'all',
                    'top_k': 5
                })
                
                knowledge['algorithmic'] = search_result.get('algorithmic', [])
                knowledge['implementation'] = search_result.get('implementation', [])
                knowledge['fault_tolerance'] = search_result.get('fault_tolerance', [])
                
                logger.debug(f"Collected {len(knowledge['algorithmic'])} algorithmic patterns")
            except Exception as e:
                logger.warning(f"Knowledge search failed: {e}")
        
        return knowledge
    
    def _analyze_problem(
        self,
        task_description: str,
        knowledge: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze the problem using LLM.
        
        Args:
            task_description: Task description
            knowledge: Collected knowledge
            context: Current context
            
        Returns:
            Problem analysis dictionary
        """
        # Build the analysis prompt
        knowledge_summary = self._format_knowledge_summary(knowledge)
        
        prompt = f"""{self.role_prompt}

## Task Description
{task_description}

## Relevant Knowledge
{knowledge_summary}

## Your Analysis
Please provide a comprehensive analysis in the following JSON format:

```json
{{
    "problem_summary": "Clear, concise description of what needs to be solved",
    "input_format": {{
        "description": "Description of expected inputs",
        "types": ["list of input types"],
        "constraints": ["list of input constraints"]
    }},
    "output_format": {{
        "description": "Description of expected output",
        "type": "output type"
    }},
    "edge_cases": [
        "List of potential edge cases to handle"
    ],
    "constraints": [
        "List of constraints mentioned"
    ],
    "algorithmic_approaches": [
        {{
            "name": "Approach name",
            "description": "Brief description",
            "complexity": "Time and space complexity",
            "pros": ["advantages"],
            "cons": ["disadvantages"]
        }}
    ],
    "difficulty_assessment": "easy|medium|hard",
    "key_observations": [
        "Important observations about the problem"
    ]
}}
```

Provide your analysis:"""
        
        response = self.generate(prompt, temperature=0.3)
        
        # Parse the JSON response
        analysis = self._parse_json_response(response)
        
        return analysis
    
    def _format_knowledge_summary(self, knowledge: Dict[str, Any]) -> str:
        """Format knowledge for prompt inclusion."""
        parts = []
        
        if knowledge.get('algorithmic'):
            parts.append("### Algorithmic Patterns")
            for pattern in knowledge['algorithmic'][:3]:
                name = pattern.get('name', 'Unknown')
                desc = pattern.get('description', '')
                parts.append(f"- **{name}**: {desc}")
        
        if knowledge.get('implementation'):
            parts.append("\n### Implementation Techniques")
            for tech in knowledge['implementation'][:3]:
                name = tech.get('name', 'Unknown')
                category = tech.get('category', '')
                parts.append(f"- **{name}** ({category})")
        
        return "\n".join(parts) if parts else "No relevant knowledge found."
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = response[start:end]
                else:
                    json_str = response
            
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            
            # Return a basic structure
            return {
                'problem_summary': response[:500],
                'raw_response': response,
                'parse_error': str(e)
            }
    
    def _extract_function_signature(self, task_description: str) -> Optional[str]:
        """
        Extract function signature from task description.
        
        Args:
            task_description: Task description
            
        Returns:
            Function signature string or None
        """
        # Look for common patterns
        patterns = [
            r'def\s+\w+\s*\([^)]*\)\s*(?:->.*)?:',  # Python function definition
            r'Function signature:\s*(def\s+\w+\s*\([^)]*\))',  # Explicit signature
            r'function\s+\w+\s*\([^)]*\)',  # Generic function pattern
        ]
        
        for pattern in patterns:
            match = re.search(pattern, task_description, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        return None
    
    def _identify_test_patterns(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Identify test case patterns from examples in task description.
        
        Args:
            task_description: Task description
            
        Returns:
            List of test pattern dictionaries
        """
        patterns = []
        
        # Look for Input/Output patterns
        example_patterns = [
            r'Input:\s*(.+?)\s*Output:\s*(.+?)(?=Input:|$)',
            r'Example[^:]*:\s*Input:\s*(.+?)\s*Output:\s*(.+?)(?=Example|$)',
            r'>>> \w+\((.+?)\)\s*\n\s*(.+?)(?=>>>|$)',
        ]
        
        for pattern in example_patterns:
            matches = re.findall(pattern, task_description, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    patterns.append({
                        'input': match[0].strip(),
                        'output': match[1].strip()
                    })
        
        return patterns
    
    def validate_understanding(
        self,
        understanding: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate the quality of problem understanding.
        
        Args:
            understanding: Understanding result to validate
            
        Returns:
            Validation result
        """
        issues = []
        
        task_understanding = understanding.get('task_understanding', {})
        
        # Check required fields
        required_fields = ['problem_summary', 'input_format', 'output_format']
        for field in required_fields:
            if not task_understanding.get(field):
                issues.append(f"Missing {field}")
        
        # Check if algorithmic approaches are identified
        approaches = task_understanding.get('algorithmic_approaches', [])
        if len(approaches) < 2:
            issues.append("Insufficient algorithmic approaches identified (need at least 2)")
        
        # Check if edge cases are identified
        edge_cases = task_understanding.get('edge_cases', [])
        if len(edge_cases) < 1:
            issues.append("No edge cases identified")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'completeness_score': 1.0 - (len(issues) / 5)  # Rough completeness metric
        }

