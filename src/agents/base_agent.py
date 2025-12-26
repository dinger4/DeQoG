"""
DeQoG Base LLM Agent

Abstract base class for all LLM agents in the DeQoG framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger("agents")


class BaseLLMAgent(ABC):
    """
    Base class for LLM Agents.
    
    Defines the common interface and behavior for all agents.
    Each agent is responsible for processing within a specific state
    of the FSM and can autonomously select and invoke tools.
    
    Attributes:
        llm_client: LLM client for generation
        role_prompt: System prompt defining the agent's role
        available_tools: Dictionary of tools the agent can use
    """
    
    def __init__(
        self,
        llm_client,
        role_prompt: str,
        available_tools: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the agent.
        
        Args:
            llm_client: LLM client for generation
            role_prompt: System prompt defining the agent's role
            available_tools: Dictionary of available tools
        """
        self.llm_client = llm_client
        self.role_prompt = role_prompt
        self.available_tools = available_tools or {}
        
        self._execution_count = 0
        self._last_result = None
    
    @abstractmethod
    def process(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process input and generate output.
        
        Each agent must implement this method to define its
        specific processing logic.
        
        Args:
            input_data: Input data for processing
            context: Current context from ContextMemory
            
        Returns:
            Processing results
        """
        pass
    
    def select_and_invoke_tool(
        self,
        tool_name: str,
        params: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Autonomously select and invoke a tool.
        
        Args:
            tool_name: Name of the tool to invoke
            params: Parameters for the tool
            
        Returns:
            Tool execution result or None if tool not found
        """
        if tool_name not in self.available_tools:
            logger.warning(f"Tool not found: {tool_name}")
            return None
        
        tool = self.available_tools[tool_name]
        
        try:
            result = tool.execute(params)
            logger.debug(f"Tool {tool_name} executed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed: {e}")
            raise
    
    def construct_prompt(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        state_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Construct a prompt for LLM generation.
        
        Args:
            input_data: Input data
            context: Current context
            state_info: Optional state-specific information
            
        Returns:
            Constructed prompt string
        """
        # Use prompt generator tool if available
        if 'dynamic_prompt_generator' in self.available_tools:
            prompt_gen = self.available_tools['dynamic_prompt_generator']
            return prompt_gen.execute({
                'state': state_info.get('state', '') if state_info else '',
                'task_info': input_data,
                'context': context
            })
        
        # Fallback: simple prompt construction
        prompt_parts = [self.role_prompt]
        
        if input_data:
            prompt_parts.append(f"\nInput:\n{input_data}")
        
        if context:
            # Include relevant context
            if context.get('task_description'):
                prompt_parts.append(f"\nTask: {context['task_description']}")
            
            if context.get('rollback_warnings'):
                warnings = context['rollback_warnings']
                prompt_parts.append(f"\nWarnings from previous attempts:\n{warnings}")
        
        return "\n".join(prompt_parts)
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: Prompt string
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            Generated text
        """
        kwargs = {}
        if temperature is not None:
            kwargs['temperature'] = temperature
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
        
        response = self.llm_client.generate(prompt, **kwargs)
        
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None
        """
        return self.available_tools.get(tool_name)
    
    def add_tool(self, name: str, tool: Any):
        """
        Add a tool to the agent.
        
        Args:
            name: Tool name
            tool: Tool instance
        """
        self.available_tools[name] = tool
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent execution statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'execution_count': self._execution_count,
            'available_tools': list(self.available_tools.keys())
        }

