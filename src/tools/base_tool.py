"""
DeQoG Base Tool

Abstract base class for all tools in the DeQoG framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger("tools")


class BaseTool(ABC):
    """
    Abstract base class for all DeQoG tools.
    
    All tools must implement the execute method and optionally
    the validate_params method for parameter validation.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the tool.
        
        Args:
            name: Tool name
            description: Tool description
        """
        self.name = name
        self.description = description
        self._execution_count = 0
        self._last_execution_time: Optional[datetime] = None
    
    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> Any:
        """
        Execute the tool functionality.
        
        Args:
            params: Tool parameters
            
        Returns:
            Tool execution result
        """
        pass
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate tool parameters.
        
        Args:
            params: Parameters to validate
            
        Returns:
            True if parameters are valid
        """
        return True
    
    def __call__(self, params: Dict[str, Any]) -> Any:
        """
        Make the tool callable.
        
        Args:
            params: Tool parameters
            
        Returns:
            Tool execution result
        """
        if not self.validate_params(params):
            raise ValueError(f"Invalid parameters for tool {self.name}")
        
        self._execution_count += 1
        self._last_execution_time = datetime.now()
        
        logger.debug(f"Executing tool: {self.name}")
        
        try:
            result = self.execute(params)
            logger.debug(f"Tool {self.name} executed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get tool execution statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'name': self.name,
            'execution_count': self._execution_count,
            'last_execution': self._last_execution_time.isoformat() 
                            if self._last_execution_time else None
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the tool's parameter schema.
        
        Returns:
            JSON schema for tool parameters
        """
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {}
        }


class ToolRegistry:
    """
    Registry for managing tools.
    """
    
    _tools: Dict[str, BaseTool] = {}
    
    @classmethod
    def register(cls, tool: BaseTool):
        """Register a tool."""
        cls._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return cls._tools.get(name)
    
    @classmethod
    def list_tools(cls) -> Dict[str, str]:
        """List all registered tools."""
        return {name: tool.description for name, tool in cls._tools.items()}
    
    @classmethod
    def clear(cls):
        """Clear all registered tools."""
        cls._tools.clear()

