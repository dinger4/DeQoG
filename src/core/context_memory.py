"""
DeQoG Context Memory Module

Manages cross-state context and memory for the DeQoG framework.
Stores task descriptions, generation history, feedback information, and tool outputs.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from copy import deepcopy

from ..utils.logger import get_logger

logger = get_logger("context_memory")


class ContextMemory:
    """
    Cross-state Context Memory Manager.
    
    Maintains information that needs to be passed between states,
    including task context, generation history, feedback, and rollback information.
    
    Attributes:
        task_context: Context data organized by state
        generation_history: History of all generations
        feedback_accumulation: Accumulated feedback from all iterations
        tool_outputs_cache: Cached outputs from tool executions
        rollback_info: Information about rollbacks for error avoidance
    """
    
    def __init__(self):
        """Initialize the context memory."""
        self.task_context: Dict[str, List[Dict[str, Any]]] = {}
        self.generation_history: List[Dict[str, Any]] = []
        self.feedback_accumulation: List[Dict[str, Any]] = []
        self.tool_outputs_cache: Dict[str, Any] = {}
        self.rollback_info: List[Dict[str, Any]] = []
        
        # Additional storage for specific data types
        self.task_description: Optional[str] = None
        self.test_cases: List[Dict[str, Any]] = []
        self.n_versions_target: int = 5
        
        logger.debug("ContextMemory initialized")
    
    def set_task(
        self,
        task_description: str,
        test_cases: List[Dict[str, Any]],
        n_versions: int = 5
    ):
        """
        Set the task information.
        
        Args:
            task_description: Description of the programming task
            test_cases: List of test cases
            n_versions: Target number of versions to generate
        """
        self.task_description = task_description
        self.test_cases = test_cases
        self.n_versions_target = n_versions
        
        logger.info(f"Task set: {len(test_cases)} test cases, {n_versions} versions target")
    
    def update_context(self, state: str, data: Dict[str, Any]):
        """
        Update context for a specific state.
        
        Args:
            state: State name (string or Enum)
            data: Context data to store
        """
        state_name = state.name if hasattr(state, 'name') else str(state)
        
        if state_name not in self.task_context:
            self.task_context[state_name] = []
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'data': deepcopy(data)
        }
        self.task_context[state_name].append(entry)
        
        logger.debug(f"Context updated for state {state_name}")
    
    def get_state_context(self, state: str) -> Dict[str, Any]:
        """
        Get context needed for a specific state.
        
        Args:
            state: State name
            
        Returns:
            Dictionary containing all relevant context for the state
        """
        state_name = state.name if hasattr(state, 'name') else str(state)
        
        return {
            'task_description': self.task_description,
            'test_cases': self.test_cases,
            'n_versions': self.n_versions_target,
            'task_info': self.task_context.get(state_name, []),
            'history': self.generation_history,
            'feedback': self.feedback_accumulation,
            'tool_outputs': self.tool_outputs_cache,
            'rollback_warnings': self.rollback_info,
            'previous_states': self._get_previous_states_context(state_name)
        }
    
    def _get_previous_states_context(self, current_state: str) -> Dict[str, Any]:
        """
        Get context from previous states.
        
        Args:
            current_state: Current state name
            
        Returns:
            Dictionary with context from previous states
        """
        state_order = [
            'STATE_1_UNDERSTANDING',
            'STATE_2_DIVERSITY_IDEATION',
            'STATE_3_CODE_SYNTHESIS',
            'STATE_4_QUALITY_VALIDATION',
            'STATE_5_COLLECTION'
        ]
        
        previous_context = {}
        try:
            current_idx = state_order.index(current_state)
            for i in range(current_idx):
                state = state_order[i]
                if state in self.task_context:
                    previous_context[state] = self.task_context[state]
        except ValueError:
            pass
        
        return previous_context
    
    def persist_to_next_state(
        self,
        source_state: str,
        target_state: str,
        data: Dict[str, Any]
    ):
        """
        Persist critical information during state transition.
        
        Args:
            source_state: Source state
            target_state: Target state
            data: Data to persist
        """
        source_name = source_state.name if hasattr(source_state, 'name') else str(source_state)
        target_name = target_state.name if hasattr(target_state, 'name') else str(target_state)
        
        entry = {
            'from': source_name,
            'to': target_name,
            'data': deepcopy(data),
            'timestamp': datetime.now().isoformat()
        }
        self.generation_history.append(entry)
        
        # Also update the target state's context
        self.update_context(target_state, {
            'inherited_from': source_name,
            'inherited_data': data
        })
        
        logger.debug(f"Data persisted from {source_name} to {target_name}")
    
    def add_feedback(self, state: str, feedback: Dict[str, Any]):
        """
        Add execution feedback.
        
        Args:
            state: State where the feedback originated
            feedback: Feedback information
        """
        state_name = state.name if hasattr(state, 'name') else str(state)
        
        entry = {
            'state': state_name,
            'feedback': deepcopy(feedback),
            'timestamp': datetime.now().isoformat()
        }
        self.feedback_accumulation.append(entry)
        
        logger.debug(f"Feedback added for state {state_name}: {feedback.get('type', 'unknown')}")
    
    def add_rollback_info(
        self,
        from_state: str,
        to_state: str,
        reason: str
    ):
        """
        Record rollback information.
        
        Args:
            from_state: State where rollback originated
            to_state: Target state of rollback
            reason: Reason for the rollback
        """
        from_name = from_state.name if hasattr(from_state, 'name') else str(from_state)
        to_name = to_state.name if hasattr(to_state, 'name') else str(to_state)
        
        entry = {
            'from': from_name,
            'to': to_name,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        self.rollback_info.append(entry)
        
        logger.info(f"Rollback recorded: {from_name} -> {to_name}, reason: {reason}")
    
    def cache_tool_output(self, tool_name: str, output: Any):
        """
        Cache a tool's output.
        
        Args:
            tool_name: Name of the tool
            output: Tool output to cache
        """
        if tool_name not in self.tool_outputs_cache:
            self.tool_outputs_cache[tool_name] = []
        
        self.tool_outputs_cache[tool_name].append({
            'output': deepcopy(output),
            'timestamp': datetime.now().isoformat()
        })
    
    def get_tool_outputs(self, tool_name: str) -> List[Any]:
        """
        Get cached outputs for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            List of cached outputs
        """
        entries = self.tool_outputs_cache.get(tool_name, [])
        return [e['output'] for e in entries]
    
    def get_latest_generation(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent generation entry.
        
        Returns:
            Latest generation entry or None
        """
        if self.generation_history:
            return self.generation_history[-1]
        return None
    
    def get_feedback_for_state(self, state: str) -> List[Dict[str, Any]]:
        """
        Get all feedback entries for a specific state.
        
        Args:
            state: State name
            
        Returns:
            List of feedback entries
        """
        state_name = state.name if hasattr(state, 'name') else str(state)
        return [f for f in self.feedback_accumulation if f['state'] == state_name]
    
    def get_rollback_warnings(self) -> List[str]:
        """
        Get rollback warnings for prompt enhancement.
        
        Returns:
            List of warning strings
        """
        warnings = []
        for info in self.rollback_info:
            warnings.append(
                f"Previous attempt from {info['from']} to {info['to']} failed: {info['reason']}"
            )
        return warnings
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all context data.
        
        Returns:
            Dictionary containing all context information
        """
        return {
            'task_description': self.task_description,
            'test_cases': self.test_cases,
            'n_versions': self.n_versions_target,
            'task_context': deepcopy(self.task_context),
            'generation_history': deepcopy(self.generation_history),
            'feedback': deepcopy(self.feedback_accumulation),
            'tool_outputs': deepcopy(self.tool_outputs_cache),
            'rollback_info': deepcopy(self.rollback_info)
        }
    
    def clear(self):
        """Clear all context data."""
        self.task_context = {}
        self.generation_history = []
        self.feedback_accumulation = []
        self.tool_outputs_cache = {}
        self.rollback_info = []
        self.task_description = None
        self.test_cases = []
        self.n_versions_target = 5
        
        logger.info("ContextMemory cleared")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context.
        
        Returns:
            Summary dictionary
        """
        return {
            'has_task': self.task_description is not None,
            'num_test_cases': len(self.test_cases),
            'n_versions_target': self.n_versions_target,
            'num_states_with_context': len(self.task_context),
            'generation_history_length': len(self.generation_history),
            'feedback_count': len(self.feedback_accumulation),
            'rollback_count': len(self.rollback_info),
            'cached_tools': list(self.tool_outputs_cache.keys())
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export context to a serializable dictionary.
        
        Returns:
            Serializable dictionary
        """
        return self.get_all()
    
    @classmethod
    def import_from_dict(cls, data: Dict[str, Any]) -> 'ContextMemory':
        """
        Import context from a dictionary.
        
        Args:
            data: Context dictionary
            
        Returns:
            ContextMemory instance
        """
        memory = cls()
        
        memory.task_description = data.get('task_description')
        memory.test_cases = data.get('test_cases', [])
        memory.n_versions_target = data.get('n_versions', 5)
        memory.task_context = data.get('task_context', {})
        memory.generation_history = data.get('generation_history', [])
        memory.feedback_accumulation = data.get('feedback', [])
        memory.tool_outputs_cache = data.get('tool_outputs', {})
        memory.rollback_info = data.get('rollback_info', [])
        
        return memory

