"""
DeQoG FSM State Controller

The core state controller that manages system state transitions,
implements retry and rollback mechanisms, and maintains cross-state context.
"""

import json
from enum import Enum, auto
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger("fsm_controller")


class SystemState(Enum):
    """
    System state definitions for the DeQoG FSM.
    
    The system progresses through five main states:
    1. Understanding: Problem comprehension and information collection
    2. Diversity Ideation: Diverse solution generation
    3. Code Synthesis: Code implementation
    4. Quality Validation: Testing and debugging
    5. Collection: Final N-version code collection
    """
    STATE_1_UNDERSTANDING = auto()
    STATE_2_DIVERSITY_IDEATION = auto()
    STATE_3_CODE_SYNTHESIS = auto()
    STATE_4_QUALITY_VALIDATION = auto()
    STATE_5_COLLECTION = auto()
    STATE_ERROR = auto()
    STATE_COMPLETE = auto()


class TransitionAction(Enum):
    """State transition actions."""
    TRANSITION = auto()  # Normal state transition
    RETRY = auto()       # Retry current state
    ROLLBACK = auto()    # Rollback to previous state
    ERROR = auto()       # Unrecoverable error


class StateController:
    """
    State Controller - Core FSM logic for DeQoG.
    
    Responsibilities:
    1. Manage system state transitions
    2. Evaluate transition conditions
    3. Handle retry and rollback mechanisms
    4. Maintain cross-state context
    
    Attributes:
        current_state: Current system state
        llm_client: LLM client for decision making
        config: System configuration
        context_memory: Cross-state context memory
        transition_history: History of state transitions
        retry_count: Retry count per state
    """
    
    def __init__(self, llm_client, config):
        """
        Initialize the state controller.
        
        Args:
            llm_client: LLM client for decision making
            config: System configuration
        """
        self.current_state = SystemState.STATE_1_UNDERSTANDING
        self.llm_client = llm_client
        self.config = config
        
        # Import here to avoid circular imports
        from .context_memory import ContextMemory
        self.context_memory = ContextMemory()
        
        self.transition_history: List[Dict[str, Any]] = []
        self.retry_count: Dict[SystemState, int] = {}
        
        # Initialize retry count for each state
        for state in SystemState:
            self.retry_count[state] = 0
        
        logger.info(f"StateController initialized. Starting state: {self.current_state.name}")
    
    def evaluate_transition(
        self,
        current_output: Dict[str, Any],
        tools_result: Optional[Dict[str, Any]] = None
    ) -> Tuple[SystemState, TransitionAction]:
        """
        Evaluate whether transition conditions are met.
        
        Uses LLM as a decision maker to evaluate the current output
        and determine the next action (transition, retry, or rollback).
        
        Args:
            current_output: Current state's LLM output
            tools_result: Tool execution results
            
        Returns:
            Tuple of (next_state, action_type)
        """
        state = self.current_state
        
        # Build decision prompt
        decision_prompt = self._construct_decision_prompt(
            state, current_output, tools_result
        )
        
        # Use LLM for decision making
        try:
            response = self.llm_client.generate(
                decision_prompt,
                temperature=self.config.fsm.decision_temperature
            )
            decision_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            return SystemState.STATE_ERROR, TransitionAction.ERROR
        
        # Parse decision
        action, next_state, reason = self._parse_decision(decision_text)
        
        # Record transition
        self.transition_history.append({
            'from_state': state,
            'to_state': next_state,
            'action': action,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"State transition decision: {state.name} -> {next_state.name} ({action.name})")
        logger.debug(f"Reason: {reason}")
        
        return next_state, action
    
    def _construct_decision_prompt(
        self,
        current_state: SystemState,
        current_output: Dict[str, Any],
        tools_result: Optional[Dict[str, Any]]
    ) -> str:
        """
        Construct the state transition decision prompt.
        
        Args:
            current_state: Current system state
            current_output: Current output data
            tools_result: Tool execution results
            
        Returns:
            Decision prompt string
        """
        conditions = self._get_state_conditions(current_state)
        next_state = self._get_next_state(current_state)
        
        # Serialize output for prompt
        output_str = json.dumps(current_output, indent=2, default=str)
        tools_str = json.dumps(tools_result, indent=2, default=str) if tools_result else "None"
        
        prompt = f"""You are the state controller for a fault-tolerant code generation system.

Current State: {current_state.name}

Current Output:
{output_str}

Tools Result:
{tools_str}

Transition Conditions for {current_state.name}:
{conditions}

Next State (if conditions met): {next_state.name if next_state else "STATE_COMPLETE"}

Based on the current output and tools result, evaluate whether:
1. TRANSITION: All conditions are met, proceed to next state
2. RETRY: Some conditions not met, but recoverable (e.g., syntax error, insufficient diversity)
3. ROLLBACK: Fundamental flaw in previous decisions (e.g., wrong approach, infeasible idea)
4. ERROR: Unrecoverable error

Respond in JSON format:
{{
    "action": "TRANSITION|RETRY|ROLLBACK|ERROR",
    "next_state": "{next_state.name if next_state else 'STATE_COMPLETE'}",
    "reason": "Brief explanation of the decision"
}}
"""
        return prompt
    
    def _get_state_conditions(self, state: SystemState) -> str:
        """
        Get transition conditions for each state.
        
        Args:
            state: System state
            
        Returns:
            Transition conditions description
        """
        conditions_map = {
            SystemState.STATE_1_UNDERSTANDING: """
            - Problem description is fully parsed
            - Relevant knowledge is collected
            - No ambiguity in requirements
            - Input/output formats are clear
            - Edge cases are identified
            """,
            
            SystemState.STATE_2_DIVERSITY_IDEATION: """
            - Generated ideas have sufficient diversity (SDP > threshold)
            - Ideas are feasible and executable
            - Number of ideas meets requirement (N versions)
            - Each idea represents a distinct approach
            - Ideas cover different algorithmic paradigms
            """,
            
            SystemState.STATE_3_CODE_SYNTHESIS: """
            - All ideas are translated to executable code
            - Code passes syntax validation
            - Implementation diversity is maintained (not too similar)
            - Each code follows the corresponding solution approach
            - Code is properly formatted and documented
            """,
            
            SystemState.STATE_4_QUALITY_VALIDATION: """
            - All codes pass functional correctness tests
            - Test pass rate meets threshold (>= 0.9)
            - No critical bugs remain
            - Edge cases are handled correctly
            - Performance is acceptable
            """,
            
            SystemState.STATE_5_COLLECTION: """
            - All N versions are collected
            - Metadata is complete
            - Diversity metrics are calculated
            - All versions are validated
            """,
        }
        
        return conditions_map.get(state, "No specific conditions")
    
    def _get_next_state(self, state: SystemState) -> Optional[SystemState]:
        """
        Get the next state in the FSM flow.
        
        Args:
            state: Current state
            
        Returns:
            Next state or None if at the end
        """
        state_order = [
            SystemState.STATE_1_UNDERSTANDING,
            SystemState.STATE_2_DIVERSITY_IDEATION,
            SystemState.STATE_3_CODE_SYNTHESIS,
            SystemState.STATE_4_QUALITY_VALIDATION,
            SystemState.STATE_5_COLLECTION,
        ]
        
        try:
            current_idx = state_order.index(state)
            if current_idx < len(state_order) - 1:
                return state_order[current_idx + 1]
            else:
                return SystemState.STATE_COMPLETE
        except ValueError:
            return None
    
    def _parse_decision(
        self,
        decision_str: str
    ) -> Tuple[TransitionAction, SystemState, str]:
        """
        Parse the LLM's decision output.
        
        Args:
            decision_str: LLM response string
            
        Returns:
            Tuple of (action, next_state, reason)
        """
        try:
            # Try to extract JSON from the response
            json_start = decision_str.find('{')
            json_end = decision_str.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = decision_str[json_start:json_end]
                decision = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
            
            action = TransitionAction[decision['action']]
            next_state_name = decision['next_state']
            next_state = SystemState[next_state_name]
            reason = decision.get('reason', 'No reason provided')
            
            return action, next_state, reason
            
        except Exception as e:
            logger.error(f"Failed to parse decision: {e}")
            logger.debug(f"Raw decision string: {decision_str}")
            
            # Default to transition if parsing fails but output looks good
            next_state = self._get_next_state(self.current_state)
            if next_state:
                return TransitionAction.TRANSITION, next_state, f"Parse error, defaulting to transition: {e}"
            return TransitionAction.ERROR, SystemState.STATE_ERROR, str(e)
    
    def execute_transition(
        self,
        next_state: SystemState,
        carry_forward_data: Dict[str, Any]
    ):
        """
        Execute the state transition.
        
        Args:
            next_state: Target state
            carry_forward_data: Data to carry to the next state
        """
        logger.info(f"Executing transition: {self.current_state.name} -> {next_state.name}")
        
        # Persist critical information
        self.context_memory.persist_to_next_state(
            self.current_state,
            next_state,
            carry_forward_data
        )
        
        # Update state
        old_state = self.current_state
        self.current_state = next_state
        
        # Reset retry count for new state
        self.retry_count[next_state] = 0
        
        logger.info(f"State transitioned successfully: {old_state.name} -> {next_state.name}")
    
    def handle_retry(self, error_info: Dict[str, Any]) -> bool:
        """
        Handle recoverable errors - retry current state.
        
        Args:
            error_info: Error information for updating prompt
            
        Returns:
            True if retry is allowed, False if max retries exceeded
        """
        self.retry_count[self.current_state] += 1
        max_retries = self.config.fsm.max_retries
        
        if self.retry_count[self.current_state] > max_retries:
            logger.warning(f"Max retries exceeded for {self.current_state.name}")
            return False
        
        logger.info(
            f"Retry {self.retry_count[self.current_state]}/{max_retries} "
            f"for state {self.current_state.name}"
        )
        
        # Update context with error information for next attempt
        self.context_memory.add_feedback(self.current_state, error_info)
        
        return True
    
    def handle_rollback(
        self,
        target_state: SystemState,
        failure_reason: str
    ):
        """
        Handle fundamental flaws - rollback to previous state.
        
        Args:
            target_state: Rollback target state
            failure_reason: Reason for the failure
        """
        if not self.config.fsm.enable_rollback:
            logger.warning("Rollback is disabled in configuration")
            return
        
        logger.warning(
            f"Rollback triggered: {self.current_state.name} -> {target_state.name}"
        )
        logger.warning(f"Reason: {failure_reason}")
        
        # Record rollback information to avoid repeating the same mistake
        self.context_memory.add_rollback_info(
            from_state=self.current_state,
            to_state=target_state,
            reason=failure_reason
        )
        
        # Execute rollback
        self.current_state = target_state
        self.retry_count[target_state] = 0
    
    def _get_previous_state(self) -> SystemState:
        """
        Get the previous state in the FSM flow.
        
        Returns:
            Previous state
        """
        state_order = [
            SystemState.STATE_1_UNDERSTANDING,
            SystemState.STATE_2_DIVERSITY_IDEATION,
            SystemState.STATE_3_CODE_SYNTHESIS,
            SystemState.STATE_4_QUALITY_VALIDATION,
            SystemState.STATE_5_COLLECTION
        ]
        
        try:
            current_idx = state_order.index(self.current_state)
            if current_idx > 0:
                return state_order[current_idx - 1]
            else:
                return SystemState.STATE_ERROR
        except ValueError:
            return SystemState.STATE_ERROR
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current complete context.
        
        Returns:
            Dictionary containing all context information
        """
        return {
            'current_state': self.current_state.name,
            'memory': self.context_memory.get_all(),
            'transition_history': self.transition_history,
            'retry_counts': {k.name: v for k, v in self.retry_count.items()}
        }
    
    def reset(self):
        """Reset the state controller to initial state."""
        self.current_state = SystemState.STATE_1_UNDERSTANDING
        self.context_memory = self.context_memory.__class__()
        self.transition_history = []
        for state in SystemState:
            self.retry_count[state] = 0
        
        logger.info("StateController reset to initial state")
    
    def is_complete(self) -> bool:
        """Check if the FSM has reached a terminal state."""
        return self.current_state in [SystemState.STATE_COMPLETE, SystemState.STATE_ERROR]
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get information about the current state.
        
        Returns:
            State information dictionary
        """
        return {
            'state': self.current_state.name,
            'retry_count': self.retry_count[self.current_state],
            'max_retries': self.config.fsm.max_retries,
            'conditions': self._get_state_conditions(self.current_state),
            'next_state': self._get_next_state(self.current_state).name 
                         if self._get_next_state(self.current_state) else None
        }

