"""
Tests for FSM Controller
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.fsm_controller import StateController, SystemState, TransitionAction


class MockConfig:
    """Mock configuration for testing."""
    
    class FSM:
        max_retries = 3
        enable_rollback = True
        decision_temperature = 0.1
    
    fsm = FSM()


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def generate(self, prompt, temperature=0.7):
        """Mock generate method."""
        mock_response = MagicMock()
        mock_response.content = '''
        {
            "action": "TRANSITION",
            "next_state": "STATE_2_DIVERSITY_IDEATION",
            "reason": "All conditions met"
        }
        '''
        return mock_response


class TestStateController(unittest.TestCase):
    """Test cases for StateController."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMClient()
        self.mock_config = MockConfig()
        self.controller = StateController(
            llm_client=self.mock_llm,
            config=self.mock_config
        )
    
    def test_initial_state(self):
        """Test that initial state is STATE_1_UNDERSTANDING."""
        self.assertEqual(
            self.controller.current_state,
            SystemState.STATE_1_UNDERSTANDING
        )
    
    def test_state_transition(self):
        """Test basic state transition."""
        self.controller.execute_transition(
            SystemState.STATE_2_DIVERSITY_IDEATION,
            {'data': 'test'}
        )
        
        self.assertEqual(
            self.controller.current_state,
            SystemState.STATE_2_DIVERSITY_IDEATION
        )
    
    def test_retry_mechanism(self):
        """Test retry mechanism."""
        initial_count = self.controller.retry_count[self.controller.current_state]
        
        result = self.controller.handle_retry({'error': 'test error'})
        
        self.assertTrue(result)
        self.assertEqual(
            self.controller.retry_count[self.controller.current_state],
            initial_count + 1
        )
    
    def test_retry_exceeds_max(self):
        """Test retry limit enforcement."""
        # Exhaust retries
        for _ in range(self.mock_config.fsm.max_retries + 1):
            self.controller.retry_count[self.controller.current_state] += 1
        
        result = self.controller.handle_retry({'error': 'test'})
        
        self.assertFalse(result)
    
    def test_rollback(self):
        """Test rollback mechanism."""
        # Move to state 3
        self.controller.current_state = SystemState.STATE_3_CODE_SYNTHESIS
        
        # Rollback to state 2
        self.controller.handle_rollback(
            SystemState.STATE_2_DIVERSITY_IDEATION,
            "Insufficient diversity"
        )
        
        self.assertEqual(
            self.controller.current_state,
            SystemState.STATE_2_DIVERSITY_IDEATION
        )
    
    def test_get_context(self):
        """Test context retrieval."""
        context = self.controller.get_context()
        
        self.assertIn('current_state', context)
        self.assertIn('memory', context)
        self.assertIn('transition_history', context)
        self.assertIn('retry_counts', context)
    
    def test_reset(self):
        """Test controller reset."""
        # Move to different state
        self.controller.current_state = SystemState.STATE_3_CODE_SYNTHESIS
        self.controller.retry_count[SystemState.STATE_3_CODE_SYNTHESIS] = 2
        
        # Reset
        self.controller.reset()
        
        self.assertEqual(
            self.controller.current_state,
            SystemState.STATE_1_UNDERSTANDING
        )
    
    def test_is_complete(self):
        """Test completion check."""
        self.assertFalse(self.controller.is_complete())
        
        self.controller.current_state = SystemState.STATE_COMPLETE
        self.assertTrue(self.controller.is_complete())
        
        self.controller.current_state = SystemState.STATE_ERROR
        self.assertTrue(self.controller.is_complete())


class TestSystemState(unittest.TestCase):
    """Test cases for SystemState enum."""
    
    def test_state_ordering(self):
        """Test that states are properly ordered."""
        states = list(SystemState)
        
        self.assertEqual(states[0], SystemState.STATE_1_UNDERSTANDING)
        self.assertEqual(states[1], SystemState.STATE_2_DIVERSITY_IDEATION)
        self.assertEqual(states[2], SystemState.STATE_3_CODE_SYNTHESIS)
        self.assertEqual(states[3], SystemState.STATE_4_QUALITY_VALIDATION)
        self.assertEqual(states[4], SystemState.STATE_5_COLLECTION)


if __name__ == '__main__':
    unittest.main()

