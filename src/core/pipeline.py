"""
DeQoG Main Pipeline

Orchestrates the complete N-version code generation workflow
through the five-state FSM.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from .fsm_controller import StateController, SystemState, TransitionAction
from .context_memory import ContextMemory
from ..agents.understanding_agent import UnderstandingAgent
from ..agents.diversity_agent import DiversityEnhancingAgent
from ..agents.code_generating_agent import CodeGeneratingAgent
from ..agents.evaluating_agent import EvaluatingAgent
from ..tools.prompt_generator import DynamicPromptGenerator
from ..tools.diversity_evaluator import DiversityEvaluator
from ..tools.code_interpreter import CodeInterpreter
from ..tools.test_executor import TestExecutor
from ..tools.debugger import Debugger
from ..tools.knowledge_search import KnowledgeSearch
from ..tools.code_collector import CodeCollector
from ..algorithms.quality_assurance import QualityAssuranceEngine
from ..utils.logger import get_logger, setup_logger
from ..utils.config import Config

logger = get_logger("pipeline")


class DeQoGPipeline:
    """
    DeQoG Main Pipeline.
    
    Coordinates the five-state workflow:
    1. Understanding: Problem analysis and knowledge collection
    2. Diversity Ideation: Multi-level diverse idea generation (HILE/IRQN)
    3. Code Synthesis: Code generation from implementation plans
    4. Quality Validation: Testing and iterative refinement
    5. Collection: Final N-version code collection
    
    Attributes:
        config: Pipeline configuration
        state_controller: FSM state controller
        agents: Dictionary of state agents
        tools: Dictionary of available tools
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        llm_client=None
    ):
        """
        Initialize the DeQoG pipeline.
        
        Args:
            config: Configuration object
            llm_client: LLM client (created from config if not provided)
        """
        self.config = config or Config({})
        
        # Setup logging
        setup_logger(self.config.get('logging', {}))
        
        # Initialize LLM client
        self.llm_client = llm_client or self._init_llm_client()
        
        # Initialize state controller
        self.state_controller = StateController(
            llm_client=self.llm_client,
            config=self.config
        )
        
        # Initialize tools
        self.tools = self._init_tools()
        
        # Initialize agents
        self.agents = self._init_agents()
        
        # Generation statistics
        self._stats = {
            'generations': 0,
            'total_versions': 0,
            'total_time': 0
        }
        
        logger.info("DeQoG Pipeline initialized")
    
    def _init_llm_client(self):
        """Initialize the LLM client from configuration."""
        from ..utils.llm_client import LLMClientFactory
        
        llm_config = self.config.llm
        
        # Determine provider from model name
        model_name = llm_config.model_name
        if 'gpt' in model_name.lower():
            provider = 'openai'
        elif 'claude' in model_name.lower():
            provider = 'anthropic'
        else:
            provider = 'openai'  # Default
        
        return LLMClientFactory.create(
            provider=provider,
            model_name=model_name,
            api_key=llm_config.api_key,
            api_base=llm_config.api_base if hasattr(llm_config, 'api_base') else None,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            timeout=llm_config.timeout if hasattr(llm_config, 'timeout') else 60
        )
    
    def _init_tools(self) -> Dict[str, Any]:
        """Initialize all tools."""
        tools_config = self.config.get('tools', {})
        
        return {
            'prompt_generator': DynamicPromptGenerator(
                template_dir=self.config.get('data.prompts')
            ),
            'diversity_evaluator': DiversityEvaluator(
                model_name=tools_config.get('diversity_evaluator', {}).get(
                    'model_name', 'microsoft/codebert-base'
                ),
                similarity_threshold=tools_config.get('diversity_evaluator', {}).get(
                    'similarity_threshold', 0.7
                ),
                llm_client=self.llm_client
            ),
            'code_interpreter': CodeInterpreter(
                timeout=tools_config.get('code_interpreter', {}).get('timeout', 5),
                sandbox_enabled=tools_config.get('code_interpreter', {}).get(
                    'sandbox_enabled', True
                )
            ),
            'test_executor': TestExecutor(
                parallel=tools_config.get('test_executor', {}).get('parallel', True),
                max_workers=tools_config.get('test_executor', {}).get('max_workers', 4),
                timeout_per_test=tools_config.get('test_executor', {}).get(
                    'timeout_per_test', 5
                )
            ),
            'debugger': Debugger(
                llm_client=self.llm_client,
                max_analysis_depth=tools_config.get('debugger', {}).get(
                    'max_analysis_depth', 3
                )
            ),
            'knowledge_search': KnowledgeSearch(
                knowledge_base_dir=self.config.get('knowledge_bases', {}).get(
                    'algorithmic_patterns'
                ),
                llm_client=self.llm_client
            ),
            'code_collector': CodeCollector()
        }
    
    def _init_agents(self) -> Dict[str, Any]:
        """Initialize all agents."""
        return {
            'understanding': UnderstandingAgent(
                llm_client=self.llm_client,
                available_tools={
                    'knowledge_search': self.tools['knowledge_search'],
                    'dynamic_prompt_generator': self.tools['prompt_generator']
                }
            ),
            'diversity_enhancing': DiversityEnhancingAgent(
                llm_client=self.llm_client,
                diversity_evaluator=self.tools['diversity_evaluator'],
                knowledge_search=self.tools['knowledge_search'],
                dynamic_prompt_generator=self.tools['prompt_generator'],
                config=self.config.diversity
            ),
            'code_generating': CodeGeneratingAgent(
                llm_client=self.llm_client,
                available_tools={
                    'code_interpreter': self.tools['code_interpreter'],
                    'diversity_evaluator': self.tools['diversity_evaluator']
                },
                config=self.config
            ),
            'evaluating': EvaluatingAgent(
                llm_client=self.llm_client,
                available_tools={
                    'test_executor': self.tools['test_executor'],
                    'debugger': self.tools['debugger'],
                    'code_interpreter': self.tools['code_interpreter']
                },
                config=self.config
            )
        }
    
    def generate_n_versions(
        self,
        task_description: str,
        test_cases: List[Dict[str, Any]],
        n: int = 5
    ) -> Dict[str, Any]:
        """
        Generate N diverse versions of fault-tolerant code.
        
        Args:
            task_description: Description of the programming task
            test_cases: List of test cases for validation
            n: Number of versions to generate
            
        Returns:
            Dictionary containing:
            - n_version_codes: List of generated code versions
            - diversity_metrics: Diversity evaluation results
            - quality_metrics: Quality evaluation results
            - generation_metadata: Generation statistics
        """
        start_time = datetime.now()
        logger.info(f"Starting N-version generation (n={n})")
        
        # Reset state controller
        self.state_controller.reset()
        
        # Set task in context
        self.state_controller.context_memory.set_task(
            task_description=task_description,
            test_cases=test_cases,
            n_versions=n
        )
        
        try:
            # State 1: Understanding
            understanding_result = self._execute_state_1(task_description)
            
            # State 2: Diversity Ideation
            diverse_ideas = self._execute_state_2(understanding_result, n)
            
            # State 3: Code Synthesis
            generated_codes = self._execute_state_3(diverse_ideas)
            
            # State 4: Quality Validation
            validated_codes = self._execute_state_4(generated_codes, test_cases)
            
            # State 5: Collection
            final_result = self._execute_state_5(validated_codes)
            
            # Add generation metadata
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            final_result['generation_metadata'] = {
                'task_description': task_description[:200] + '...' if len(task_description) > 200 else task_description,
                'n_target': n,
                'n_generated': len(final_result.get('n_version_codes', [])),
                'generation_time': generation_time,
                'timestamp': end_time.isoformat(),
                'state_history': [
                    {
                        'from': h['from_state'].name if hasattr(h['from_state'], 'name') else str(h['from_state']),
                        'to': h['to_state'].name if hasattr(h['to_state'], 'name') else str(h['to_state']),
                        'action': h['action'].name if hasattr(h['action'], 'name') else str(h['action'])
                    }
                    for h in self.state_controller.transition_history
                ]
            }
            
            # Update statistics
            self._stats['generations'] += 1
            self._stats['total_versions'] += len(final_result.get('n_version_codes', []))
            self._stats['total_time'] += generation_time
            
            logger.info(
                f"Generation completed in {generation_time:.2f}s. "
                f"Generated {len(final_result.get('n_version_codes', []))} versions."
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _execute_state_1(self, task_description: str) -> Dict[str, Any]:
        """
        State 1: Problem Understanding.
        
        Args:
            task_description: Task description
            
        Returns:
            Understanding result
        """
        logger.info("Executing State 1: Understanding")
        
        context = self.state_controller.context_memory.get_state_context(
            SystemState.STATE_1_UNDERSTANDING
        )
        
        agent = self.agents['understanding']
        result = agent.process({'task_description': task_description}, context)
        
        # Update context
        self.state_controller.context_memory.update_context(
            SystemState.STATE_1_UNDERSTANDING,
            result
        )
        
        # Transition to next state
        self.state_controller.execute_transition(
            SystemState.STATE_2_DIVERSITY_IDEATION,
            result
        )
        
        return result
    
    def _execute_state_2(
        self,
        understanding_result: Dict[str, Any],
        n: int
    ) -> Dict[str, Any]:
        """
        State 2: Diversity Ideation (HILE + IRQN).
        
        Args:
            understanding_result: Result from State 1
            n: Target number of versions
            
        Returns:
            Diverse ideas result
        """
        logger.info("Executing State 2: Diversity Ideation")
        
        context = self.state_controller.context_memory.get_state_context(
            SystemState.STATE_2_DIVERSITY_IDEATION
        )
        
        agent = self.agents['diversity_enhancing']
        diverse_ideas = agent.process(understanding_result, context)
        
        # Check diversity threshold
        diversity_threshold = self.config.diversity.threshold
        diversity_scores = diverse_ideas.get('diversity_scores', {})
        
        # Get semantic diversity score
        sem_div = diversity_scores.get('implementations_semantic_diversity', 0)
        
        retry_count = 0
        max_retries = self.config.fsm.max_retries
        
        while sem_div < diversity_threshold and retry_count < max_retries:
            logger.warning(
                f"Diversity {sem_div:.3f} below threshold {diversity_threshold}. "
                f"Retry {retry_count + 1}/{max_retries}"
            )
            
            # Add feedback for retry
            self.state_controller.context_memory.add_feedback(
                SystemState.STATE_2_DIVERSITY_IDEATION,
                {
                    'type': 'low_diversity',
                    'message': f'Diversity score {sem_div:.3f} below threshold',
                    'scores': diversity_scores
                }
            )
            
            # Retry with enhanced IRQN
            diverse_ideas = agent.process(understanding_result, context)
            diversity_scores = diverse_ideas.get('diversity_scores', {})
            sem_div = diversity_scores.get('implementations_semantic_diversity', 0)
            retry_count += 1
        
        # Update context
        self.state_controller.context_memory.update_context(
            SystemState.STATE_2_DIVERSITY_IDEATION,
            diverse_ideas
        )
        
        # Transition
        self.state_controller.execute_transition(
            SystemState.STATE_3_CODE_SYNTHESIS,
            diverse_ideas
        )
        
        return diverse_ideas
    
    def _execute_state_3(
        self,
        diverse_ideas: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        State 3: Code Synthesis.
        
        Args:
            diverse_ideas: Diverse implementation plans from State 2
            
        Returns:
            Generated codes result
        """
        logger.info("Executing State 3: Code Synthesis")
        
        context = self.state_controller.context_memory.get_state_context(
            SystemState.STATE_3_CODE_SYNTHESIS
        )
        
        # Add function signature if available
        prev_context = context.get('previous_states', {})
        state_1_data = prev_context.get('STATE_1_UNDERSTANDING', [])
        if state_1_data:
            latest = state_1_data[-1].get('data', {})
            context['function_signature'] = latest.get('function_signature', '')
        
        agent = self.agents['code_generating']
        generated_codes = agent.process(diverse_ideas, context)
        
        # Check for failures and handle rollback if needed
        if not generated_codes.get('success'):
            logger.warning("Code synthesis failed, triggering rollback")
            self.state_controller.handle_rollback(
                SystemState.STATE_2_DIVERSITY_IDEATION,
                "Code synthesis failed"
            )
            # In a full implementation, we would re-run State 2 here
        
        # Update context
        self.state_controller.context_memory.update_context(
            SystemState.STATE_3_CODE_SYNTHESIS,
            generated_codes
        )
        
        # Transition
        self.state_controller.execute_transition(
            SystemState.STATE_4_QUALITY_VALIDATION,
            generated_codes
        )
        
        return generated_codes
    
    def _execute_state_4(
        self,
        generated_codes: Dict[str, Any],
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        State 4: Quality Validation.
        
        Args:
            generated_codes: Codes from State 3
            test_cases: Test cases for validation
            
        Returns:
            Validated codes result
        """
        logger.info("Executing State 4: Quality Validation")
        
        context = self.state_controller.context_memory.get_state_context(
            SystemState.STATE_4_QUALITY_VALIDATION
        )
        context['test_cases'] = test_cases
        
        agent = self.agents['evaluating']
        validated_codes = agent.process(generated_codes, context)
        
        # Update context
        self.state_controller.context_memory.update_context(
            SystemState.STATE_4_QUALITY_VALIDATION,
            validated_codes
        )
        
        # Transition
        self.state_controller.execute_transition(
            SystemState.STATE_5_COLLECTION,
            validated_codes
        )
        
        return validated_codes
    
    def _execute_state_5(
        self,
        validated_codes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        State 5: N-Version Code Collection.
        
        Args:
            validated_codes: Validated codes from State 4
            
        Returns:
            Final collected result
        """
        logger.info("Executing State 5: Collection")
        
        code_collector = self.tools['code_collector']
        diversity_evaluator = self.tools['diversity_evaluator']
        
        # Get validated code list
        codes = validated_codes.get('validated_codes', [])
        
        if not codes:
            logger.warning("No validated codes to collect")
            return {
                'success': False,
                'n_version_codes': [],
                'diversity_metrics': {},
                'quality_metrics': validated_codes.get('quality_metrics', {}),
                'error': 'No validated codes available'
            }
        
        # Calculate final diversity metrics
        code_strings = [c['code'] for c in codes]
        diversity_metrics = diversity_evaluator.get_diversity_report(
            code_strings,
            self.llm_client
        )
        
        # Collect codes
        result = code_collector.execute({
            'validated_codes': codes,
            'task_description': self.state_controller.context_memory.task_description,
            'diversity_metrics': diversity_metrics.get('metrics', {}),
            'quality_metrics': validated_codes.get('quality_metrics', {}),
            'metadata': self.state_controller.context_memory.get_summary()
        })
        
        # Transition to complete
        self.state_controller.execute_transition(
            SystemState.STATE_COMPLETE,
            result
        )
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self._stats,
            'avg_time_per_generation': (
                self._stats['total_time'] / self._stats['generations']
                if self._stats['generations'] > 0 else 0
            ),
            'avg_versions_per_generation': (
                self._stats['total_versions'] / self._stats['generations']
                if self._stats['generations'] > 0 else 0
            )
        }
    
    def reset(self):
        """Reset the pipeline state."""
        self.state_controller.reset()
        self.tools['code_collector'].clear()
        self.tools['diversity_evaluator'].clear_cache()
        logger.info("Pipeline reset")

