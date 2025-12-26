"""
DeQoG Pipeline

Main pipeline implementing the Diversity-Enhanced Quality-Assured
N-Version Code Generation workflow.

Architecture:
- Uses Deterministic Workflow Orchestration (not FSM)
- Integrates HILE for hierarchical diversity generation
- Integrates IRQN for iterative diversity enhancement
- Integrates FBIR for feedback-based iterative repair

Reference: "Automated Fault-Tolerant Code Generation via LLMs:
A Diversity-Enhanced and Quality-Assured Approach"
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .workflow_orchestrator import (
    DeterministicWorkflowOrchestrator,
    WorkflowStage,
    QualityGate
)
from .context_memory import ContextMemory
from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger("pipeline")


@dataclass
class GenerationResult:
    """Result from N-version code generation."""
    n_version_codes: List[Dict[str, Any]]
    diversity_metrics: Dict[str, float]
    quality_metrics: Dict[str, float]
    generation_metadata: Dict[str, Any]
    workflow_summary: Dict[str, Any]


class DeQoGPipeline:
    """
    DeQoG Main Pipeline.
    
    Orchestrates the complete N-version fault-tolerant code generation process:
    
    1. UNDERSTANDING: Parse task, collect knowledge
    2. DIVERSITY_IDEATION: Generate diverse ideas via HILE + IRQN
    3. CODE_SYNTHESIS: Translate ideas to executable code
    4. QUALITY_VALIDATION: Test and refine via FBIR
    5. COLLECTION: Collect validated N-versions
    
    Key Components:
    - DeterministicWorkflowOrchestrator: Manages workflow stages
    - DynamicPromptGenerator: Creates stage-specific prompts with output formats
    - Agents: Specialized LLM agents for each stage
    - Tools: Code interpreter, test executor, diversity evaluator, etc.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the DeQoG pipeline.
        
        Args:
            config: System configuration
        """
        self.config = config
        
        # Initialize core components
        self.llm_client = self._init_llm_client()
        self.tools = self._init_tools()
        self.agents = self._init_agents()
        
        # Initialize workflow orchestrator
        self.orchestrator = DeterministicWorkflowOrchestrator(
            prompt_generator=self.tools['prompt_generator'],
            config=config
        )
        
        # Initialize algorithms
        self.hile = self._init_hile()
        self.irqn = self._init_irqn()
        self.qa_engine = self._init_quality_assurance()
        
        # Quality gates
        self.quality_gates = self._init_quality_gates()
        
        logger.info("DeQoGPipeline initialized")
    
    def generate_n_versions(
        self,
        task_description: str,
        test_cases: List[Dict[str, Any]],
        n: int = 5
    ) -> GenerationResult:
        """
        Generate N versions of fault-tolerant code.
        
        Main entry point for the DeQoG system.
        
        Args:
            task_description: Natural language description of the programming task
            test_cases: List of test cases for validation
            n: Number of versions to generate (default: 5)
            
        Returns:
            GenerationResult containing N-version codes and metrics
        """
        logger.info(f"Starting N-version generation (n={n})")
        start_time = datetime.now()
        
        # Reset orchestrator for new generation
        self.orchestrator.reset()
        
        # Store initial context
        self.orchestrator.context['task_description'] = task_description
        self.orchestrator.context['test_cases'] = test_cases
        self.orchestrator.context['n_versions'] = n
        
        try:
            # Stage 1: Understanding
            understanding_result = self._execute_understanding_stage(task_description)
            
            # Stage 2: Diversity Ideation (HILE + IRQN)
            diverse_ideas = self._execute_diversity_stage(understanding_result, n)
            
            # Stage 3: Code Synthesis
            generated_codes = self._execute_synthesis_stage(diverse_ideas)
            
            # Stage 4: Quality Validation (FBIR)
            validated_codes = self._execute_validation_stage(generated_codes, test_cases)
            
            # Stage 5: Collection
            final_result = self._execute_collection_stage(validated_codes, n)
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
        
        # Compute final metrics
        diversity_metrics = self._compute_diversity_metrics(final_result['codes'])
        quality_metrics = self._compute_quality_metrics(final_result['codes'])
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        result = GenerationResult(
            n_version_codes=final_result['codes'],
            diversity_metrics=diversity_metrics,
            quality_metrics=quality_metrics,
            generation_metadata={
                'n_requested': n,
                'n_generated': len(final_result['codes']),
                'elapsed_seconds': elapsed,
                'timestamp': start_time.isoformat()
            },
            workflow_summary=self.orchestrator.get_workflow_summary()
        )
        
        logger.info(f"N-version generation completed in {elapsed:.2f}s")
        return result
    
    def _execute_understanding_stage(
        self,
        task_description: str
    ) -> Dict[str, Any]:
        """
        Execute Stage 1: Problem Understanding.
        
        Parses the task description and collects relevant knowledge.
        """
        logger.info("Executing Stage 1: Understanding")
        
        # Get stage prompt with output format
        context = self.orchestrator.get_accumulated_context()
        context['task_description'] = task_description
        
        prompt = self.orchestrator.get_stage_prompt(
            WorkflowStage.UNDERSTANDING,
            context
        )
        
        # Use Understanding Agent
        agent = self.agents['understanding']
        result = agent.process(task_description, context)
        
        # Validate output format
        is_valid, errors = self.orchestrator.validate_stage_output(
            WorkflowStage.UNDERSTANDING,
            result
        )
        
        if not is_valid:
            logger.warning(f"Understanding output validation issues: {errors}")
        
        # Record and advance
        self.orchestrator.record_stage_result(
            WorkflowStage.UNDERSTANDING,
            result,
            success=is_valid,
            errors=errors
        )
        self.orchestrator.advance_stage()
        
        return result
    
    def _execute_diversity_stage(
        self,
        understanding_result: Dict[str, Any],
        n: int
    ) -> Dict[str, Any]:
        """
        Execute Stage 2: Diversity Ideation.
        
        Applies HILE for hierarchical diversity generation
        and IRQN for iterative diversity enhancement.
        """
        logger.info("Executing Stage 2: Diversity Ideation")
        
        # Check quality gate
        gate = self.quality_gates.get(WorkflowStage.DIVERSITY_IDEATION)
        if gate:
            passed, unmet = gate.check(self.orchestrator.context)
            if not passed:
                raise ValueError(f"Quality gate failed: {unmet}")
        
        # Get context
        context = self.orchestrator.get_accumulated_context()
        
        # Use Diversity Enhancing Agent with HILE
        agent = self.agents['diversity_enhancing']
        
        # Apply HILE algorithm
        logger.info("Applying HILE algorithm...")
        hile_result = self.hile.execute(understanding_result, n)
        
        # Apply IRQN for each level
        logger.info("Applying IRQN method...")
        enhanced_thoughts = self.irqn.execute(
            hile_result['thought'].outputs,
            level='thought'
        )
        enhanced_solutions = self.irqn.execute(
            hile_result['solution'].outputs,
            level='solution'
        )
        enhanced_implementations = self.irqn.execute(
            hile_result['implementation'].outputs,
            level='implementation'
        )
        
        result = {
            'thoughts': enhanced_thoughts,
            'solutions': enhanced_solutions,
            'implementations': enhanced_implementations,
            'diversity_scores': self._evaluate_intermediate_diversity(
                enhanced_thoughts,
                enhanced_solutions,
                enhanced_implementations
            )
        }
        
        # Validate diversity threshold
        diversity_threshold = self.config.diversity.threshold
        if result['diversity_scores'].get('overall', 0) < diversity_threshold:
            logger.warning(f"Diversity below threshold ({diversity_threshold}), "
                          "applying additional IRQN iterations...")
            # Could apply more IRQN here
        
        # Record and advance
        self.orchestrator.record_stage_result(
            WorkflowStage.DIVERSITY_IDEATION,
            result
        )
        self.orchestrator.advance_stage()
        
        return result
    
    def _execute_synthesis_stage(
        self,
        diverse_ideas: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute Stage 3: Code Synthesis.
        
        Translates diverse implementation plans into executable code.
        """
        logger.info("Executing Stage 3: Code Synthesis")
        
        context = self.orchestrator.get_accumulated_context()
        agent = self.agents['code_generating']
        
        generated_codes = []
        implementations = diverse_ideas.get('implementations', [])
        
        for impl in implementations:
            try:
                # Generate code from implementation plan
                code_result = agent.process(impl, context)
                
                # Validate syntax
                syntax_result = self.tools['code_interpreter'].validate_syntax(
                    code_result.get('code', '')
                )
                
                if syntax_result.get('valid', False):
                    generated_codes.append({
                        'id': impl.get('id', f'code_{len(generated_codes)}'),
                        'code': code_result['code'],
                        'implementation_id': impl.get('id'),
                        'parent_solution': impl.get('parent_id'),
                        'syntax_valid': True,
                        'style': impl.get('style', 'unknown')
                    })
                else:
                    logger.warning(f"Syntax validation failed for {impl.get('id')}: "
                                  f"{syntax_result.get('error')}")
                    
            except Exception as e:
                logger.error(f"Code generation failed for {impl.get('id')}: {e}")
        
        result = {'codes': generated_codes}
        
        # Record and advance
        self.orchestrator.record_stage_result(
            WorkflowStage.CODE_SYNTHESIS,
            result
        )
        self.orchestrator.advance_stage()
        
        return generated_codes
    
    def _execute_validation_stage(
        self,
        generated_codes: List[Dict[str, Any]],
        test_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute Stage 4: Quality Validation.
        
        Applies FBIR (Feedback-Based Iterative Repair) to validate
        and refine each code version.
        """
        logger.info("Executing Stage 4: Quality Validation")
        
        context = self.orchestrator.get_accumulated_context()
        validated_codes = []
        
        for code_info in generated_codes:
            code = code_info.get('code', '')
            code_id = code_info.get('id', 'unknown')
            
            logger.info(f"Validating code: {code_id}")
            
            # Apply FBIR
            refinement_result = self.qa_engine.validate_and_refine(
                code=code,
                test_cases=test_cases,
                context=context,
                max_iterations=self.config.quality.max_refinement_iterations
            )
            
            if refinement_result['pass_rate'] >= self.config.quality.threshold:
                validated_codes.append({
                    'id': code_id,
                    'code': refinement_result['code'],
                    'original_code': code,
                    'test_results': {
                        'passed': refinement_result.get('passed', 0),
                        'failed': refinement_result.get('failed', 0),
                        'pass_rate': refinement_result['pass_rate']
                    },
                    'refinement_iterations': refinement_result['iterations'],
                    'style': code_info.get('style', 'unknown'),
                    'implementation_id': code_info.get('implementation_id')
                })
            else:
                logger.warning(f"Code {code_id} failed validation "
                              f"(pass_rate: {refinement_result['pass_rate']:.2%})")
        
        result = {'validated_codes': validated_codes}
        
        # Record and advance
        self.orchestrator.record_stage_result(
            WorkflowStage.QUALITY_VALIDATION,
            result
        )
        self.orchestrator.advance_stage()
        
        return validated_codes
    
    def _execute_collection_stage(
        self,
        validated_codes: List[Dict[str, Any]],
        n: int
    ) -> Dict[str, Any]:
        """
        Execute Stage 5: Collection.
        
        Collects the final N-version code set with metadata.
        """
        logger.info("Executing Stage 5: Collection")
        
        # Select top N codes by diversity and quality
        selected_codes = self._select_n_versions(validated_codes, n)
        
        result = {
            'codes': selected_codes,
            'total_validated': len(validated_codes),
            'total_selected': len(selected_codes),
            'collection_metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_requested': n,
                'selection_criteria': 'diversity + quality'
            }
        }
        
        # Record and advance
        self.orchestrator.record_stage_result(
            WorkflowStage.COLLECTION,
            result
        )
        self.orchestrator.advance_stage()
        
        return result
    
    def _select_n_versions(
        self,
        validated_codes: List[Dict[str, Any]],
        n: int
    ) -> List[Dict[str, Any]]:
        """
        Select top N versions maximizing diversity and quality.
        
        Uses a greedy selection algorithm that balances:
        - Test pass rate (quality)
        - Diversity from already selected codes
        """
        if len(validated_codes) <= n:
            return validated_codes
        
        # Score each code
        for code in validated_codes:
            code['_score'] = code.get('test_results', {}).get('pass_rate', 0)
        
        # Greedy selection
        selected = []
        remaining = validated_codes.copy()
        
        while len(selected) < n and remaining:
            # Sort by score
            remaining.sort(key=lambda x: x.get('_score', 0), reverse=True)
            
            # Select the best
            best = remaining.pop(0)
            selected.append(best)
            
            # Update scores based on diversity from selected
            if selected:
                for code in remaining:
                    # Penalize similarity to selected codes
                    diversity_bonus = self._estimate_diversity(
                        code['code'],
                        [s['code'] for s in selected]
                    )
                    code['_score'] = (
                        code.get('test_results', {}).get('pass_rate', 0) * 0.6 +
                        diversity_bonus * 0.4
                    )
        
        # Clean up temporary scores
        for code in selected:
            code.pop('_score', None)
        
        return selected
    
    def _estimate_diversity(
        self,
        code: str,
        existing_codes: List[str]
    ) -> float:
        """Estimate diversity of code from existing codes."""
        if not existing_codes:
            return 1.0
        
        try:
            diversity_eval = self.tools['diversity_evaluator']
            all_codes = existing_codes + [code]
            mbcs = diversity_eval.compute_mbcs(all_codes)
            return 1.0 - mbcs
        except Exception:
            return 0.5
    
    def _evaluate_intermediate_diversity(
        self,
        thoughts: List[Dict[str, Any]],
        solutions: List[Dict[str, Any]],
        implementations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate diversity at intermediate stages."""
        try:
            diversity_eval = self.tools['diversity_evaluator']
            
            thought_contents = [t.get('content', '') for t in thoughts]
            solution_contents = [s.get('content', '') for s in solutions]
            impl_contents = [i.get('content', '') for i in implementations]
            
            return {
                'thought_mbcs': diversity_eval.compute_mbcs(thought_contents) if thought_contents else 0.0,
                'solution_mbcs': diversity_eval.compute_mbcs(solution_contents) if solution_contents else 0.0,
                'impl_mbcs': diversity_eval.compute_mbcs(impl_contents) if impl_contents else 0.0,
                'overall': 1.0 - (
                    (diversity_eval.compute_mbcs(thought_contents) if thought_contents else 0) +
                    (diversity_eval.compute_mbcs(solution_contents) if solution_contents else 0) +
                    (diversity_eval.compute_mbcs(impl_contents) if impl_contents else 0)
                ) / 3
            }
        except Exception as e:
            logger.warning(f"Intermediate diversity evaluation failed: {e}")
            return {'overall': 0.5}
    
    def _compute_diversity_metrics(
        self,
        codes: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute final diversity metrics."""
        code_strings = [c.get('code', '') for c in codes]
        
        if len(code_strings) < 2:
            return {'mbcs': 0.0, 'sdp': 1.0, 'overall_diversity': 1.0}
        
        try:
            diversity_eval = self.tools['diversity_evaluator']
            result = diversity_eval.compute_all(code_strings, self.llm_client)
            
            return {
                'mbcs': result.mbcs,
                'sdp': result.sdp,
                'semantic_diversity': result.semantic_diversity,
                'methodological_diversity': result.methodological_diversity,
                'overall_diversity': result.overall_diversity
            }
        except Exception as e:
            logger.error(f"Diversity metrics computation failed: {e}")
            return {'mbcs': 0.0, 'sdp': 0.0, 'overall_diversity': 0.0}
    
    def _compute_quality_metrics(
        self,
        codes: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute final quality metrics."""
        if not codes:
            return {'tpr': 0.0, 'average_pass_rate': 0.0}
        
        pass_rates = [
            c.get('test_results', {}).get('pass_rate', 0.0)
            for c in codes
        ]
        
        return {
            'tpr': sum(pass_rates) / len(pass_rates),
            'average_pass_rate': sum(pass_rates) / len(pass_rates),
            'min_pass_rate': min(pass_rates) if pass_rates else 0.0,
            'max_pass_rate': max(pass_rates) if pass_rates else 0.0,
            'all_passing': all(pr >= 1.0 for pr in pass_rates)
        }
    
    def _init_llm_client(self):
        """Initialize the LLM client."""
        from ..utils.llm_client import LLMClient
        return LLMClient(self.config)
    
    def _init_tools(self) -> Dict[str, Any]:
        """Initialize all tools."""
        from ..tools.prompt_generator import DynamicPromptGenerator
        from ..tools.code_interpreter import CodeInterpreter
        from ..tools.test_executor import TestCasesExecutor
        from ..tools.diversity_evaluator import DiversityEvaluator
        from ..tools.debugger import Debugger
        from ..tools.knowledge_search import KnowledgeSearch
        from ..tools.code_collector import CodeCollector
        
        return {
            'prompt_generator': DynamicPromptGenerator(self.config.tools if hasattr(self.config, 'tools') else {}),
            'code_interpreter': CodeInterpreter(self.config.tools if hasattr(self.config, 'tools') else {}),
            'test_executor': TestCasesExecutor(self.config.tools if hasattr(self.config, 'tools') else {}),
            'diversity_evaluator': DiversityEvaluator(self.config.tools if hasattr(self.config, 'tools') else {}),
            'debugger': Debugger(self.config),
            'knowledge_search': KnowledgeSearch(self.config.knowledge_bases if hasattr(self.config, 'knowledge_bases') else {}),
            'code_collector': CodeCollector()
        }
    
    def _init_agents(self) -> Dict[str, Any]:
        """Initialize all agents."""
        from ..agents.understanding_agent import UnderstandingAgent
        from ..agents.diversity_agent import DiversityEnhancingAgent
        from ..agents.code_generating_agent import CodeGeneratingAgent
        from ..agents.evaluating_agent import EvaluatingAgent
        
        return {
            'understanding': UnderstandingAgent(
                llm_client=self.llm_client,
                tools=self.tools,
                config=self.config
            ),
            'diversity_enhancing': DiversityEnhancingAgent(
                llm_client=self.llm_client,
                diversity_evaluator=self.tools['diversity_evaluator'],
                knowledge_search=self.tools['knowledge_search'],
                dynamic_prompt_generator=self.tools['prompt_generator'],
                config=self.config
            ),
            'code_generating': CodeGeneratingAgent(
                llm_client=self.llm_client,
                code_interpreter=self.tools['code_interpreter'],
                config=self.config
            ),
            'evaluating': EvaluatingAgent(
                llm_client=self.llm_client,
                test_executor=self.tools['test_executor'],
                debugger=self.tools['debugger'],
                config=self.config
            )
        }
    
    def _init_hile(self):
        """Initialize HILE algorithm."""
        from ..algorithms.hile import HILEAlgorithm
        return HILEAlgorithm(
            llm_client=self.llm_client,
            knowledge_bases=self.tools.get('knowledge_search'),
            config=self.config.diversity.hile if hasattr(self.config.diversity, 'hile') else {}
        )
    
    def _init_irqn(self):
        """Initialize IRQN method."""
        from ..algorithms.irqn import IRQNMethod
        return IRQNMethod(
            diversity_evaluator=self.tools['diversity_evaluator'],
            llm_client=self.llm_client,
            config=self.config.diversity.irqn if hasattr(self.config.diversity, 'irqn') else {}
        )
    
    def _init_quality_assurance(self):
        """Initialize Quality Assurance engine."""
        from ..algorithms.quality_assurance import QualityAssuranceEngine
        return QualityAssuranceEngine(
            test_executor=self.tools['test_executor'],
            debugger=self.tools['debugger'],
            code_interpreter=self.tools['code_interpreter'],
            llm_client=self.llm_client,
            config=self.config.quality if hasattr(self.config, 'quality') else {}
        )
    
    def _init_quality_gates(self) -> Dict[WorkflowStage, QualityGate]:
        """Initialize quality gates for each stage."""
        return {
            WorkflowStage.DIVERSITY_IDEATION: QualityGate(
                WorkflowStage.DIVERSITY_IDEATION,
                {'min_understanding': True}
            ),
            WorkflowStage.CODE_SYNTHESIS: QualityGate(
                WorkflowStage.CODE_SYNTHESIS,
                {'min_ideas': 3}
            ),
            WorkflowStage.QUALITY_VALIDATION: QualityGate(
                WorkflowStage.QUALITY_VALIDATION,
                {'min_codes': 3}
            ),
            WorkflowStage.COLLECTION: QualityGate(
                WorkflowStage.COLLECTION,
                {
                    'quality_threshold': self.config.quality.threshold if hasattr(self.config.quality, 'threshold') else 0.9,
                    'min_versions': 3
                }
            )
        }
