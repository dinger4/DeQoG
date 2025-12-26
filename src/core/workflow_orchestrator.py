"""
DeQoG Deterministic Workflow Orchestrator

Replaces FSM-based state control with deterministic workflow orchestration
through dynamic prompt generation and output format templates.

Based on the latest paper: "Automated Fault-Tolerant Code Generation via LLMs:
A Diversity-Enhanced and Quality-Assured Approach"
"""

from enum import Enum, auto
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

from ..utils.logger import get_logger

logger = get_logger("workflow_orchestrator")


class WorkflowStage(Enum):
    """
    Workflow stages for deterministic orchestration.
    
    Unlike FSM states, these stages follow a fixed deterministic flow
    controlled by prompt templates and output format specifications.
    """
    UNDERSTANDING = auto()           # Stage 1: Problem understanding
    DIVERSITY_IDEATION = auto()      # Stage 2: Diverse idea generation (HILE + IRQN)
    CODE_SYNTHESIS = auto()          # Stage 3: Code implementation
    QUALITY_VALIDATION = auto()      # Stage 4: Testing and refinement (FBIR)
    COLLECTION = auto()              # Stage 5: N-version collection
    COMPLETE = auto()


@dataclass
class OutputFormat:
    """
    Output format template for deterministic LLM output control.
    
    Ensures LLM outputs follow a structured, parseable format
    to convert unpredictable outputs into deterministic templated responses.
    """
    stage: WorkflowStage
    template: Dict[str, Any]
    required_fields: List[str]
    validation_rules: Dict[str, Callable] = field(default_factory=dict)
    
    def validate(self, output: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate output against template requirements."""
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in output:
                errors.append(f"Missing required field: {field}")
        
        # Apply validation rules
        for field, rule in self.validation_rules.items():
            if field in output and not rule(output[field]):
                errors.append(f"Validation failed for field: {field}")
        
        return len(errors) == 0, errors


@dataclass
class StageResult:
    """Result from executing a workflow stage."""
    stage: WorkflowStage
    success: bool
    output: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DeterministicWorkflowOrchestrator:
    """
    Deterministic Workflow Orchestrator.
    
    Core orchestration component that replaces FSM-based state control.
    Uses dynamic prompt generation and output format templates to ensure
    deterministic and controllable LLM outputs.
    
    Key Features:
    1. Fixed workflow stages (no dynamic state transitions)
    2. Output format templates for structured responses
    3. Dynamic prompt generator integration
    4. Quality gates between stages
    """
    
    def __init__(
        self,
        prompt_generator,
        config,
        output_formats: Optional[Dict[WorkflowStage, OutputFormat]] = None
    ):
        """
        Initialize the workflow orchestrator.
        
        Args:
            prompt_generator: Dynamic prompt generator for stage-specific prompts
            config: System configuration
            output_formats: Custom output format templates per stage
        """
        self.prompt_generator = prompt_generator
        self.config = config
        self.output_formats = output_formats or self._default_output_formats()
        
        # Workflow state
        self.current_stage = WorkflowStage.UNDERSTANDING
        self.stage_results: Dict[WorkflowStage, StageResult] = {}
        self.context: Dict[str, Any] = {}
        
        logger.info("DeterministicWorkflowOrchestrator initialized")
    
    def _default_output_formats(self) -> Dict[WorkflowStage, OutputFormat]:
        """Define default output format templates for each stage."""
        return {
            WorkflowStage.UNDERSTANDING: OutputFormat(
                stage=WorkflowStage.UNDERSTANDING,
                template={
                    "problem_summary": "string",
                    "input_format": "string",
                    "output_format": "string",
                    "constraints": ["list of strings"],
                    "edge_cases": ["list of strings"],
                    "knowledge_retrieved": ["list of relevant knowledge"]
                },
                required_fields=["problem_summary", "input_format", "output_format"],
                validation_rules={
                    "problem_summary": lambda x: len(x) > 20
                }
            ),
            
            WorkflowStage.DIVERSITY_IDEATION: OutputFormat(
                stage=WorkflowStage.DIVERSITY_IDEATION,
                template={
                    "thoughts": [{
                        "id": "string",
                        "paradigm": "string",
                        "description": "string",
                        "complexity": {"time": "string", "space": "string"}
                    }],
                    "solutions": [{
                        "id": "string",
                        "parent_thought": "string",
                        "pseudocode": "string",
                        "data_structures": ["list"]
                    }],
                    "implementations": [{
                        "id": "string",
                        "parent_solution": "string",
                        "style": "string",
                        "plan": "string"
                    }]
                },
                required_fields=["thoughts", "solutions", "implementations"],
                validation_rules={
                    "thoughts": lambda x: len(x) >= 3,
                    "solutions": lambda x: len(x) >= 3,
                    "implementations": lambda x: len(x) >= 3
                }
            ),
            
            WorkflowStage.CODE_SYNTHESIS: OutputFormat(
                stage=WorkflowStage.CODE_SYNTHESIS,
                template={
                    "codes": [{
                        "id": "string",
                        "implementation_id": "string",
                        "code": "string",
                        "language": "python",
                        "syntax_valid": "boolean"
                    }]
                },
                required_fields=["codes"],
                validation_rules={
                    "codes": lambda x: all("code" in c for c in x)
                }
            ),
            
            WorkflowStage.QUALITY_VALIDATION: OutputFormat(
                stage=WorkflowStage.QUALITY_VALIDATION,
                template={
                    "validated_codes": [{
                        "id": "string",
                        "code": "string",
                        "test_results": {
                            "passed": "int",
                            "failed": "int",
                            "pass_rate": "float"
                        },
                        "refinement_iterations": "int"
                    }]
                },
                required_fields=["validated_codes"]
            ),
            
            WorkflowStage.COLLECTION: OutputFormat(
                stage=WorkflowStage.COLLECTION,
                template={
                    "n_version_codes": ["list of validated codes"],
                    "diversity_metrics": {},
                    "quality_metrics": {},
                    "metadata": {}
                },
                required_fields=["n_version_codes"]
            )
        }
    
    def get_stage_prompt(
        self,
        stage: WorkflowStage,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate stage-specific prompt with output format specification.
        
        Uses dynamic prompt generator to create prompts that enforce
        deterministic output formats.
        """
        output_format = self.output_formats.get(stage)
        
        # Build prompt with format specification
        prompt = self.prompt_generator.generate_stage_prompt(
            stage=stage,
            context=context,
            output_format=output_format.template if output_format else None
        )
        
        return prompt
    
    def validate_stage_output(
        self,
        stage: WorkflowStage,
        output: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Validate stage output against format template.
        
        Returns:
            Tuple of (is_valid, list of errors)
        """
        output_format = self.output_formats.get(stage)
        if not output_format:
            return True, []
        
        return output_format.validate(output)
    
    def record_stage_result(
        self,
        stage: WorkflowStage,
        output: Dict[str, Any],
        success: bool = True,
        errors: List[str] = None
    ):
        """Record the result of a completed stage."""
        result = StageResult(
            stage=stage,
            success=success,
            output=output,
            errors=errors or [],
            metadata={
                "validation_passed": success,
                "output_format_used": stage.name
            }
        )
        
        self.stage_results[stage] = result
        self.context[stage.name] = output
        
        logger.info(f"Stage {stage.name} completed: success={success}")
    
    def get_next_stage(self, current: WorkflowStage) -> Optional[WorkflowStage]:
        """Get the next stage in the deterministic workflow."""
        stage_order = [
            WorkflowStage.UNDERSTANDING,
            WorkflowStage.DIVERSITY_IDEATION,
            WorkflowStage.CODE_SYNTHESIS,
            WorkflowStage.QUALITY_VALIDATION,
            WorkflowStage.COLLECTION,
            WorkflowStage.COMPLETE
        ]
        
        try:
            idx = stage_order.index(current)
            if idx < len(stage_order) - 1:
                return stage_order[idx + 1]
        except ValueError:
            pass
        
        return None
    
    def advance_stage(self) -> WorkflowStage:
        """Advance to the next workflow stage."""
        next_stage = self.get_next_stage(self.current_stage)
        if next_stage:
            logger.info(f"Advancing: {self.current_stage.name} -> {next_stage.name}")
            self.current_stage = next_stage
        return self.current_stage
    
    def get_accumulated_context(self) -> Dict[str, Any]:
        """Get all accumulated context from previous stages."""
        return {
            "current_stage": self.current_stage.name,
            "completed_stages": {
                stage.name: result.output
                for stage, result in self.stage_results.items()
            },
            "context": self.context
        }
    
    def is_complete(self) -> bool:
        """Check if workflow has completed."""
        return self.current_stage == WorkflowStage.COMPLETE
    
    def reset(self):
        """Reset the orchestrator to initial state."""
        self.current_stage = WorkflowStage.UNDERSTANDING
        self.stage_results.clear()
        self.context.clear()
        logger.info("Workflow orchestrator reset")
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire workflow execution."""
        return {
            "current_stage": self.current_stage.name,
            "completed_stages": [
                {
                    "stage": stage.name,
                    "success": result.success,
                    "timestamp": result.timestamp,
                    "errors": result.errors
                }
                for stage, result in self.stage_results.items()
            ],
            "is_complete": self.is_complete()
        }


class QualityGate:
    """
    Quality gate for stage transitions.
    
    Enforces quality requirements between workflow stages
    without dynamic FSM-style decision making.
    """
    
    def __init__(self, stage: WorkflowStage, requirements: Dict[str, Any]):
        """
        Initialize quality gate.
        
        Args:
            stage: The stage this gate guards entry to
            requirements: Quality requirements to pass the gate
        """
        self.stage = stage
        self.requirements = requirements
    
    def check(self, context: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Check if quality requirements are met.
        
        Returns:
            Tuple of (passed, list of unmet requirements)
        """
        unmet = []
        
        # Stage-specific checks
        if self.stage == WorkflowStage.DIVERSITY_IDEATION:
            # Must have understanding result
            if "UNDERSTANDING" not in context:
                unmet.append("Missing understanding stage output")
        
        elif self.stage == WorkflowStage.CODE_SYNTHESIS:
            # Must have diverse ideas
            div_output = context.get("DIVERSITY_IDEATION", {})
            min_ideas = self.requirements.get("min_ideas", 3)
            if len(div_output.get("implementations", [])) < min_ideas:
                unmet.append(f"Insufficient diverse ideas (need {min_ideas})")
        
        elif self.stage == WorkflowStage.QUALITY_VALIDATION:
            # Must have synthesized codes
            code_output = context.get("CODE_SYNTHESIS", {})
            if not code_output.get("codes"):
                unmet.append("No codes to validate")
        
        elif self.stage == WorkflowStage.COLLECTION:
            # Must have validated codes meeting quality threshold
            qa_output = context.get("QUALITY_VALIDATION", {})
            validated = qa_output.get("validated_codes", [])
            threshold = self.requirements.get("quality_threshold", 0.9)
            passing = [c for c in validated if c.get("test_results", {}).get("pass_rate", 0) >= threshold]
            if len(passing) < self.requirements.get("min_versions", 3):
                unmet.append(f"Insufficient validated codes meeting quality threshold ({threshold})")
        
        return len(unmet) == 0, unmet


# Backward compatibility alias
StateController = DeterministicWorkflowOrchestrator

