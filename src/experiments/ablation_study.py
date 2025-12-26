"""
DeQoG Ablation Study

Implements ablation study experiments to evaluate component contributions.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger("ablation_study")


@dataclass
class AblationVariant:
    """Definition of an ablation variant."""
    name: str
    description: str
    disabled_components: List[str]


class AblationStudy:
    """
    Ablation Study Framework.
    
    Evaluates the contribution of each DeQoG component by
    systematically disabling components and measuring impact.
    
    Variants:
    - DeQoG-Full: Complete system
    - DeQoG-NoDiv: Without diversity enhancement (HILE/IRQN)
    - DeQoG-NoQA: Without quality assurance (iterative refinement)
    - DeQoG-NoFSM: Without FSM state control
    """
    
    def __init__(self, pipeline=None):
        """
        Initialize ablation study.
        
        Args:
            pipeline: DeQoG pipeline to study
        """
        self.pipeline = pipeline
        
        # Define ablation variants
        self.variants = {
            'DeQoG-Full': AblationVariant(
                name='DeQoG-Full',
                description='Complete DeQoG system',
                disabled_components=[]
            ),
            'DeQoG-NoDiv': AblationVariant(
                name='DeQoG-NoDiv',
                description='Without diversity enhancement',
                disabled_components=['hile', 'irqn', 'diversity_evaluator']
            ),
            'DeQoG-NoQA': AblationVariant(
                name='DeQoG-NoQA',
                description='Without quality assurance',
                disabled_components=['quality_assurance', 'debugger', 'iterative_refinement']
            ),
            'DeQoG-NoFSM': AblationVariant(
                name='DeQoG-NoFSM',
                description='Without FSM state control',
                disabled_components=['fsm_controller', 'rollback', 'retry']
            ),
            'DeQoG-NoHILE': AblationVariant(
                name='DeQoG-NoHILE',
                description='Without HILE algorithm',
                disabled_components=['hile']
            ),
            'DeQoG-NoIRQN': AblationVariant(
                name='DeQoG-NoIRQN',
                description='Without IRQN method',
                disabled_components=['irqn']
            ),
        }
    
    def run_study(
        self,
        task_description: str,
        test_cases: List[Dict[str, Any]],
        variants: Optional[List[str]] = None,
        n_versions: int = 5
    ) -> Dict[str, Any]:
        """
        Run ablation study.
        
        Args:
            task_description: Programming task
            test_cases: Test cases for evaluation
            variants: List of variant names to run
            n_versions: Number of versions to generate
            
        Returns:
            Study results dictionary
        """
        logger.info("Starting ablation study")
        
        if variants is None:
            variants = ['DeQoG-Full', 'DeQoG-NoDiv', 'DeQoG-NoQA']
        
        results = {}
        
        for variant_name in variants:
            logger.info(f"Running variant: {variant_name}")
            
            variant = self.variants.get(variant_name)
            if not variant:
                logger.warning(f"Unknown variant: {variant_name}")
                continue
            
            # Run with this variant configuration
            variant_result = self._run_variant(
                variant,
                task_description,
                test_cases,
                n_versions
            )
            
            results[variant_name] = variant_result
        
        # Compute comparative analysis
        comparison = self._compare_variants(results)
        
        return {
            'variant_results': results,
            'comparison': comparison
        }
    
    def _run_variant(
        self,
        variant: AblationVariant,
        task_description: str,
        test_cases: List[Dict[str, Any]],
        n_versions: int
    ) -> Dict[str, Any]:
        """
        Run a single ablation variant.
        
        Args:
            variant: Ablation variant to run
            task_description: Task description
            test_cases: Test cases
            n_versions: Number of versions
            
        Returns:
            Variant result dictionary
        """
        if self.pipeline is None:
            logger.warning("No pipeline configured, returning mock results")
            return self._mock_variant_result(variant)
        
        # Configure pipeline for this variant
        original_config = self._save_config()
        self._apply_variant_config(variant)
        
        try:
            # Run pipeline
            result = self.pipeline.generate_n_versions(
                task_description=task_description,
                test_cases=test_cases,
                n=n_versions
            )
            
            return {
                'variant': variant.name,
                'disabled_components': variant.disabled_components,
                'n_version_codes': result.get('n_version_codes', []),
                'diversity_metrics': result.get('diversity_metrics', {}),
                'quality_metrics': result.get('quality_metrics', {}),
                'generation_time': result.get('generation_metadata', {}).get('time', 0)
            }
            
        finally:
            # Restore original config
            self._restore_config(original_config)
    
    def _apply_variant_config(self, variant: AblationVariant):
        """Apply variant configuration to pipeline."""
        if self.pipeline is None:
            return
        
        for component in variant.disabled_components:
            # Disable the component
            if hasattr(self.pipeline, f'disable_{component}'):
                getattr(self.pipeline, f'disable_{component}')()
            elif hasattr(self.pipeline, 'config'):
                # Try to disable via config
                pass
    
    def _save_config(self) -> Dict[str, Any]:
        """Save current pipeline configuration."""
        if self.pipeline is None:
            return {}
        return {}  # Placeholder
    
    def _restore_config(self, config: Dict[str, Any]):
        """Restore pipeline configuration."""
        pass  # Placeholder
    
    def _mock_variant_result(self, variant: AblationVariant) -> Dict[str, Any]:
        """Generate mock result for testing."""
        import random
        
        # Simulate different performance based on disabled components
        base_diversity = 0.7
        base_quality = 0.85
        
        if 'hile' in variant.disabled_components or 'irqn' in variant.disabled_components:
            base_diversity *= 0.6
        
        if 'quality_assurance' in variant.disabled_components:
            base_quality *= 0.7
        
        return {
            'variant': variant.name,
            'disabled_components': variant.disabled_components,
            'n_version_codes': [],
            'diversity_metrics': {
                'mbcs': 1.0 - base_diversity + random.uniform(-0.05, 0.05),
                'sdp': base_diversity + random.uniform(-0.05, 0.05)
            },
            'quality_metrics': {
                'tpr': base_quality + random.uniform(-0.05, 0.05),
                'validation_rate': base_quality + random.uniform(-0.05, 0.05)
            },
            'generation_time': random.uniform(10, 60)
        }
    
    def _compare_variants(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare results across variants.
        
        Args:
            results: Results for each variant
            
        Returns:
            Comparison analysis
        """
        if not results:
            return {}
        
        comparison = {
            'diversity_comparison': {},
            'quality_comparison': {},
            'component_impact': {}
        }
        
        # Get baseline (Full version)
        baseline = results.get('DeQoG-Full', {})
        baseline_div = baseline.get('diversity_metrics', {})
        baseline_qual = baseline.get('quality_metrics', {})
        
        for variant_name, variant_result in results.items():
            if variant_name == 'DeQoG-Full':
                continue
            
            div_metrics = variant_result.get('diversity_metrics', {})
            qual_metrics = variant_result.get('quality_metrics', {})
            
            # Compute deltas
            div_delta = {}
            for key in baseline_div:
                if key in div_metrics:
                    div_delta[key] = div_metrics[key] - baseline_div.get(key, 0)
            
            qual_delta = {}
            for key in baseline_qual:
                if key in qual_metrics:
                    qual_delta[key] = qual_metrics[key] - baseline_qual.get(key, 0)
            
            comparison['diversity_comparison'][variant_name] = div_delta
            comparison['quality_comparison'][variant_name] = qual_delta
            
            # Analyze component impact
            disabled = variant_result.get('disabled_components', [])
            comparison['component_impact'][variant_name] = {
                'disabled': disabled,
                'diversity_impact': sum(div_delta.values()) if div_delta else 0,
                'quality_impact': sum(qual_delta.values()) if qual_delta else 0
            }
        
        return comparison
    
    def generate_report(
        self,
        study_results: Dict[str, Any]
    ) -> str:
        """
        Generate a markdown report of the ablation study.
        
        Args:
            study_results: Results from run_study
            
        Returns:
            Markdown report string
        """
        lines = [
            "# DeQoG Ablation Study Report",
            "",
            "## Summary",
            ""
        ]
        
        variant_results = study_results.get('variant_results', {})
        
        # Results table
        lines.extend([
            "| Variant | SDP | MBCS | TPR | Components Disabled |",
            "|---------|-----|------|-----|---------------------|"
        ])
        
        for variant_name, result in variant_results.items():
            div = result.get('diversity_metrics', {})
            qual = result.get('quality_metrics', {})
            disabled = result.get('disabled_components', [])
            
            lines.append(
                f"| {variant_name} | "
                f"{div.get('sdp', 0):.3f} | "
                f"{div.get('mbcs', 0):.3f} | "
                f"{qual.get('tpr', 0):.3f} | "
                f"{', '.join(disabled) or 'None'} |"
            )
        
        # Analysis
        comparison = study_results.get('comparison', {})
        component_impact = comparison.get('component_impact', {})
        
        lines.extend([
            "",
            "## Component Impact Analysis",
            ""
        ])
        
        for variant_name, impact in component_impact.items():
            lines.extend([
                f"### {variant_name}",
                f"- Disabled: {', '.join(impact.get('disabled', []))}",
                f"- Diversity Impact: {impact.get('diversity_impact', 0):.3f}",
                f"- Quality Impact: {impact.get('quality_impact', 0):.3f}",
                ""
            ])
        
        return "\n".join(lines)

