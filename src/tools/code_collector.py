"""
DeQoG Code Collector

Collects and organizes all validated N-version codes.
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

from .base_tool import BaseTool
from ..utils.logger import get_logger

logger = get_logger("code_collector")


@dataclass
class CodeVersion:
    """Represents a single code version."""
    version_id: str
    code: str
    algorithm: str
    approach_description: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class NVersionCodeSet:
    """Represents a complete N-version code set."""
    task_id: str
    task_description: str
    n_versions: int
    versions: List[CodeVersion] = field(default_factory=list)
    diversity_metrics: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class CodeCollector(BaseTool):
    """
    Code Collector Tool.
    
    Collects, organizes, and exports N-version code sets
    with associated metadata and metrics.
    """
    
    def __init__(self):
        """Initialize the code collector."""
        super().__init__(
            name="code_collector",
            description="Collects and organizes N-version code sets"
        )
        
        self._collected_sets: Dict[str, NVersionCodeSet] = {}
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect validated codes into an N-version set.
        
        Args:
            params: Dictionary containing:
                - validated_codes: List of validated code dictionaries
                - task_description: Description of the task
                - diversity_metrics: Diversity evaluation results
                - quality_metrics: Quality evaluation results
                - metadata: Additional generation metadata
                
        Returns:
            Dictionary with the collected N-version code set
        """
        validated_codes = params.get('validated_codes', [])
        task_description = params.get('task_description', '')
        diversity_metrics = params.get('diversity_metrics', {})
        quality_metrics = params.get('quality_metrics', {})
        metadata = params.get('metadata', {})
        
        # Generate task ID
        task_id = self._generate_task_id(task_description)
        
        # Create code versions
        versions = []
        for i, code_info in enumerate(validated_codes):
            version = CodeVersion(
                version_id=f"{task_id}_v{i+1}",
                code=code_info.get('code', ''),
                algorithm=code_info.get('meta', {}).get('algorithm', f'approach_{i+1}'),
                approach_description=code_info.get('meta', {}).get('description', ''),
                metrics=code_info.get('metrics', {}),
                metadata=code_info.get('meta', {})
            )
            versions.append(version)
        
        # Create N-version code set
        code_set = NVersionCodeSet(
            task_id=task_id,
            task_description=task_description,
            n_versions=len(versions),
            versions=versions,
            diversity_metrics=diversity_metrics,
            quality_metrics=quality_metrics,
            generation_metadata=metadata
        )
        
        # Store the set
        self._collected_sets[task_id] = code_set
        
        logger.info(f"Collected {len(versions)} code versions for task {task_id}")
        
        return self._format_result(code_set)
    
    def _generate_task_id(self, task_description: str) -> str:
        """Generate a unique task ID."""
        import hashlib
        
        # Create hash from description and timestamp
        content = f"{task_description}_{datetime.now().isoformat()}"
        hash_obj = hashlib.md5(content.encode())
        return f"task_{hash_obj.hexdigest()[:8]}"
    
    def _format_result(self, code_set: NVersionCodeSet) -> Dict[str, Any]:
        """Format the code set for output."""
        return {
            'task_id': code_set.task_id,
            'n_version_codes': [
                {
                    'version_id': v.version_id,
                    'code': v.code,
                    'algorithm': v.algorithm,
                    'metrics': v.metrics,
                    'meta': {
                        'approach_description': v.approach_description,
                        **v.metadata
                    }
                }
                for v in code_set.versions
            ],
            'diversity_metrics': code_set.diversity_metrics,
            'quality_metrics': code_set.quality_metrics,
            'generation_metadata': {
                **code_set.generation_metadata,
                'n_versions': code_set.n_versions,
                'created_at': code_set.created_at
            }
        }
    
    def get_code_set(self, task_id: str) -> Optional[NVersionCodeSet]:
        """
        Get a collected code set by task ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            NVersionCodeSet or None
        """
        return self._collected_sets.get(task_id)
    
    def get_all_code_sets(self) -> List[NVersionCodeSet]:
        """Get all collected code sets."""
        return list(self._collected_sets.values())
    
    def export_to_json(
        self,
        task_id: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export a code set to JSON.
        
        Args:
            task_id: Task identifier
            output_path: Optional output file path
            
        Returns:
            JSON string
        """
        code_set = self._collected_sets.get(task_id)
        if not code_set:
            raise ValueError(f"Code set not found: {task_id}")
        
        result = self._format_result(code_set)
        json_str = json.dumps(result, indent=2, default=str)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"Exported code set to {output_path}")
        
        return json_str
    
    def export_codes_only(
        self,
        task_id: str,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Export just the code files.
        
        Args:
            task_id: Task identifier
            output_dir: Optional output directory
            
        Returns:
            List of code strings
        """
        code_set = self._collected_sets.get(task_id)
        if not code_set:
            raise ValueError(f"Code set not found: {task_id}")
        
        codes = [v.code for v in code_set.versions]
        
        if output_dir:
            from pathlib import Path
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            for i, version in enumerate(code_set.versions):
                file_path = out_path / f"{version.version_id}.py"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f'"""\n{version.approach_description}\n"""\n\n')
                    f.write(version.code)
            
            logger.info(f"Exported {len(codes)} code files to {output_dir}")
        
        return codes
    
    def get_summary(self, task_id: str) -> Dict[str, Any]:
        """
        Get a summary of a code set.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Summary dictionary
        """
        code_set = self._collected_sets.get(task_id)
        if not code_set:
            raise ValueError(f"Code set not found: {task_id}")
        
        return {
            'task_id': code_set.task_id,
            'n_versions': code_set.n_versions,
            'algorithms': [v.algorithm for v in code_set.versions],
            'diversity_metrics': code_set.diversity_metrics,
            'quality_metrics': code_set.quality_metrics,
            'avg_pass_rate': sum(
                v.metrics.get('pass_rate', 0) for v in code_set.versions
            ) / code_set.n_versions if code_set.n_versions > 0 else 0
        }
    
    def clear(self):
        """Clear all collected code sets."""
        self._collected_sets.clear()
        logger.info("Cleared all collected code sets")
    
    def validate_diversity_requirements(
        self,
        task_id: str,
        min_sdp: float = 0.5,
        max_mbcs: float = 0.7
    ) -> Dict[str, Any]:
        """
        Validate that a code set meets diversity requirements.
        
        Args:
            task_id: Task identifier
            min_sdp: Minimum SDP score required
            max_mbcs: Maximum MBCS score allowed
            
        Returns:
            Validation results
        """
        code_set = self._collected_sets.get(task_id)
        if not code_set:
            raise ValueError(f"Code set not found: {task_id}")
        
        sdp = code_set.diversity_metrics.get('sdp', 0)
        mbcs = code_set.diversity_metrics.get('mbcs', 1)
        
        sdp_valid = sdp >= min_sdp
        mbcs_valid = mbcs <= max_mbcs
        
        return {
            'valid': sdp_valid and mbcs_valid,
            'sdp': {'value': sdp, 'valid': sdp_valid, 'threshold': min_sdp},
            'mbcs': {'value': mbcs, 'valid': mbcs_valid, 'threshold': max_mbcs}
        }

