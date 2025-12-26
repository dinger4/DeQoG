"""
DeQoG Fault Injection Experiment

Implements fault injection experiments for evaluating N-version fault tolerance.
"""

import random
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from ..metrics.fault_tolerance_metrics import FaultToleranceMetrics
from ..tools.code_interpreter import CodeInterpreter
from ..utils.logger import get_logger

logger = get_logger("fault_injection")


@dataclass
class FaultPattern:
    """Definition of a fault pattern."""
    name: str
    description: str
    num_faulty_versions: int  # -1 means calculated based on N
    fault_type: str  # 'code_level' or 'algorithm_level'


class FaultInjectionExperiment:
    """
    Fault Injection Experiment Framework.
    
    Implements fault injection at two levels:
    - Code Level (Pat-CL): Inject bugs into specific code versions
    - Algorithm Level (Pat-AL): Inject Common Mode Failures (CMFs)
    
    Patterns:
    - Pat-CL 0 / Pat-AL 0: No faults
    - Pat-CL 1 / Pat-AL 1: One fault/CMF
    - Pat-CL 2 / Pat-AL 2: floor((N-1)/2) faults/CMFs
    - Pat-CL 3 / Pat-AL 3: floor((N+1)/2) faults/CMFs  
    - Pat-CL 4 / Pat-AL 4: All versions faulty / all CMFs
    """
    
    def __init__(
        self,
        n_versions: int = 5,
        code_interpreter: Optional[CodeInterpreter] = None
    ):
        """
        Initialize the fault injection experiment.
        
        Args:
            n_versions: Number of versions in N-version system
            code_interpreter: Code interpreter for execution
        """
        self.n_versions = n_versions
        self.code_interpreter = code_interpreter or CodeInterpreter()
        self.fault_metrics = FaultToleranceMetrics()
        
        # Define fault patterns
        self.fault_patterns = self._define_fault_patterns()
        
        # Common mode failures library
        self.cmf_library = self._define_cmf_library()
    
    def _define_fault_patterns(self) -> Dict[str, FaultPattern]:
        """Define all fault patterns."""
        n = self.n_versions
        
        return {
            'Pat-CL 0': FaultPattern(
                name='Pat-CL 0',
                description='No code-level faults',
                num_faulty_versions=0,
                fault_type='code_level'
            ),
            'Pat-CL 1': FaultPattern(
                name='Pat-CL 1',
                description='Exactly one version with fault',
                num_faulty_versions=1,
                fault_type='code_level'
            ),
            'Pat-CL 2': FaultPattern(
                name='Pat-CL 2',
                description='floor((N-1)/2) versions with faults',
                num_faulty_versions=(n - 1) // 2,
                fault_type='code_level'
            ),
            'Pat-CL 3': FaultPattern(
                name='Pat-CL 3',
                description='floor((N+1)/2) versions with faults',
                num_faulty_versions=(n + 1) // 2,
                fault_type='code_level'
            ),
            'Pat-CL 4': FaultPattern(
                name='Pat-CL 4',
                description='All versions with faults',
                num_faulty_versions=n,
                fault_type='code_level'
            ),
            'Pat-AL 0': FaultPattern(
                name='Pat-AL 0',
                description='No algorithm-level CMFs',
                num_faulty_versions=0,
                fault_type='algorithm_level'
            ),
            'Pat-AL 1': FaultPattern(
                name='Pat-AL 1',
                description='All versions with 1 CMF',
                num_faulty_versions=1,
                fault_type='algorithm_level'
            ),
            'Pat-AL 2': FaultPattern(
                name='Pat-AL 2',
                description='All versions with floor((N-1)/2) CMFs',
                num_faulty_versions=(n - 1) // 2,
                fault_type='algorithm_level'
            ),
            'Pat-AL 3': FaultPattern(
                name='Pat-AL 3',
                description='All versions with floor((N+1)/2) CMFs',
                num_faulty_versions=(n + 1) // 2,
                fault_type='algorithm_level'
            ),
            'Pat-AL 4': FaultPattern(
                name='Pat-AL 4',
                description='All versions with all available CMFs',
                num_faulty_versions=-1,  # All available
                fault_type='algorithm_level'
            ),
        }
    
    def _define_cmf_library(self) -> List[Dict[str, Any]]:
        """Define Common Mode Failures library."""
        return [
            {
                'id': 'cmf_off_by_one',
                'description': 'Off-by-one error in loop bounds',
                'pattern': r'range\((\w+)\)',
                'replacement': r'range(\1 - 1)'
            },
            {
                'id': 'cmf_boundary',
                'description': 'Incorrect boundary condition',
                'pattern': r'<=',
                'replacement': '<'
            },
            {
                'id': 'cmf_init',
                'description': 'Wrong initialization',
                'pattern': r'= 0\b',
                'replacement': '= 1'
            },
            {
                'id': 'cmf_return',
                'description': 'Wrong return value',
                'pattern': r'return (\w+)',
                'replacement': r'return \1 + 1'
            },
            {
                'id': 'cmf_operator',
                'description': 'Wrong operator',
                'pattern': r'\+',
                'replacement': '-'
            },
        ]
    
    def run_experiment(
        self,
        n_version_codes: List[str],
        test_cases: List[Dict[str, Any]],
        patterns: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete fault injection experiment.
        
        Args:
            n_version_codes: List of N version code strings
            test_cases: List of test cases
            patterns: Optional dict of patterns to run
                     {'code_level': [...], 'algorithm_level': [...]}
                     
        Returns:
            Experiment results dictionary
        """
        logger.info(f"Starting fault injection experiment with {len(n_version_codes)} versions")
        
        results = {
            'code_level': {},
            'algorithm_level': {},
            'summary': {}
        }
        
        # Default patterns if not specified
        if patterns is None:
            patterns = {
                'code_level': ['Pat-CL 0', 'Pat-CL 1', 'Pat-CL 2', 'Pat-CL 3', 'Pat-CL 4'],
                'algorithm_level': ['Pat-AL 0', 'Pat-AL 1', 'Pat-AL 2', 'Pat-AL 3', 'Pat-AL 4']
            }
        
        # Run code-level fault injection
        for pattern_name in patterns.get('code_level', []):
            logger.info(f"Running {pattern_name}")
            
            faulty_codes = self.inject_code_level_faults(
                n_version_codes, pattern_name
            )
            
            voting_results = self._run_voting(faulty_codes, test_cases)
            metrics = self.fault_metrics.compute_all(voting_results)
            
            results['code_level'][pattern_name] = {
                'metrics': metrics,
                'num_faulty': self.fault_patterns[pattern_name].num_faulty_versions
            }
        
        # Run algorithm-level fault injection
        for pattern_name in patterns.get('algorithm_level', []):
            logger.info(f"Running {pattern_name}")
            
            faulty_codes = self.inject_algorithm_level_faults(
                n_version_codes, pattern_name
            )
            
            voting_results = self._run_voting(faulty_codes, test_cases)
            metrics = self.fault_metrics.compute_all(voting_results)
            
            results['algorithm_level'][pattern_name] = {
                'metrics': metrics,
                'pattern': pattern_name
            }
        
        # Compute summary
        results['summary'] = self._compute_summary(results)
        
        logger.info("Fault injection experiment completed")
        return results
    
    def inject_code_level_faults(
        self,
        codes: List[str],
        pattern_name: str
    ) -> List[str]:
        """
        Inject code-level faults according to pattern.
        
        Args:
            codes: List of original codes
            pattern_name: Fault pattern name (e.g., 'Pat-CL 1')
            
        Returns:
            List of codes with faults injected
        """
        pattern = self.fault_patterns.get(pattern_name)
        if not pattern:
            logger.warning(f"Unknown pattern: {pattern_name}")
            return codes.copy()
        
        n = len(codes)
        num_faulty = min(pattern.num_faulty_versions, n)
        
        if num_faulty <= 0:
            return codes.copy()
        
        # Select which versions to inject faults
        faulty_indices = random.sample(range(n), num_faulty)
        
        faulty_codes = codes.copy()
        for idx in faulty_indices:
            faulty_codes[idx] = self._introduce_code_fault(codes[idx])
        
        logger.debug(f"Injected faults in versions: {faulty_indices}")
        return faulty_codes
    
    def inject_algorithm_level_faults(
        self,
        codes: List[str],
        pattern_name: str
    ) -> List[str]:
        """
        Inject algorithm-level CMFs according to pattern.
        
        Args:
            codes: List of original codes
            pattern_name: Fault pattern name (e.g., 'Pat-AL 1')
            
        Returns:
            List of codes with CMFs injected
        """
        pattern = self.fault_patterns.get(pattern_name)
        if not pattern:
            logger.warning(f"Unknown pattern: {pattern_name}")
            return codes.copy()
        
        num_cmfs = pattern.num_faulty_versions
        if num_cmfs == -1:
            # All available CMFs
            num_cmfs = len(self.cmf_library)
        
        if num_cmfs <= 0:
            return codes.copy()
        
        # Select CMFs to apply
        selected_cmfs = self.cmf_library[:num_cmfs]
        
        # Apply CMFs to ALL versions (algorithm-level)
        faulty_codes = []
        for code in codes:
            faulty_code = self._apply_cmfs(code, selected_cmfs)
            faulty_codes.append(faulty_code)
        
        logger.debug(f"Applied {num_cmfs} CMFs to all versions")
        return faulty_codes
    
    def _introduce_code_fault(self, code: str) -> str:
        """
        Introduce a code-level fault.
        
        Types of faults:
        - Syntax errors (commented out for valid code)
        - Semantic errors
        - Logic errors
        """
        fault_types = [
            self._inject_off_by_one,
            self._inject_wrong_operator,
            self._inject_wrong_condition,
            self._inject_wrong_return,
        ]
        
        # Try each fault type until one works
        for fault_func in fault_types:
            faulty_code = fault_func(code)
            if faulty_code != code:
                return faulty_code
        
        # Fallback: return modified code
        return self._inject_generic_fault(code)
    
    def _inject_off_by_one(self, code: str) -> str:
        """Inject off-by-one error."""
        # Modify range bounds
        return re.sub(
            r'range\((\w+)\)',
            r'range(\1 - 1)',
            code,
            count=1
        )
    
    def _inject_wrong_operator(self, code: str) -> str:
        """Inject wrong operator."""
        # Change + to -
        if '+' in code and '++' not in code and '+=' not in code:
            return code.replace('+', '-', 1)
        return code
    
    def _inject_wrong_condition(self, code: str) -> str:
        """Inject wrong condition."""
        # Change <= to <
        if '<=' in code:
            return code.replace('<=', '<', 1)
        if '>=' in code:
            return code.replace('>=', '>', 1)
        return code
    
    def _inject_wrong_return(self, code: str) -> str:
        """Inject wrong return value."""
        # Modify return statement
        match = re.search(r'return\s+(\w+)\s*$', code, re.MULTILINE)
        if match:
            var = match.group(1)
            return code.replace(
                f'return {var}',
                f'return {var} + 1',
                1
            )
        return code
    
    def _inject_generic_fault(self, code: str) -> str:
        """Inject a generic fault by modifying a number."""
        # Find and modify a number
        match = re.search(r'\b(\d+)\b', code)
        if match:
            num = int(match.group(1))
            return code.replace(str(num), str(num + 1), 1)
        return code
    
    def _apply_cmfs(
        self,
        code: str,
        cmfs: List[Dict[str, Any]]
    ) -> str:
        """Apply multiple CMFs to code."""
        result = code
        for cmf in cmfs:
            pattern = cmf.get('pattern', '')
            replacement = cmf.get('replacement', '')
            result = re.sub(pattern, replacement, result, count=1)
        return result
    
    def _run_voting(
        self,
        codes: List[str],
        test_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run all versions on test cases and perform voting.
        
        Args:
            codes: List of code versions
            test_cases: List of test cases
            
        Returns:
            List of voting results
        """
        voting_results = []
        
        for tc in test_cases:
            outputs = []
            
            for code in codes:
                output = self._execute_code(code, tc)
                outputs.append(output)
            
            # Perform voting
            voted_result, vote_info = self.fault_metrics.majority_vote(outputs)
            
            # Check correctness
            expected = tc.get('expected_output')
            correct = self._compare_outputs(voted_result, expected)
            
            voting_results.append({
                'outputs': outputs,
                'voted_result': voted_result,
                'expected': expected,
                'system_correct': correct,
                'has_majority_consensus': vote_info['has_majority'],
                'all_agree': vote_info['all_agree']
            })
        
        return voting_results
    
    def _execute_code(
        self,
        code: str,
        test_case: Dict[str, Any]
    ) -> Any:
        """Execute code on a test case."""
        try:
            result = self.code_interpreter.execute({
                'code': code,
                'test_input': test_case.get('input')
            })
            
            if result['success']:
                return result.get('output', '').strip()
            else:
                return None
        except Exception:
            return None
    
    def _compare_outputs(self, result: Any, expected: Any) -> bool:
        """Compare result with expected output."""
        if result is None:
            return False
        
        if result == expected:
            return True
        
        try:
            return str(result) == str(expected)
        except:
            return False
    
    def _compute_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute experiment summary."""
        summary = {
            'code_level_avg_failure_rate': 0.0,
            'algorithm_level_avg_failure_rate': 0.0,
        }
        
        # Average code-level failure rate
        cl_failures = [
            r['metrics']['failure_rate']
            for r in results['code_level'].values()
        ]
        if cl_failures:
            summary['code_level_avg_failure_rate'] = sum(cl_failures) / len(cl_failures)
        
        # Average algorithm-level failure rate
        al_failures = [
            r['metrics']['failure_rate']
            for r in results['algorithm_level'].values()
        ]
        if al_failures:
            summary['algorithm_level_avg_failure_rate'] = sum(al_failures) / len(al_failures)
        
        return summary

