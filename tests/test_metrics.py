"""
Tests for DeQoG Metrics
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from metrics.diversity_metrics import DiversityMetrics
from metrics.correctness_metrics import CorrectnessMetrics
from metrics.fault_tolerance_metrics import FaultToleranceMetrics


class TestDiversityMetrics(unittest.TestCase):
    """Test cases for DiversityMetrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = DiversityMetrics()
    
    def test_compute_mbcs_empty(self):
        """Test MBCS with empty list."""
        result = self.metrics.compute_mbcs([])
        self.assertEqual(result, 0.0)
    
    def test_compute_mbcs_single(self):
        """Test MBCS with single item."""
        result = self.metrics.compute_mbcs(["code"])
        self.assertEqual(result, 0.0)
    
    def test_compute_mbcs_range(self):
        """Test MBCS returns value in valid range."""
        codes = [
            "def add(a, b): return a + b",
            "def subtract(a, b): return a - b",
            "def multiply(a, b): return a * b"
        ]
        
        result = self.metrics.compute_mbcs(codes)
        
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    def test_compute_all(self):
        """Test computing all metrics."""
        codes = ["code1", "code2"]
        
        result = self.metrics.compute_all(codes)
        
        self.assertIn('mbcs', result)
        self.assertIn('semantic_diversity', result)


class TestCorrectnessMetrics(unittest.TestCase):
    """Test cases for CorrectnessMetrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = CorrectnessMetrics()
    
    def test_compute_pass_rate_all_pass(self):
        """Test pass rate with all passing tests."""
        results = [
            {'passed': True},
            {'passed': True},
            {'passed': True}
        ]
        
        rate = self.metrics.compute_pass_rate(results)
        
        self.assertEqual(rate, 1.0)
    
    def test_compute_pass_rate_all_fail(self):
        """Test pass rate with all failing tests."""
        results = [
            {'passed': False},
            {'passed': False}
        ]
        
        rate = self.metrics.compute_pass_rate(results)
        
        self.assertEqual(rate, 0.0)
    
    def test_compute_pass_rate_mixed(self):
        """Test pass rate with mixed results."""
        results = [
            {'passed': True},
            {'passed': False},
            {'passed': True},
            {'passed': False}
        ]
        
        rate = self.metrics.compute_pass_rate(results)
        
        self.assertEqual(rate, 0.5)
    
    def test_compute_tpr(self):
        """Test TPR across versions."""
        results_per_version = [
            [{'passed': True}, {'passed': True}],  # 100%
            [{'passed': True}, {'passed': False}],  # 50%
            [{'passed': False}, {'passed': False}]  # 0%
        ]
        
        tpr = self.metrics.compute_tpr(results_per_version)
        
        self.assertEqual(tpr, 0.5)
    
    def test_common_failures(self):
        """Test identification of common failures."""
        version_results = {
            'v1': [{'passed': True}, {'passed': False}, {'passed': False}],
            'v2': [{'passed': True}, {'passed': True}, {'passed': False}],
            'v3': [{'passed': False}, {'passed': True}, {'passed': False}],
        }
        
        common = self.metrics.common_failures(version_results)
        
        # Test index 2 fails in all versions
        self.assertIn(2, common)


class TestFaultToleranceMetrics(unittest.TestCase):
    """Test cases for FaultToleranceMetrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = FaultToleranceMetrics()
    
    def test_compute_failure_rate(self):
        """Test failure rate computation."""
        voting_results = [
            {'system_correct': True},
            {'system_correct': True},
            {'system_correct': False},
            {'system_correct': True}
        ]
        
        rate = self.metrics.compute_failure_rate(voting_results)
        
        self.assertEqual(rate, 0.25)
    
    def test_compute_mcr(self):
        """Test MCR computation."""
        voting_results = [
            {'has_majority_consensus': True},
            {'has_majority_consensus': True},
            {'has_majority_consensus': False}
        ]
        
        mcr = self.metrics.compute_mcr(voting_results)
        
        self.assertAlmostEqual(mcr, 2/3)
    
    def test_compute_ccr(self):
        """Test CCR computation."""
        voting_results = [
            {'all_agree': True},
            {'all_agree': False},
            {'all_agree': True}
        ]
        
        ccr = self.metrics.compute_ccr(voting_results)
        
        self.assertAlmostEqual(ccr, 2/3)
    
    def test_majority_vote(self):
        """Test majority voting."""
        outputs = [1, 1, 2, 1, 3]
        
        result, info = self.metrics.majority_vote(outputs)
        
        self.assertEqual(result, 1)
        self.assertTrue(info['has_majority'])
        self.assertEqual(info['winning_count'], 3)
    
    def test_majority_vote_no_majority(self):
        """Test voting without clear majority."""
        outputs = [1, 2, 3, 4]
        
        result, info = self.metrics.majority_vote(outputs)
        
        self.assertFalse(info['has_majority'])
    
    def test_majority_vote_unanimous(self):
        """Test unanimous voting."""
        outputs = [42, 42, 42]
        
        result, info = self.metrics.majority_vote(outputs)
        
        self.assertEqual(result, 42)
        self.assertTrue(info['all_agree'])
    
    def test_compute_all(self):
        """Test computing all metrics."""
        voting_results = [
            {
                'system_correct': True,
                'has_majority_consensus': True,
                'all_agree': True
            },
            {
                'system_correct': False,
                'has_majority_consensus': True,
                'all_agree': False
            }
        ]
        
        metrics = self.metrics.compute_all(voting_results)
        
        self.assertIn('failure_rate', metrics)
        self.assertIn('mcr', metrics)
        self.assertIn('ccr', metrics)
        self.assertIn('success_rate', metrics)


if __name__ == '__main__':
    unittest.main()

