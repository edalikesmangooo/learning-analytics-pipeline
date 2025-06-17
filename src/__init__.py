"""
Learning Analytics Pipeline for Educational Games
================================================

A comprehensive framework for analyzing learning patterns and behavioral 
changes in multiplayer educational games through advanced statistical methods.

Key Features:
- Novel behavioral change detection using dual z-score analysis
- Strategic Focus vs. Behavioral Scattering theory
- Comprehensive visualization and analysis tools
- Synthetic data generation for testing and validation

Author: [Eda]
Institution: [UW-Madison]
"""

__version__ = "1.0.0"
__author__ = "Eda"
__email__ = "eda.zhang@wisc.edu"

# Core imports - will be available after implementing the modules
try:
    from .change_detection.zscore_method import BehavioralChangeDetector
    from .visualization.session_plots import BehavioralVisualization
    from .data_processing.synthetic_data import generate_synthetic_session_data
    
    __all__ = [
        "BehavioralChangeDetector",
        "BehavioralVisualization", 
        "generate_synthetic_session_data"
    ]
except ImportError:
    # During initial setup, modules might not exist yet
    __all__ = []

# Package metadata
__title__ = "learning-analytics-pipeline"
__description__ = "Behavioral change detection framework for educational games"
__url__ = "https://github.com/edalikesmangooo/learning-analytics-pipeline"
__license__ = "MIT"
__copyright__ = "Copyright 2025 nmaking"