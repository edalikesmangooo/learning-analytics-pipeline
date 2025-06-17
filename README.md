# Learning Analytics Pipeline for Learning Games

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://edalikesmangooo.github.io/learning-analytics/)

A comprehensive framework for analyzing learning patterns and behavioral changes in multiplayer educational games through advanced statistical methods and machine learning techniques.

## Key Research Contributions

### 1. **Novel Behavioral Change Detection Framework**
- **Dual Z-Score Analysis**: Reference-based vs. session-specific change point detection
- **Strategic Focus vs. Behavioral Scattering Theory**: Identification of learning success patterns
- **Action vs. State Separation**: Methodological breakthrough distinguishing inherited game states from active player decisions

### 2. **High-Performance Predictive Modeling**
- **94% AUC Achievement**: Robust prediction of learning outcomes using behavioral features
- **24+ Pathway Features**: Advanced spatial trajectory analysis for learning progression tracking
- **Multi-Method Validation**: Comparison of Z-score, HMM, K-means, and Bayesian change point detection

### 3. **Educational Impact & Applications**
- **Real-Time Learning Analytics**: Early identification of struggling students
- **Adaptive Intervention Framework**: Evidence-based recommendations for learning strategy optimization
- **Educational Game Design Insights**: Data-driven recommendations for effective learning progression

## Research Methodology

### Core Innovation: Strategic Focus vs. Behavioral Scattering
Our research reveals a fundamental paradigm in educational gaming:

- **Successful Learners**: Low scattering ratio (0.2-0.3) + High strategic diversity (13-18 unique behavioral states)
- **Unsuccessful Learners**: High scattering ratio (0.6+) + Limited strategic range (7-10 unique states)

### Technical Framework
```
Data Processing → Feature Engineering → Change Detection → Pattern Analysis
       ↓                 ↓                    ↓               ↓
  Session Inference  Event Aggregation   Z-Score Analysis  Learning Outcomes
  Player Inference   Pathway Features    HMM Modeling      Success Prediction
  Time Windowing     Delta Features      Clustering        Intervention Design
```

## Quick Start

### Installation
```bash
git clone https://github.com/edalikesmangooo/learning-analytics-pipeline.git
cd learning-analytics-pipeline
pip install -r requirements.txt
```

### Basic Usage
```python
from src.change_detection.zscore_method import BehavioralChangeDetector
from src.data_processing.session_inference import SessionProcessor

# Initialize the framework
detector = BehavioralChangeDetector()

# Load your educational game data
session_data = pd.read_csv('your_session_data.csv')

# Run complete analysis
results = detector.analyze_session(session_data, session_id='demo_session')

# Access change points
change_points = results['reference_analysis']['change_points']
print(f"Detected {len(change_points)} behavioral change points")
```

### Advanced Analysis
```python
# Multi-method comparison
from src.change_detection.hmm_method import HMMChangeDetector
from src.change_detection.clustering_method import ClusteringChangeDetector

# Compare different detection methods
detectors = {
    'zscore': BehavioralChangeDetector(),
    'hmm': HMMChangeDetector(n_states=3),
    'clustering': ClusteringChangeDetector(n_clusters=4)
}

results = {}
for name, detector in detectors.items():
    results[name] = detector.analyze_session(session_data)
```

## Interactive Demonstrations

### Live Dashboard
Explore our interactive analysis dashboard: [Learning Analytics Dashboard](https://huggingface.co/spaces/km3d4/rainbow-survey-data-viz)

### Documentation & Results
Comprehensive analysis documentation: [Research Documentation](https://edalikesmangooo.github.io/learning-analytics/)

## Architecture

### Core Modules

#### 1. Data Processing (`src/data_processing/`)
- **Session Inference**: Intelligent session segmentation using ≥40-second gaps
- **Player Inference**: Multi-criteria player assignment with collaborative action tracking
- **Time Windowing**: Adaptive temporal binning based on session duration

#### 2. Feature Engineering (`src/feature_engineering/`)
- **Event Aggregation**: Time-windowed behavioral event counting
- **Z-Score Calculation**: Dual-method statistical analysis
- **Pathway Analysis**: Spatial trajectory feature extraction

#### 3. Change Detection (`src/change_detection/`)
- **Z-Score Method**: Statistical anomaly detection for behavioral shifts
- **HMM Method**: Hidden Markov Model approach for state transitions
- **Clustering Method**: K-means based behavioral regime identification

#### 4. Visualization (`src/visualization/`)
- **Session Plots**: Comprehensive behavioral timeline visualization
- **Pathway Visualization**: Learning trajectory mapping and analysis

### Data Flow
```
Raw Game Logs → Session Segmentation → Player Assignment → Time Windowing
       ↓
Event Aggregation → Feature Engineering → Change Detection
       ↓
Statistical Analysis → Pattern Recognition → Learning Outcome Prediction
       ↓
Visualization → Intervention Recommendations → Educational Insights
```

## Research Results

### Key Findings

#### Behavioral Pattern Discovery
- **Multiple Failure Patterns**: Over-exploration vs. under-engagement vs. strategic incoherence
- **Learning Phase Transitions**: Early exploration → Mid-game strategy → Late optimization
- **Collaborative vs. Individual Learning**: Different success patterns for collaborative gameplay

#### Predictive Performance
| Method | AUC Score | Precision | Recall | F1-Score |
|--------|-----------|-----------|--------|----------|
| Logistic Regression | 0.94 | 0.89 | 0.91 | 0.90 |
| Multinomial Naive Bayes | 0.92 | 0.87 | 0.88 | 0.87 |
| Random Forest | 0.91 | 0.85 | 0.89 | 0.87 |

#### Change Point Detection Validation
- **Reference Method**: Higher sensitivity to global behavioral patterns
- **Session Method**: Better detection of within-session learning transitions
- **Combined Approach**: Optimal balance of specificity and sensitivity

## Methodology Details

### Innovation 1: Action vs. State Analysis
Traditional approaches analyze cumulative game states. Our framework separates:
- **Inherited States**: What players receive from previous actions
- **Active Decisions**: What players deliberately choose to do

This separation enables clearer understanding of strategic decision-making vs. circumstantial gameplay.

### Innovation 2: Collaborative Action Tracking
Enhanced player inference that captures:
- **Individual Actions**: Clear single-player decisions
- **Collaborative Actions**: Shared decision-making moments
- **Ambiguous Actions**: Intelligent assignment based on temporal context

### Innovation 3: Multi-Scale Temporal Analysis
Adaptive time windowing that adjusts to session characteristics:
- **Short Sessions (< 5 min)**: Percentage-based normalized windows
- **Long Sessions (≥ 5 min)**: Fixed 1-minute real-time windows
- **Variable Granularity**: 5-20 windows per session based on content density

## Applications

### Educational Technology
- **Adaptive Learning Systems**: Real-time difficulty adjustment based on behavioral patterns
- **Student Assessment**: Automated evaluation of learning progression and engagement
- **Curriculum Design**: Data-driven optimization of educational content sequencing

### Game Development
- **Player Experience Optimization**: Identification of engagement drop-off points
- **Content Balancing**: Statistical analysis of game mechanic effectiveness
- **Social Learning Design**: Framework for multiplayer educational game development

### Learning Sciences Research
- **Behavioral Pattern Analysis**: Quantitative framework for learning strategy research
- **Intervention Effectiveness**: Measurement tools for educational intervention impact
- **Cross-Platform Analysis**: Generalizable methodology for different educational contexts

## Documentation

### API Reference
- [Core Classes Documentation](docs/api_reference.md)
- [Method Comparisons](docs/methodology.md)
- [Example Notebooks](examples/)

### Research Papers
- *Strategic Focus vs. Behavioral Scattering in Educational Games* (In Preparation)
- *Multi-Method Change Point Detection for Learning Analytics* (Under Review)
- *Collaborative Learning Pattern Recognition in Digital Environments* (Submitted)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone https://github.com/yourusername/learning-analytics-pipeline.git
cd learning-analytics-pipeline

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Generate documentation
cd docs && make html
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{EZhang2025learning,
  title={Strategic Focus vs. Behavioral Scattering: A Novel Framework for Learning Analytics in Educational Games},
  author={Your Name and Collaborators},
  journal={Journal of Educational Technology Research},
  year={2025},
  publisher={Educational Technology Publishers}
}
```

## Contact & Support

- **Primary Author**: Eda - eda.zhang@wisc.edu
- **Research Group**: Complex Play
- **Institution**: UW-Madison



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Research Collaborators**: Matthew Berland, David Weindtrop, Stephen Uzzo, Leilah Lyons, Katie Culp, Mac Cannady
- **Data Providers**: NYSCI, Lawrence Hall of Science
- **Funding**: NSF #1713439


---

