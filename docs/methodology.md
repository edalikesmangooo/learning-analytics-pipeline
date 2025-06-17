#### **docs/methodology.md**
```markdown
# Research Methodology

## Overview

This framework implements a novel approach to behavioral change detection in educational gaming environments, specifically designed for multiplayer learning scenarios.

## Key Innovations

### 1. Dual Z-Score Analysis
- **Reference-based method**: Compares behavior to large reference dataset
- **Session-specific method**: Identifies unusual patterns within individual sessions
- **Combined approach**: Provides comprehensive behavioral analysis

### 2. Strategic Focus vs. Behavioral Scattering Theory
Our research reveals fundamental patterns in educational gaming:
- **Successful learners**: Low scattering ratio (0.2-0.3) + High strategic diversity
- **Struggling learners**: High scattering ratio (0.6+) + Limited strategic range

### 3. Action vs. State Separation
Novel methodology distinguishing:
- **Inherited states**: What players receive from previous actions
- **Active decisions**: What players deliberately choose to do

## Technical Implementation

### Change Point Detection Pipeline
1. **Data preprocessing**: Session segmentation and player inference
2. **Feature engineering**: Event aggregation and temporal windowing
3. **Statistical analysis**: Dual z-score calculation
4. **Change detection**: Threshold-based identification
5. **Interpretation**: Key event extraction and visualization

### Validation Approach
- Synthetic data generation with ground truth
- Cross-method validation (Z-score, HMM, K-means)
- Performance metrics: AUC, precision, recall
- Educational outcome prediction accuracy

## Research Impact

This methodology enables:
- Real-time identification of learning difficulties
- Personalized intervention recommendations
- Educational game design optimization
- Scalable learning analytics deployment