"""
evaluate_results.py
Analyze and visualize results
"""
# Create folder structure
import os
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Create basic README.md
readme_content = """
# Dysarthria Classification Project

## Proposed Solution
Enhanced Wav2Vec2 feature-based classifier with improved architecture including batch normalization, deeper layers, and better optimization.

## Files
- `model_proposed.py`: Proposed and baseline model architectures
- `run_experiments.py`: Experimental setup and training loops
- `evaluate_results.py`: Result analysis and visualization

## Results
See `results/` folder for performance comparisons and `plots/` for training curves.
"""

with open('README.md', 'w') as f:
    f.write(readme_content)
