# Climate Risk Classification

This project contains code for our best model to classify whether recorded deaths fall into a climate-sensitive category.

## Quick Results
- **OOF F1-Score**: 0.7712
- **OOF AUC**: 0.8108
- **Final Score**: **0.7871**

## Project Setup
- `data.py`: Loads the datasets and combines them.
- `feature_engineering.py`: Computes dates and weather range columns.
- `model.py`: Performs 5-fold training with LightGBM.
