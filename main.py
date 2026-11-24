"""
Entry point for the Climate Risk & Health Prediction pipeline.

Usage:
    python main.py --train Train.csv --test Test.csv --climate climate_features.csv
"""

import argparse
import pandas as pd
import numpy as np

from data import load_data
from feature_engineering import add_features
from model import run_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',   default='Train.csv')
    parser.add_argument('--test',    default='Test.csv')
    parser.add_argument('--climate', default='climate_features.csv')
    parser.add_argument('--out',     default='submission.csv')
    args = parser.parse_args()

    print("── Loading data ─────────────────────────────")
    train, test = load_data(args.train, args.test, args.climate)

    print("\n── Feature engineering ──────────────────────")
    train, test = add_features(train, test)
    print(f"  train shape: {train.shape}  |  test shape: {test.shape}")

    print("\n── Training ─────────────────────────────────")
    test_preds, oof_preds, threshold = run_model(train, test)

    print("\n── Saving submission ────────────────────────")
    # Competition requires TWO columns:
    #   TargetF1   – binary prediction, scored on F1  (threshold-tuned)
    #   TargetRAUC – probability,       scored on AUC (raw, unclamped)
    submission = pd.DataFrame({
        'ID':         test['ID'],
        'TargetF1':   (test_preds >= threshold).astype(int),
        'TargetRAUC': test_preds,
    })
    submission.to_csv(args.out, index=False)
    print(f"  Saved → {args.out}  (threshold={threshold:.2f})")

if __name__ == '__main__':
    main()
