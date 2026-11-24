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

    print("loading data...")
    train, test = load_data(args.train, args.test, args.climate)

    print("feature engineering...")
    train, test = add_features(train, test)
    print(f"train: {train.shape}  test: {test.shape}")

    print("training...")
    test_preds, oof_preds, threshold = run_model(train, test)

    # competition wants two separate columns:
    # TargetF1 is the binary prediction (uses tuned threshold)
    # TargetRAUC is the raw probability (scored independently on AUC)
    submission = pd.DataFrame({
        'ID':         test['ID'],
        'TargetF1':   (test_preds >= threshold).astype(int),
        'TargetRAUC': test_preds,
    })
    submission.to_csv(args.out, index=False)
    print(f"saved {args.out} (threshold={threshold:.2f})")


if __name__ == '__main__':
    main()
