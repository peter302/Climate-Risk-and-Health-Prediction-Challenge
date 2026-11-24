import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb

def run_model(train, test):
    # Clean columns we don't need for training
    drop_cols = ['ID', 'deathdate', 'location', 'is_climate_sensitive']
    features = [c for c in train.columns if c not in drop_cols]
    
    # Convert categories to category type
    for col in ['zone', 'gender']:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

    X = train[features]
    y = train['is_climate_sensitive']
    X_test = test[features]

    # Setup 5-fold split
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(train))
    test_predictions = np.zeros(len(test))

    # LGBM configuration settings
    model_settings = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'random_state': 42,
        'class_weight': 'balanced'
    }

    # Train across 5 folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        clf = lgb.LGBMClassifier(**model_settings)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        # Save fold results
        val_probs = clf.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = val_probs
        test_predictions += clf.predict_proba(X_test)[:, 1] / 5

    # Print summary performance metrics
    score_f1 = f1_score(y, (oof_predictions >= 0.5).astype(int))
    score_auc = roc_auc_score(y, oof_predictions)
    final_score = 0.60 * score_f1 + 0.40 * score_auc
    print(f"Local Validation -> F1: {score_f1:.4f} | AUC: {score_auc:.4f} | Weighted: {final_score:.4f}")
    
    return test_predictions, oof_predictions
