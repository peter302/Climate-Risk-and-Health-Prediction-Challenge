import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb

# ── columns to never use as features ─────────────────────────────────────────
DROP_COLS = ['ID', 'deathdate', 'location', 'is_climate_sensitive',
             'latitude', 'longitude',           # replaced by lat_bin, lon_bin, lat_lon_product
             'zone_mean_temp', 'zone_std_temp']  # leakage-prone aggregates kept separately

# ── competition scoring ───────────────────────────────────────────────────────
def competition_score(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    f1  = f1_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    return 0.60 * f1 + 0.40 * auc, f1, auc

def find_best_threshold(y_true, probs, search=np.arange(0.3, 0.71, 0.01)):
    """Grid-search the threshold that maximises the weighted competition metric."""
    best_t, best_s = 0.5, -1
    for t in search:
        s, _, _ = competition_score(y_true, probs, threshold=t)
        if s > best_s:
            best_s, best_t = s, t
    return best_t, best_s

# ── model config ──────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    'objective':        'binary',
    'metric':           'auc',
    'learning_rate':    0.03,       # slower → better generalisation
    'num_leaves':       63,         # was 31; more expressive
    'max_depth':        -1,         # let num_leaves control depth
    'min_child_samples': 20,        # prevent overfit on small leaves
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq':     1,
    'reg_alpha':        0.1,        # L1 regularisation
    'reg_lambda':       1.0,        # L2 regularisation
    'verbose':         -1,
    'random_state':     42,
    'class_weight':    'balanced',
    'n_estimators':    2000,        # large; early stopping will trim this
}

def run_model(train, test, n_splits=5):
    drop = [c for c in DROP_COLS if c in train.columns]
    features = [c for c in train.columns if c not in drop]

    # Encode categoricals
    cat_cols = [c for c in ['zone', 'gender', 'season', 'geo_cell'] if c in features]
    for col in cat_cols:
        train[col] = train[col].astype('category')
        test[col]  = test[col].astype('category')

    X, y    = train[features], train['is_climate_sensitive']
    X_test  = test[features]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds  = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        clf = lgb.LGBMClassifier(**LGBM_PARAMS)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(period=-1)]
        )

        val_probs = clf.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs
        test_preds += clf.predict_proba(X_test)[:, 1] / n_splits

        fold_s, fold_f1, fold_auc = competition_score(y_val, val_probs)
        fold_scores.append(fold_s)
        print(f"  Fold {fold+1}: F1={fold_f1:.4f}  AUC={fold_auc:.4f}  "
              f"Score={fold_s:.4f}  trees={clf.best_iteration_}")

    # ── find optimal threshold on OOF ────────────────────────────────────────
    best_t, best_s = find_best_threshold(y, oof_preds)
    final_s, final_f1, final_auc = competition_score(y, oof_preds, threshold=best_t)

    print(f"\nOOF Summary")
    print(f"  Threshold (tuned): {best_t:.2f}")
    print(f"  F1={final_f1:.4f}  AUC={final_auc:.4f}  "
          f"Weighted Score={final_s:.4f}  "
          f"(CV std={np.std(fold_scores):.4f})")

    # ── feature importance ────────────────────────────────────────────────────
    fi = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop 15 features:")
    print(fi.head(15).to_string())

    # Return raw probs for TargetRAUC and best threshold for TargetF1
    return test_preds, oof_preds, best_t
