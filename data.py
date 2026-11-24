import pandas as pd
import numpy as np


def load_data(train_path, test_path, climate_path):
    train   = pd.read_csv(train_path)
    test    = pd.read_csv(test_path)
    climate = pd.read_csv(climate_path)

    # deathdate shows up in climate_features too — drop it before merging
    if 'deathdate' in climate.columns:
        climate = climate.drop(columns=['deathdate'])

    train_data = train.merge(climate, on='ID', how='left')
    test_data  = test.merge(climate, on='ID', how='left')

    # flag any columns that didn't join cleanly
    for name, df in [("train", train_data), ("test", test_data)]:
        miss = df.isnull().sum()
        miss = miss[miss > 0]
        if not miss.empty:
            print(f"{name} has nulls after merge:")
            print(miss.to_string())

    return train_data, test_data
