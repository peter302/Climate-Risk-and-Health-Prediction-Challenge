import pandas as pd

def add_features(train, test):
    # Extract basic dates and differences
    for dataset in [train, test]:
        dataset['deathdate'] = pd.to_datetime(dataset['deathdate'], errors='coerce')
        dataset['day_of_year'] = dataset['deathdate'].dt.dayofyear
        dataset['month'] = dataset['deathdate'].dt.month
        dataset['year'] = dataset['deathdate'].dt.year
        dataset['temperature_range'] = dataset['max_temperature'] - dataset['min_temperature']
        dataset['is_rainy_day_current'] = (dataset['precipitation'] > 0).astype(int)
    return train, test
