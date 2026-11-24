import pandas as pd

def load_data(train_path, test_path, climate_path):
    # Load the main tables
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    climate = pd.read_csv(climate_path)

    # Get rid of extra deathdate column in climate features to avoid confusion
    if 'deathdate' in climate.columns:
        climate = climate.drop(columns=['deathdate'])

    # Merge everything on ID
    train_data = train.merge(climate, on='ID', how='left')
    test_data = test.merge(climate, on='ID', how='left')
    
    return train_data, test_data
