import pandas as pd
import numpy as np

# tweak these if the data distribution looks different
HEAT_THRESHOLD  = 35
FROST_THRESHOLD = 0
HEAVY_RAIN_MM   = 20
HIGH_WIND_MS    = 10


def add_features(train, test):
    for df in [train, test]:

        df['deathdate']   = pd.to_datetime(df['deathdate'], errors='coerce')
        df['day_of_year'] = df['deathdate'].dt.dayofyear
        df['month']       = df['deathdate'].dt.month
        df['year']        = df['deathdate'].dt.year
        df['day_of_week'] = df['deathdate'].dt.dayofweek

        # month as a raw int makes dec and jan look far apart, sin/cos fixes that
        df['month_sin']     = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos']     = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_yr_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_yr_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,
             3: 1, 4: 1, 5: 1,
             6: 2, 7: 2, 8: 2,
             9: 3, 10: 3, 11: 3
        })

        df['temperature_range'] = df['max_temperature'] - df['min_temperature']
        df['mean_temperature']  = (df['max_temperature'] + df['min_temperature']) / 2

        # simplified Steadman heat index — only bother if humidity is actually in the data
        T = df['mean_temperature']
        H = df.get('humidity', pd.Series(np.nan, index=df.index))
        if H.notna().mean() > 0.5:
            df['heat_index'] = (
                -8.78469 + 1.61139*T + 2.33855*H
                - 0.14611*T*H - 0.01230*T**2
                - 0.01642*H**2 + 0.00221*T**2*H
                + 0.00072*T*H**2 - 0.00000358*T**2*H**2
            )
            df['heat_index'] = df['heat_index'].clip(lower=df['mean_temperature'])

        df['is_extreme_heat']    = (df['max_temperature'] > HEAT_THRESHOLD).astype(int)
        df['is_frost_day']       = (df['min_temperature'] < FROST_THRESHOLD).astype(int)
        df['is_high_temp_range'] = (df['temperature_range'] > df['temperature_range'].quantile(0.75)).astype(int)

        df['is_rainy_day']  = (df['precipitation'] > 0).astype(int)
        df['is_heavy_rain'] = (df['precipitation'] > HEAVY_RAIN_MM).astype(int)
        df['log_precip']    = np.log1p(df['precipitation'])

        if 'wind_speed' in df.columns:
            df['is_high_wind'] = (df['wind_speed'] > HIGH_WIND_MS).astype(int)

        if 'humidity' in df.columns:
            df['hot_humid'] = df['is_extreme_heat'] * df.get('humidity', 0)
        df['hot_rainy']   = df['is_extreme_heat'] * df['is_rainy_day']
        df['temp_season'] = df['mean_temperature'] * df['season']

        # how different is today from the last 30 days — these columns come from climate_features.csv
        if 'tavg_30d' in df.columns:
            df['temp_anomaly_30d'] = df['mean_temperature'] - df['tavg_30d']
        if 'rain_sum_30d' in df.columns:
            df['log_rain_30d']  = np.log1p(df['rain_sum_30d'])
            df['precip_vs_30d'] = df['precipitation'] - df['rain_sum_30d'] / 30

        if 'ndvi_30d' in df.columns:
            df['low_ndvi'] = (df['ndvi_30d'] < df['ndvi_30d'].quantile(0.25)).astype(int)
            df['hot_dry']  = df['is_extreme_heat'] * df['low_ndvi']

        if 'elevation' in df.columns:
            df['log_elevation'] = np.log1p(df['elevation'].clip(lower=0))
        if 'slope' in df.columns:
            df['high_slope'] = (df['slope'] > df['slope'].quantile(0.75)).astype(int)

        # bin lat/lon into a grid instead of dropping them raw
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['lat_bin']        = pd.cut(df['latitude'],  bins=20, labels=False)
            df['lon_bin']        = pd.cut(df['longitude'], bins=20, labels=False)
            df['lat_lon_product'] = df['latitude'] * df['longitude']

        if 'age' in df.columns:
            df['is_elderly']   = (df['age'] >= 65).astype(int)
            df['is_young']     = (df['age'] <= 5).astype(int)
            df['elderly_heat'] = df['is_elderly'] * df['is_extreme_heat']

        if 'zone' in df.columns:
            zone_stats = df.groupby('zone')['mean_temperature'].agg(
                zone_mean_temp='mean', zone_std_temp='std'
            ).reset_index()
            df = df.merge(zone_stats, on='zone', how='left')
            df['temp_vs_zone']     = df['mean_temperature'] - df['zone_mean_temp']
            df['precip_zone_mean'] = df.groupby('zone')['precipitation'].transform('mean')

    return train, test
