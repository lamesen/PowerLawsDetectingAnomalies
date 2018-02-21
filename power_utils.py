import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics, cluster
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm


def import_data():
    """Data munging for Power Laws: Detecting Anomalies in Usage competition
    https://www.drivendata.org/competitions/52/anomaly-detection-electricity/page/102/

    :return: train
    a pandas DataFrame containing a combined training set.
    """

    # Read all data from csv
    data = {
        'tra': pd.read_csv('./input/train.csv',
                           index_col=0,
                           header=0,
                           names=['obs_id', 'meter_id', 'Timestamp', 'Values'],
                           parse_dates=['Timestamp'],
                           low_memory=False),
        'meta': pd.read_csv('./input/metadata.csv',
                            index_col=False,
                            low_memory=False),
        'hol': pd.read_csv('./input/holidays.csv',
                           index_col=0,
                           parse_dates=['Date'],
                           low_memory=False),
        'wx': pd.read_csv('./input/weather.csv',
                          index_col=0,
                          header=0,
                          names=['wx_id', 'Timestamp', 'Temperature', 'Distance', 'site_id'],
                          parse_dates=['Timestamp'],
                          low_memory=False)
    }

    # Date object creation and formatting
    for df in ['tra', 'wx']:
        data[df]['Date'] = data[df]['Timestamp'].dt.date
    data['hol']['Date'] = data['hol']['Date'].dt.date

    # Join datasets
    train = pd.merge(data['tra'], data['meta'], how='left', on=['meter_id'])
    train = pd.merge(train, data['hol'], how='left', on=['Date', 'site_id'])
    train = pd.merge(train, data['wx'], how='left', on=['Timestamp', 'site_id'])
    train = train.drop(['Date_y'], axis=1)
    train = train.rename(index=str, columns={'Date_x': 'Date'})

    # Handle fill missing values with -1 flag
    train = train.fillna(-1)

    return train
