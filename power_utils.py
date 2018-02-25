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

    # Import all data from csv
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

    # Date feature creation and formatting
    for df in ['tra', 'wx']:
        data[df]['Date'] = data[df]['Timestamp'].dt.date
    data['hol']['Date'] = data['hol']['Date'].dt.date

    # Join train and meta DataFrames
    final = pd.merge(data['tra'], data['meta'], how='left', on=['meter_id'])

    # Clean holiday DataFrame and then join to the final DataFrame
    data['hol'].Holiday = 1
    data['hol'] = data['hol'].drop_duplicates()
    final = pd.merge(final, data['hol'], how='left', on=['Date', 'site_id'])
    final['Holiday'].fillna(value=0, inplace=True)

    # Clean weather DataFrame and then join to the final DataFrame
    data['wx'] = data['wx'].drop_duplicates()
    agg = data['wx'].groupby(['Timestamp', 'site_id']).mean()
    final = pd.merge(final, agg.reset_index(), how='left', on=['Timestamp', 'site_id'])
    final = final.rename(str.lower, axis='columns')

    return final


def create_train_test(dataset):
    """This function performs feature engineering and further data cleansing and produces a train and test set
    ::param:: dataset
    the input data set to be prepared for modeling

    ::return:: train, test
    two Pandas DataFrame objects with train and test set splits
    """
    # Convert Wh to kWh

    # Handle fill missing values with -1 flag
    # train = train.fillna(-1)
