import numba
import pandas as pd
# import numpy as np
from sklearn.preprocessing import MinMaxScaler


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

    final.to_csv('./tmp/train_merged.csv')

    return final


def create_train_set():
    """This function performs feature engineering and further data cleansing and produces a train and test set
    ::param:: dataset
    the input data set to be prepared for modeling

    ::return:: train, test
    two Pandas DataFrame objects with train and test set splits
    """
    dataset = import_data()
    dataset.sort_values(by=['meter_id', 'timestamp'], inplace=True)
    # dataset = dataset.set_index('timestamp')
    # dataset.index = dataset.index.to_datetime()

    # Drop meter id: Too many levels
    dataset.drop('meter_id', inplace=True, axis=1)

    # Convert Wh to kWh
    dataset['values'] = dataset.apply(lambda r: (r['values'] / 1000) if r['units'] == 'Wh' else r['values'], axis=1)
    dataset['units'] = dataset.apply(lambda r: 'kWh' if r['units'] == 'Wh' else r['units'], axis=1)

    # Lags
    # dataset['values_business_day_lag_1'] = dataset['values'].tshift(1, freq='B')
    # dataset['values_day_lag_1'] = dataset['values'].tshift(1, freq='D')
    # dataset['values_day_lag_7'] = dataset['values'].tshift(7, freq='D')
    #
    # # Differencing
    # dataset['values_business_day_diff_1'] = dataset['values'] - dataset['values_business_day_lag_1']
    # dataset['values_day_diff_1'] = dataset['values'] - dataset['values_day_lag_1']
    # dataset['values_day_diff_7'] = dataset['values'] - dataset['values_day_lag_7']

    # Day of the week
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset['dow'] = dataset['date'].dt.dayofweek
    dataset['wom'] = (dataset['date'].dt.day - 1) // 7 + 1
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.month
    dataset['hour'] = dataset['date'].dt.hour
    dataset['minute'] = dataset['date'].dt.minute
    dataset.drop('date', inplace=True, axis=1)

    # Business Hours
    # dataset['business_hours'] = dataset.apply(lambda r: 1 if 8 <= r.index.hour < 17 else 0)
    # dataset['evening'] = dataset.apply(lambda r: 1 if 17 <= r.index.hour < 23 else 0)
    # dataset['overnight'] = dataset.apply(lambda r: 0 if r['business_hours'] == 1 or r['evening'] == 1 else 0)

    # Dummy coding
    categorical_vars = ['site_id', 'meter_description', 'units', 'activity']
    for var in categorical_vars:
        s = pd.get_dummies(dataset[var], prefix=var)
        s.columns = s.columns.str.replace(" ", "_")
        dataset.drop(var, inplace=True, axis=1)
        dataset = pd.concat([dataset, s], axis=1)

    # Center and scale variables
    numerical_vars = ['surface', 'temperature', 'distance']
    min_max_scaler = MinMaxScaler()
    s = pd.DataFrame(min_max_scaler.fit_transform(dataset[numerical_vars].fillna(-1)), columns=numerical_vars)
    s.index = dataset.index
    dataset.drop(numerical_vars, inplace=True, axis=1)
    dataset = pd.concat([dataset, s], axis=1)

    # Split the data set
    train = dataset
    dataset.to_csv('./tmp/train_prepared.csv')

    return train
