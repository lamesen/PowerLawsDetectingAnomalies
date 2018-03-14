import pandas as pd
import numpy as np
from datetime import datetime


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
                          low_memory=False),
        'tes': pd.read_csv('./input/submission_format.csv',
                           parse_dates=['Timestamp'],
                           low_memory=False)
    }

    # Date feature creation and formatting
    for df in ['tra', 'wx', 'tes']:
        data[df]['Date'] = data[df]['Timestamp'].dt.date
    data['hol']['Date'] = data['hol']['Date'].dt.date

    # Join train and meta DataFrames
    train = pd.merge(data['tra'], data['meta'], how='left', on=['meter_id'])
    test = pd.merge(data['tes'], data['meta'], how='left', on=['meter_id'])

    # Clean holiday DataFrame and then join to the train, test DataFrame
    data['hol'].Holiday = 1
    data['hol'] = data['hol'].drop_duplicates()
    train = pd.merge(train, data['hol'], how='left', on=['Date', 'site_id'])
    train['Holiday'].fillna(value=0, inplace=True)

    test = pd.merge(test, data['hol'], how='left', on=['Date', 'site_id'])
    test['Holiday'].fillna(value=0, inplace=True)

    # Clean weather DataFrame and then join to the final DataFrame
    data['wx'] = data['wx'].drop_duplicates()
    agg = data['wx'].groupby(['Timestamp', 'site_id']).mean()

    train = pd.merge(train, agg.reset_index(), how='left', on=['Timestamp', 'site_id'])
    train = train.rename(str.lower, axis='columns')

    test = pd.merge(test, agg.reset_index(), how='left', on=['Timestamp', 'site_id'])
    test = test.rename(str.lower, axis='columns')

    train.to_csv('./tmp/train_merged.csv')
    test.to_csv('./tmp/test_merged.csv')

    return train, test


def create_train_set():
    """This function performs feature engineering and further data cleansing and produces a fully
    pre-processed training set

    ::return:: train
    a Pandas DataFrame object for analytical modeling
    """
    train, test = import_data()

    # Remove observations from the training set for meter_descriptions and activities not in the test set
    train = train[train.meter_description.isin(test.meter_description.unique().tolist())]
    train = train[train.activity.isin(test.activity.unique().tolist())]

    train.sort_values(by=['meter_id', 'timestamp'], inplace=True)

    # Convert Wh to kWh
    train['values'] = train.apply(lambda r: (r['values'] / 1000) if r['units'] == 'Wh' else r['values'], axis=1)
    train['units'] = train.apply(lambda r: 'kWh' if r['units'] == 'Wh' else r['units'], axis=1)
    train['obs_id'] = train.index

    # Lags
    lagging = train.set_index('timestamp')
    s = lagging.groupby('meter_id')['values'].shift(1, freq='B').reset_index()
    s = s.rename({'values': 'values_business_day_lag_1'}, axis='columns')
    train = pd.merge(train, s, how='left', on=['meter_id', 'timestamp'])

    s = lagging.groupby('meter_id')['values'].shift(1, freq='D').reset_index()
    s = s.rename({'values': 'values_day_lag_1'}, axis='columns')
    train = pd.merge(train, s, how='left', on=['meter_id', 'timestamp'])

    s = lagging.groupby('meter_id')['values'].shift(7, freq='D').reset_index()
    s = s.rename({'values': 'values_day_lag_7'}, axis='columns')
    train = pd.merge(train, s, how='left', on=['meter_id', 'timestamp'])

    # Differencing
    train['values_business_day_diff_1'] = train['values'] - train['values_business_day_lag_1']
    train['values_day_diff_1'] = train['values'] - train['values_day_lag_1']
    train['values_day_diff_7'] = train['values'] - train['values_day_lag_7']

    # Day of the week
    train['date'] = pd.to_datetime(train['timestamp'])
    train['dow'] = train['date'].dt.dayofweek
    train['wom'] = (train['date'].dt.day - 1) // 7 + 1
    train['year'] = train['date'].dt.year
    train['month'] = train['date'].dt.month
    train['day'] = train['date'].dt.day
    train['hour'] = train['date'].dt.hour
    train['minute'] = train['date'].dt.minute
    train.drop('date', inplace=True, axis=1)

    # Business Hours
    train['business_hours'] = train.apply(lambda r: 1 if 8 <= r['hour'] < 17 else 0, axis=1)
    train['evening'] = train.apply(lambda r: 1 if 17 <= r['hour'] < 23 or 0 <= r['hour'] < 8 else 0, axis=1)
    train['overnight'] = train.apply(lambda r: 0 if r['business_hours'] == 1 or r['evening'] == 1 else 0, axis=1)

    # Save the training set
    train.to_csv('./tmp/train_prepared.csv', index=False)

    return train


def absolute_difference(obs_val, pred_val):
    return np.absolute(pred_val - obs_val)


def score_predictions(train_loc, test_loc, prediction_loc, actual_colname,
                      prediction_colname, prediction_cutoff, output_note):
    train = pd.read_csv(train_loc, dtype={'meter_id': 'object', 'site_id': 'object'})
    predictions = pd.read_csv(prediction_loc)
    test = pd.read_csv(test_loc)

    train['predictions'] = predictions[prediction_colname]

    train_clean = train.drop_duplicates(['timestamp', 'meter_id'])

    # Left join the test subset with the train superset
    combined = pd.merge(train_clean, test, how='left', left_on=['timestamp', 'meter_id'],
                        right_on=['Timestamp', 'meter_id'])

    # Calculate absolute error for each 'values' prediction
    combined['abs_err'] = combined.apply(lambda r: absolute_difference(r[actual_colname], r['predictions']), axis=1)
    # Calculate the average within meter_id group of absolute errors
    combined = pd.merge(combined, combined.groupby('meter_id')['abs_err'].mean().reset_index(name='average_abs_err'),
                        how='left', on='meter_id')
    # Calculate the anomaly score ratio of absolute error over the average absolute error (within meter_id group)
    combined['anomaly_score'] = combined['abs_err'] / combined['average_abs_err']

    # What should the cutoff be?? experiment with quantile cutoffs
    anomaly_cutoff = combined['anomaly_score'].quantile(q=prediction_cutoff)
    print('Anomaly cutoff: {}'.format(anomaly_cutoff))
    test_final = combined.dropna(subset=['is_abnormal'])
    test_final['is_abnormal'] = test_final.apply(lambda r: r['anomaly_score'] > anomaly_cutoff,
                                                 axis=1)

    print(test_final.is_abnormal.value_counts())

    test_scored = test_final[['obs_id_y', 'meter_id', 'Timestamp', 'is_abnormal']]
    test_scored.columns = ['obs_id', 'meter_id', 'Timestamp', 'is_abnormal']
    test_scored['obs_id'] = test_scored.loc[:, 'obs_id'].astype(int)
    now = datetime.now()
    test_scored.to_csv('./output/test_scored_{}{}.csv'.format(now.strftime("%y%m%d%H%M%S"), output_note),
                       index=False)


def score_clusters(train_loc, test_loc, prediction_loc, prediction_colname, output_note):
    train = pd.read_csv(train_loc, dtype={'meter_id': 'object', 'site_id': 'object'})
    predictions = pd.read_csv(prediction_loc)
    test = pd.read_csv(test_loc)

    train['predictions'] = predictions[prediction_colname]

    train_clean = train.drop_duplicates(['timestamp', 'meter_id'])
    combined = pd.merge(train_clean, test, how='left', left_on=['timestamp', 'meter_id'],
                        right_on=['Timestamp', 'meter_id'])

    test_final = combined.dropna(subset=['is_abnormal'])

    test_final['is_abnormal'] = test_final.apply(lambda r: r['predictions'] == 0 or r['predictions'] == 4 or
                                                 r['predictions'] == 7 or r['predictions'] == 9, axis=1)

    print(test_final['is_abnormal'].value_counts())

    test_scored = test_final[['obs_id_y', 'meter_id', 'Timestamp', 'is_abnormal']]
    test_scored.columns = ['obs_id', 'meter_id', 'Timestamp', 'is_abnormal']
    test_scored['obs_id'] = test_scored.loc[:, 'obs_id'].astype(int)
    now = datetime.now()
    test_scored.to_csv('./output/test_scored_{}_{}.csv'.format(now.strftime("%y%m%d%H%M%S"), output_note),
                       index=False)


def score_novel_clusters(train_loc, test_loc, prediction_loc, cluster_loc, cluster_thresh, actual_colname,
                         prediction_colname, prediction_cutoff, output_note):
    train = pd.read_csv(train_loc, dtype={'meter_id': 'object', 'site_id': 'object'})
    predictions = pd.read_csv(prediction_loc)
    clusters = pd.read_csv(cluster_loc)
    test = pd.read_csv(test_loc)

    train['predictions'] = predictions[prediction_colname]

    train_clean = train.drop_duplicates(['timestamp', 'meter_id'])

    # Left join the test subset with the train superset
    combined = pd.merge(train_clean, test, how='left', left_on=['timestamp', 'meter_id'],
                        right_on=['Timestamp', 'meter_id'])

    combined = pd.merge(combined, clusters, how='left', 'left_on')

    # Calculate absolute error for each 'values' prediction
    combined['abs_err'] = combined.apply(lambda r: absolute_difference(r[actual_colname], r['predictions']), axis=1)
    # Calculate the average within meter_id group of absolute errors
    combined = pd.merge(combined, combined.groupby('meter_id')['abs_err'].mean().reset_index(name='average_abs_err'),
                        how='left', on='meter_id')
    # Calculate the anomaly score ratio of absolute error over the average absolute error (within meter_id group)
    combined['anomaly_score'] = combined['abs_err'] / combined['average_abs_err']

    # What should the cutoff be?? experiment with quantile cutoffs
    anomaly_cutoff = combined['anomaly_score'].quantile(q=prediction_cutoff)
    print('Anomaly cutoff: {}'.format(anomaly_cutoff))
    test_final = combined.dropna(subset=['is_abnormal'])
    test_final['is_abnormal'] = test_final.apply(lambda r: r['anomaly_score'] > anomaly_cutoff,
                                                 axis=1)

    print('Cluster threshold: {}'.format(cluster_thresh))
    test_final['is_abnormal']

    print(test_final.is_abnormal.value_counts())

    test_scored = test_final[['obs_id_y', 'meter_id', 'Timestamp', 'is_abnormal']]
    test_scored.columns = ['obs_id', 'meter_id', 'Timestamp', 'is_abnormal']
    test_scored['obs_id'] = test_scored.loc[:, 'obs_id'].astype(int)
    now = datetime.now()
    test_scored.to_csv('./output/test_scored_{}{}.csv'.format(now.strftime("%y%m%d%H%M%S"), output_note),
                       index=False)