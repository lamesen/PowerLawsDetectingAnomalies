import power_utils
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML


def convert_columns_as_factor(hdf, column_list):
    list_count = len(column_list)
    if list_count is 0:
        return "Error: You don't have a list of binary columns."
    if (len(hdf.columns)) is 0:
        return "Error: You don't have any columns in your data frame."
    for i in range(list_count):
        try:
            hdf[column_list[i]] = hdf[column_list[i]].asfactor()
            print('Column ' + column_list[i] + " is converted into factor/catagorical.")
        except ValueError:
            print('Error: ' + str(column_list[i]) + " not found in the data frame.")


if __name__ == "__main__":
    train = pd.read_csv('./tmp/train_prepared.csv')

    # Initialize h2o cluster
    h2o.init()

    # Drop un-needed ID variables and target variable transformations
    train_subset = train.drop(['obs_id', 'timestamp'], axis=1)

    # Create an H2O data frame (HDF) from the pandas data frame
    h2o_train = h2o.H2OFrame(train_subset)

    # User-defined function to convert columns to HDF factors
    convert_columns_as_factor(h2o_train, ['site_id', 'meter_description', 'units', 'activity', 'holiday',
                                          'hour', 'minute', 'dow', 'wom', 'year', 'month', 'day',
                                          'business_hours', 'evening', 'overnight'])

    # Setup Auto ML to run for approximately 10 hours
    aml = H2OAutoML(max_runtime_secs=36000)

    # Train Auto ML
    aml.train(y="values",
              training_frame=h2o_train)

    # Save results
    model_path = h2o.save_model(model=aml.leader, path="./tmp/my_model", force=True)

    # Output leaderboard
    lb = aml.leaderboard

    preds = aml.predict(h2o_train)

    train = pd.read_csv('./tmp/train_prepared.csv')
    train['predictions'] = preds

    # Save original train/test data
    train.to_csv('./tmp/train_out.csv')
