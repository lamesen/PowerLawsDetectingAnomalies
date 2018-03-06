import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def load_train(columns):
    print('Reading file')
    train = pd.read_csv('./tmp/train_prepared.csv', names=columns, skiprows=1)
    remove = ['obs_id', 'timestamp', 'values']
    predictors = [x for x in columns if x not in remove]
    target = ['values']

    dtrain = xgb.DMatrix(train.ix[:, predictors], train[2])
    dtrain.save_binary('./tmp/train.dmatrix')

    return dtrain

if __name__ == "__main__":

    columns = ['obs_id',
               'timestamp',
               'values',
               'holiday',
               'dow',
               'wom',
               'year',
               'month',
               'day',
               'hour',
               'minute',
               'site_id_038',
               'site_id_234_203',
               'site_id_334_61',
               'meter_description_Lighting',
               'meter_description_RTE_meter',
               'meter_description_RTE_meter_cos_phi',
               'meter_description_RTE_meter_demand',
               'meter_description_RTE_meter_reactive',
               'meter_description_Test_Bay',
               'meter_description_cold_group',
               'meter_description_compressed_air',
               'meter_description_cuisine',
               'meter_description_elevators',
               'meter_description_generator',
               'meter_description_guardhouse',
               'meter_description_heating',
               'meter_description_laboratory',
               'meter_description_lighting',
               'meter_description_main_meter',
               'meter_description_main_meter__demand',
               'meter_description_main_meter_cos_phi',
               'meter_description_main_meter_reactive_energy',
               'meter_description_other',
               'meter_description_outside_temperature',
               'meter_description_temperature',
               'meter_description_total_workers',
               'meter_description_virtual_main',
               'meter_description_virtual_meter',
               'units_count',
               'units_degree_celsius',
               'units_kWh',
               'activity_general',
               'activity_laboratory',
               'activity_office',
               'activity_restaurant',
               'surface',
               'temperature',
               'distance']

    print('Training model')
    dtrain = load_train(columns)

    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'gpu:reg:linear'}
    param['gpu_id'] = 0
    param['max_bin'] = 16
    param['tree_method'] = 'gpu_hist'

    num_round = 2

    bst = xgb.train(param, dtrain, num_round)
    predictions = bst.predict(dtrain)

    train = pd.read_csv('./tmp/train_prepared.csv', names=columns, skiprows=1)

    print('Applying predictions')
    train['predictions'] = predictions
    train.to_csv('./tmp/train_scored.csv')
