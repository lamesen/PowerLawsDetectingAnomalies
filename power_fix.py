import pandas as pd
import numpy as np



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

    print('Reading file')
    train = pd.read_csv('./tmp/train_prepared.csv', names=columns, skiprows=1)
    print('Writing file')
    train.to_csv('./tmp/train_prepared_fixed.csv', index=False)
    print('Complete.')

