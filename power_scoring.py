import pandas as pd
import power_utils

if __name__ == "__main__":
    train_loc = './tmp/train_prepared.csv'
    test_loc = './input/submission_format.csv'
    prediction_loc = './output/data_robot_gbm_diff_predictions.csv'
    actual_colname = 'values_day_diff_7'
    prediction_colname = 'Prediction'
    prediction_cutoff = 0.50
    output_note = 'gbm_diff_7'

    power_utils.score_predictions(train_loc, test_loc, prediction_loc, actual_colname, prediction_colname,
                                  prediction_cutoff, output_note)
