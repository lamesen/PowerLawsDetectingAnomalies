import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit


def load_train():
    print('Reading file')
    train = pd.read_csv('./tmp/train_prepared.csv')
    remove = ['obs_id', 'timestamp', 'values']
    predictors = [x for x in train.columns if x not in remove]

    X = train[predictors]
    y = train['values']

    return X, y


if __name__ == "__main__":
    X, y = load_train()

    xgb_model = xgb.XGBClassifier()

    print('Training model')
    cv = 5

    parameters = {'nthread': [1],
                  'objective': ['reg:linear'],
                  'learning_rate': [0.05],
                  'max_depth': [6],
                  'min_child_weight': [11],
                  'silent': [1],
                  'subsample': [0.8],
                  'colsample_bytree': [0.7],
                  'n_estimators': [1000],
                  'missing': [-1],
                  'seed': [1337]}

    clf = GridSearchCV(xgb_model, parameters, n_jobs=5,
                       cv=TimeSeriesSplit(n_splits=cv).get_n_splits([X, y]),
                       scoring='neg_mean_squared_error',
                       verbose=2, refit=True)

    clf.fit(X, y)

    best_parameters, score, _ = max(clf.cv_results_, key=lambda x: x[1])
    print('Raw RMSE score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    predictions = clf.predict(X)[:, 1]

    print('Applying predictions')
    train = pd.read_csv('./tmp/train_prepared.csv')
    train['predictions'] = predictions
    train.to_csv('./tmp/train_scored.csv')
