import xgboost as xgb
import joblib
import os

from src import common

xgb_params = {'eta': 0.1, 'seed': common.RANDOM_SEED, 'subsample': 1, 'colsample_bytree': 0.7,
              'objective': 'reg:linear', 'max_depth': 13,
              'min_child_weight': 20, 'eval_metric': 'rmse', 'silent': 1}


def train(X_train, y_train):
    X_train_xgb = xgb.DMatrix(X_train, label=y_train)
    bst = xgb.train(xgb_params, X_train_xgb)

    return bst


def save(bst, model_dirpath):
    joblib.dump(bst, os.path.join(model_dirpath, "xgb.bin"))


def load(model_dirpath):
    return joblib.load(os.path.join(model_dirpath, "xgb.bin"))


def predict(bst, X_test):
    X_test_xgb = xgb.DMatrix(X_test)
    return bst.predict(X_test_xgb)
