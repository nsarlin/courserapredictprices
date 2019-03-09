# -*- coding: utf-8 -*-
import click
import logging
import os

from sklearn.metrics import mean_squared_error

import scipy.sparse as sp
import numpy as np

import dnn
import xgb


@click.command()
@click.argument('data_dirpath', type=click.Path(exists=True))
@click.argument('model_dirpath', type=click.Path(exists=True))
@click.argument('preds_dirpath', type=click.Path(exists=True))
def main(data_dirpath, model_dirpath, preds_dirpath):
    """
    Make predictions with model from model_dirpath on data from data_dirpath
    and stores them in preds_dirpath
    """
    logger = logging.getLogger(__name__)

    logger.info("Loading data")
    X_test = sp.load_npz(os.path.join(data_dirpath, "X_test.npz"))
    y_test = np.load(os.path.join(data_dirpath, "y_test.npy"))

    logger.info("Making predictions")
    dnn_model = dnn.load(model_dirpath)
    y_preds = dnn.predict(dnn_model, X_test)
    np.save(os.path.join(data_dirpath, "y_preds_dnn.npy"), y_preds)
    print("DNN RMSE: {}".format(mean_squared_error(y_test, y_preds)))
    xgb_model = xgb.load(model_dirpath)
    y_preds = xgb.predict(xgb_model, X_test)
    np.save(os.path.join(data_dirpath, "y_preds_xgb.npy"), y_preds)
    print("XGB RMSE: {}".format(mean_squared_error(y_test, y_preds)))


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
