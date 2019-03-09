# -*- coding: utf-8 -*-
import click
import logging
import os

import scipy.sparse as sp
import numpy as np

import dnn
import xgb


@click.command()
@click.argument('data_dirpath', type=click.Path(exists=True))
@click.argument('model_dirpath', type=click.Path(exists=True))
def main(data_dirpath, model_dirpath):
    """
    Trains model from data found in data_dirpath, and stores it in
    model_dirpath.
    """
    logger = logging.getLogger(__name__)

    logger.info("Loading data")
    X_train = sp.load_npz(os.path.join(data_dirpath, "X_train.npz")).tocsr()
    y_train = np.load(os.path.join(data_dirpath, "y_train.npy"))

    logger.info("Training DNN")
    dnn_model = dnn.train(X_train, y_train)
    dnn.save(dnn_model, model_dirpath)
    logger.info("Training XGB")
    xgb_model = xgb.train(X_train, y_train)
    xgb.save(xgb_model, model_dirpath)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
