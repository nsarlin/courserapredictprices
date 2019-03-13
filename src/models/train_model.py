# -*- coding: utf-8 -*-
import click
import logging
import os

import scipy.sparse as sp
import numpy as np


@click.command()
@click.argument('model', type=click.STRING)
@click.argument('data_dirpath', type=click.Path(exists=True))
@click.argument('model_dirpath', type=click.Path(exists=True))
def main(model, data_dirpath, model_dirpath):
    """
    Trains model from data found in data_dirpath, and stores it in
    model_dirpath.
    """
    logger = logging.getLogger(__name__)

    logger.info("Loading data")
    X_train = sp.load_npz(os.path.join(data_dirpath, "X_train.npz")).tocsr()
    y_train = np.load(os.path.join(data_dirpath, "y_train.npy"))

    if model == "dnn":
        import dnn
        logger.info("Training DNN")
        dnn_model = dnn.train(X_train, y_train)
        dnn.save(dnn_model, model_dirpath)
        logger.info("DNN train done")
    elif model == "xgb":
        import xgb
        logger.info("Training XGB")
        xgb_model = xgb.train(X_train, y_train)
        xgb.save(xgb_model, model_dirpath)
        logger.info("XGB train done")
    else:
        logger.error("Invalid model: {}".format(model))


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
