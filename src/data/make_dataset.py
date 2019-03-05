# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import os
from dotenv import find_dotenv, load_dotenv
import pandas as pd

import prepare


@click.command()
@click.argument('input_dirpath', type=click.Path(exists=True))
@click.argument('output_dirpath', type=click.Path(exists=True))
@click.option('-i', '--interim-dirpath', type=click.Path(exists=True),
              help="Directory to save/restore intermediate data.")
@click.option('--sub/--val', default=False,
              help="Build dataset for validation or submission")
@click.option('--sample/--full', default=False,
              help="Prepare only a sample of data for quick tests")
def main(input_dirpath, output_dirpath, interim_dirpath, sample, sub):
    """ Runs data processing scripts to turn raw data from input_dirpath into
        cleaned data ready to be analyzed (stored in output_dirpath).
    """
    logger = logging.getLogger(__name__)
    if sub:
        logger.info('making final data set from raw data for submission')
    else:
        logger.info('making final data set from raw data for validation')
    if interim_dirpath:
        logger.info('using intermediate data from {}'.format(interim_dirpath))
        store = pd.HDFStore(os.path.join(interim_dirpath, "save.h5"))

    prepare.prepare_all(input_dirpath, output_dirpath, not sub, sample, store)
    logger.info('done')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
