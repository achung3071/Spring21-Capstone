# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data import download_raw_imgs
from src.utils import get_project_dir


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('img_type')
def main(input_filepath, output_filepath, img_type):
    """ 
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    project_dir = get_project_dir()
    cwd = os.getcwd()
    full_input_path = f'{cwd}/{input_filepath}'
    full_output_path = f'{cwd}/{output_filepath}'

    if img_type not in ('dress', 'shirt', 'toptee'):
        raise ValueError('Image type must be one of (dress, shirt, toptee)')

    if (full_input_path == f'{project_dir}/data/external' and
            full_output_path == f'{project_dir}/data/raw'):
        logger = logging.getLogger(__name__)
        logger.info('Downloading raw dataset from external URLs...')
        download_raw_imgs(img_type)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
