import json
import os
from pathlib import Path
from shutil import copyfile
import urllib.request

import dask.dataframe as dd
import pandas as pd

from src.utils import get_project_dir


def url_to_jpg(row):
    """
    Saves images based on the split (train/test/val) and image type
    (dress/shirt/toptee). Operates on a DataFrame row.
    """
    print(row)
    project_dir = get_project_dir()
    img_type_dir = f"{project_dir}/data/raw/{row['img_type']}"
    split_dir = Path(f"{img_type_dir}/{row['split_type']}")

    # create directory + missing parents. Is OK if directory already exists
    split_dir.mkdir(parents=True, exist_ok=True)
    img_path = f"{split_dir}/{row['id']}.jpg"

    try:
        urllib.request.urlretrieve(row['url'], img_path)

    # copy image from /broken_links folder
    except urllib.error.HTTPError:
        broken_link = f"{project_dir}/data/exernal/image_url/broken_links/{row['id']}.jpg"
        if os.path.exists(broken_link):
            copyfile(broken_link, img_path)
        else:
            missing_img_csv = f"{img_type_dir}/missing_imgs.csv"
            mode = 'a' if os.path.exists(missing_img_csv) else 'w'
            row.to_frame().T.to_csv(missing_img_csv, mode=mode)

    return row


def read_json_split(base_dir, file):
    """
    Reads a JSON split of image ids (an array) and returns a corresponding
    DataFrame with columns (id, split_type, img_type).
        - `split_type` will be one of 'train', 'test', or 'val'
        - `img_type` will be one of 'dress', 'shirt', or 'toptee'.
    """
    _, img_type, split_type, _ = file.split('.')
    split_df = pd.read_json(base_dir + file)
    split_df.columns = ['id']
    split_df['id'] = split_df['id'].str.strip()
    split_df['split_type'] = split_type
    split_df['img_type'] = img_type
    return split_df


def read_splits(img_type):
    """
    Get a DataFrame of image ids and the corresponding split (train, test or val)
    for a certain type of image (dress, shirt or toptee).
    """
    if img_type not in ('dress', 'shirt', 'toptee'):
        raise ValueError('Argument must be one of (dress, shirt, toptee)')

    project_dir = str(get_project_dir())
    img_splits_dir = f'{project_dir}/data/external/image_splits/'
    dfs = []
    for file in os.listdir(img_splits_dir):
        if img_type not in file:
            continue
        dfs.append(read_json_split(img_splits_dir, file))

    return pd.concat(dfs).reset_index(drop=True)


def read_img_urls(img_type):
    """
    Get a DataFrame of image ids and the corresponding URL for a certain type
    of image (dress, shirt or toptee).
    """
    if img_type not in ('dress', 'shirt', 'toptee'):
        raise ValueError('Argument must be one of (dress, shirt, toptee)')

    project_dir = str(get_project_dir())
    img_urls_dir = f'{project_dir}/data/external/image_url'
    df = pd.read_csv(f'{img_urls_dir}/asin2url.{img_type}.txt', delimiter='\t', header=None)
    df.columns = ['id', 'url']
    df['id'] = df['id'].str.strip()
    df['url'] = df['url'].str.strip()

    return df


def download_raw_imgs(img_type, npartitions=10):
    """
    Download the raw images for a given image type to sub-folders in `data/raw`
    based on the train/test/val splits.
    """
    split_df = read_splits(img_type)
    url_df = read_img_urls(img_type)
    combined_df = url_df.merge(split_df, on='id')

    # ensure no images have been left out
    assert len(split_df) == len(url_df) == len(combined_df)

    # download only images that haven't been downloaded
    img_type_dir = f'{get_project_dir()}/data/raw/{img_type}'
    exists = combined_df.apply(lambda row:
                               os.path.exists(f"{img_type_dir}/{row['split_type']}/{row['id']}.jpg"), axis=1)

    missing_img_csv = f"{img_type_dir}/missing_imgs.csv"
    missing_df = pd.read_csv(missing_img_csv) if os.path.exists(missing_img_csv) else pd.DataFrame({'id': []})
    remaining_df = combined_df[(~exists) & (~(combined_df['id'].isin(missing_df['id'])))]

    # download raw images in parallel for given image type
    ddata = dd.from_pandas(remaining_df, npartitions=npartitions)
    res = ddata.map_partitions(lambda df: df.apply(url_to_jpg, axis=1),
                               meta=pd.DataFrame(columns=['id', 'img_type', 'split_type', 'url'])).compute(scheduler='threads')
