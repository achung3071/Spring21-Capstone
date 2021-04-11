# Script for counting the number of images in each split.
# Output shown in `data/raw/sample_counts.json` from the root directory.

import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
IMAGE_SPLITS_DIR = f'{ROOT_DIR}/data/external/image_splits'
sample_counts = {}

for file in os.listdir(IMAGE_SPLITS_DIR):
    with open(f'{IMAGE_SPLITS_DIR}/{file}', 'r') as read_file:
        SEGMENT = '.'.join(file.split('.')[1:3])
        data = json.load(read_file)
        sample_counts[SEGMENT] = len(data)

sample_counts = dict(sorted(sample_counts.items()))
with open(f'{ROOT_DIR}/data/external/sample_counts.json', 'w') as out_file:
    json.dump(sample_counts, out_file)
