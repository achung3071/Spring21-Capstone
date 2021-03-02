import json
import os

IMAGE_SPLITS_DIR = './data/image_splits'
sample_counts = {}

for file in os.listdir(IMAGE_SPLITS_DIR):
    with open(f'{IMAGE_SPLITS_DIR}/{file}', 'r') as read_file:
        SEGMENT = '.'.join(file.split('.')[1:3])
        data = json.load(read_file)
        sample_counts[SEGMENT] = len(data)

sample_counts = dict(sorted(sample_counts.items()))
with open('sample_counts.json', 'w') as out_file:
    json.dump(sample_counts, out_file)
