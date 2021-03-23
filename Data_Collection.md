# Data Collection

## Authors

Andrew Chung - Northwestern University

## Introduction

This repository contains an exploration of three datasets that I was interested in analyzing for a personal ML project. The dataset that I ended up choosing was the **FashionIQ dataset** (Dataset 3). Details of the dataset are contained in this repository.

## Dataset 1: JHU-CROWD++

### Basic Information

[JHU-CROWD++: Large-Scale Crowd Counting Dataset and A Benchmark Method](https://arxiv.org/pdf/2004.03597.pdf)

_by Sindagi, Vishwanath A and Yasarla, Rajeev and Patel, Vishal M_

[Link to Dataset](https://drive.google.com/drive/folders/1-ikeDj7zJJkry1af_bFB_FzNcMMXxcUA?usp=sharing)

Total images: **4372**

- For a regular dataset, this would be fairly small, but this is actually on the larger side for crowd-counting datasets (which are limited in size due to annotation difficulty). There are in fact a total of **1,515,005 instances** in the entire dataset, which has a total size of **3.16GB**.

### Idea for Use Case

No one wants to go to a crowded place to study or work, especially with COVID going on as well. However, it’s hard to know exactly how crowded a certain coffee shop or library might be as of current. One idea is to place a camera for live streaming the current situation of the place in question — however, exposing the faces of people in the video to random onlookers would be a blatant violation of personal privacy.

Instead, an ML model can count the number of human heads present in the video. This number can then be exposed (along with the maximum capacity) to users who would benefit from knowing how many people are there.

### Dataset Description

The dataset description has been adapted from the README.txt file in the original dataset directory.

#### Directory Info

1. The dataset directory contains 3 sub-directories: **train, val and test**. In other words, the train-test split has already been done.

2. Each of these contain 2 sub-directories (**images, gt**) and a file "image_labels.txt". The "images" directory contains images and the "gt" directory contains **ground-truth files** corresponding to the images in the images directory. (Further description about ground-truth labels can be found below).

3. The number of image samples in train, val and test split are **2272, 500, 1600** respectively.

#### Ground-Truth Annotations: Head-Level

1. Each ground-truth file in the "gt" directory contains space-separated values with each row indicating _x y w h o b_.

2. _x,y_ indicate the location of each head in the image.

3. _w,h_ indicate approximate width and height of the head.

4. _o_ indicates occlusion-level (i.e., the extent to which the head is hidden from view) and it can take 3 possible values: 1,2,3.
   - o=1 indicates "visible"
   - o=2 indicates "partial-occlusion"
   - o=3 indicates "full-occlusion"
5. _b_ indicates blur-level and it can take 2 possible values: 0,1.
   - b=0 indicates no-blur
   - b=1 indicates blur

#### Ground-Truth Annotations: Image-Level

1. Each split in the dataset (train, val, test) contains a file "image_labels.txt". This file contains **image-level labels**.

2. The values in the file are **comma-separated** and each row indicates: _filename, total-count, scene-type, weather-condition, distractor_
3. _total-count_ indicates the total number of people in the image

4. _scene-type_ is an image-level label describing the scene (e.g., running, mall, stadium)

5. _weather-condition_ indicates the weather-degradation in the image and can take 4 values: 0,1,2,3
   - weather-condition=0 indicates "no weather degradation"
   - weather-condition=1 indicates "fog/haze"
   - weather-condition=2 indicates "rain"
   - weather-condition=3 indicates "snow"
6. _distractor_ indicates if the image is a distractor. (A distractor is an image with many counts of another object other than people, such as birds or kites.) It can take 2 values: 0, 1
   - distractor=0 indicates "not a distractor"
   - distractor=1 indicates "distractor"

## Dataset 2: RWF-2000

### Basic Information

[RWF-2000: An Open Large Scale Video Database for Violence Detection](https://arxiv.org/pdf/1911.05913.pdf)

_by Ming Cheng, Kunjing Cai, and Ming Li_

The original GitHub repo for processing the dataset [can be found here](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection). Redistribution of the dataset is not allowed according to the creators, which is why it has not been linked here. However, the structure of the data is described below.

Total videos: **2000**

- There are 2000 videos at 30fps that are all within 5 seconds, which translates to around 300,000 distinct frames. The entire dataset is **12.71GB** in size.

### Idea for Use Case

One of the biggest societal problems in my home country of South Korea is child abuse. There have been shocking videos of nursery school teachers hitting kids to the point where they aren’t even able to make a sound. However, while this kind of footage exists, others are unable to notice the problem earlier because no one thinks of these teachers as being suspicious to begin with. The goal is to make surveillance cameras “smart” — in other words, they will be an eyewitness in the moment, and the software will report to higher authorities whenever the cameras witness violence. While it might be hard to tune the model to just instances of child abuse, there is a database of violence footage (RWF-2000) as well as previous studies that can be referred to. This can be integrated into a software application that allows anyone to connect a camera’s live footage and have it be analyzed by the model, so that a notification is sent whenever violent action is detected.

### Dataset Description

For the dataset itself, it is separated into two subdirectories: **train, val**. The training set consists of 1600 videos, divided into two equal subdirectories labeled **Fight, NonFight**. The test set consists of 400 surveillance videos with the same directory structure (with a 50/50 split between Fight and NonFight videos). There is no separate validation set (which should be separated from the train set instead).

Some clips in the dataset may have one of the following characteristics, which are potential problems that should be addressed in model implementation:

- Only part of the person appears in the frame
- Violence amongst crowds and chaos
- Small object at a far distance
- Low resolution
- Transient (abrupt and quick) action

Additionally, there are some notes to make about the original GitHub repository (linked above). The description below has been adapted from the repository's README:

1. **Preprocess** contains the python script to transform original video dataset to .npy files. Each .npy file is a tensor with shape = **[nb_frames, img_height, img_width, 5]**. The last channel contains 3 layers for RGB components and 2 layers for optical flows (vertical and horizontal components, respectively ).

2. **Networks** contain the keras implemention of the model proposed by the original authors of this dataset. Also, the training scripts of single stream are provided here.

3. **Models** contains the pre-trained model implemented by Keras.

## Dataset 3: FashionIQ

### Basic Information

[The Fashion IQ Dataset: Retrieving Images by Combining Side Information and Relative Natural Language Feedback](https://arxiv.org/pdf/1905.12794.pdf)

_by Guo, Xiaoxiao and Wu, Hui and Gao, Yupeng and Rennie, Steven and Feris, Rogerio_

The original GitHub repo with starter code [can be found here](https://github.com/XiaoxiaoGuo/fashion-iq). The image metadata [can be found here](https://github.com/hongwang600/fashion-iq-metadata). All necessary info for the FashionIQ dataset is contained within this repository.

Based on the counts determined by running `count_samples.py`, the dataset size is as follows:

| segment | train | val  | test |
| ------- | ----- | ---- | ---- |
| dress   | 11452 | 3817 | 3818 |
| shirt   | 19036 | 6346 | 6346 |
| toptee  | 16121 | 4373 | 4374 |

### Idea for Use Case

(See competition and papers from CVPR 2020.) Using FashionIQ, a dataset at the intersection of NLP and CV, it could potentially be made easier for someone to find a piece of clothing that they are interested in. Specifically, the goal would be to create a chatbot that, given a textual input from a user, can return an image similar to the description given. This is a task that has already been done as part of CVPR 2020 (see [this link](https://sites.google.com/view/cvcreative2020/fashion-iq?authuser=0) for more info on FashionIQ at CVPR 2020).

### Dataset Description

The dataset consists of three unique categories of clothing: women's dresses, women's tops, and men's shirts. The actual images are contained in the `image_url` subdirectory, where there are three .txt files that correspond to the three types of clothing (each named in the format `asin2url.CATEGORY.txt`).

In the .txt files, each line represents a distinct image, with the unique ID of the image followed by 3 spaces and then the URL address of the image. The `broken_links` subdirectory in `image_url` contains jpg files replacing the broken links in the .txt files.

The `image_splits` subdirectory has 9 JSON files named in the pattern `split.CATEGORY.SPLITTYPE.json`. Each file is a list of image IDs which fall into that particular category (shirt, dress, toptee) and split (train, val, test).

The `captions` subdirectory consists of multiple JSON files named in the pattern `cap.CATEGORY.SPLITTYPE.json`. Each is a list of objects with the following properties:

- _captions_: A list of two **relative captions** between the candidate and target image. A relative caption is a caption that describes how the target image differs from the candidate image. For example, if the candidate image and target image are both black but the target image is longer and has no sleeves, then a potential "relative caption" would be "black but longer and with more sleeves."
- _candidate_: The ID of the 'initial' image to start off with.
- _target_: The ID of the 'subsequent' image which the relative captions describe in relation to the candidate image.

The target IDs of the captions in the test JSON files have not been released in this repository due to potential privacy issues.
