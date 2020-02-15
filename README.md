# BTS - PyTorch
This repository contains the unofficial PyTorch implementation of From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation

[Paper](https://arxiv.org/abs/1907.10326)  
[Official Tensorflow Implementation](https://github.com/cogaplex-bts/bts)

This repository is tested on Windows and Ubuntu on PyTorch 1.2, 1.3 and 13.1, installed from pip and built from source.

Kitti Validation results:

| Model           | Silog  | rmse  | rmse log | abs relative | sqrt relative |
| --------------- | ------ | ----- | -------- | ------------ | ------------- |
| BTS - PyTorch   | 9.83  | 3.03  |   0.10   |     0.06     |      0.29     |
| BTS - Official  | 9.08    | 2.82  |   0.09   |     0.05     |      0.26     |

As can be seen on table above, this implementation performs slightly worse than original implementation, which is very likely due some additional hyper-parameter tuning done by authors, due computational reasons I couldnt fine tune training parameters further.

## Videos

[![Screenshot](https://i.ytimg.com/vi/ekezJiGaiQk/hqdefault.jpg?sqp=-oaymwEZCNACELwBSFXyq4qpAwsIARUAAIhCGAFwAQ==&rs=AOn4CLC9PTHPb2ykWP6x6HnV7tGOxfTJrw)](https://youtu.be/ekezJiGaiQk)

## Setup
```
pip install -r requirements.txt
```
[Download pretrained model](https://drive.google.com/file/d/1_mENn0G9YlLAAr3N8DVDt4Hk2SBbo1pl/view?usp=sharing) and put it under models directory

## Prediction

Please refer to prediction_example.ipynb

## Dataset Preperation
Kitti:
Preperation process is same as the official tensorflow implementation. But use "kitti_archives_to_download.txt" provided in this reposity which contains more runs.

## Evaluation
Change following lines at the start of the configs.py
```
model_path = "models/btspytorch"
dataset_path = "e://Code/Tez/bts_eren/kitti"
```
Run Test.py

## Training
Change following lines at the start of the configs.py
```
experiment_name = "Balatkan"  # This determines folder names used for saving tensorboard logs and model files
dataset_path = "e://Code/Tez/bts_eren/kitti"
```

Takes around 100 hours to train on GTX 1080.