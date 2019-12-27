# BTS - PyTorch
This repository contains the unofficial PyTorch implementation of From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation

[Paper](https://arxiv.org/abs/1907.10326)  
[Official Tensorflow Implementation](https://github.com/cogaplex-bts/bts)

The biggest difference between original implementation and this repository, aside from being PyTorch based that is, is that this implementation uses Numba library for Cuda accelerated implementation Local Planar Guidance layer. This saves us from bothersome setup process, but it also requires you to have a GPU with Cuda support.

This repository is tested on Windows and Ubuntu on PyTorch 1.2, 1.3 and 13.1, installed from pip and built from source.

Kitti Validation results:

| Model  | Silog | rmse | rmse log | abs relative | sqrt relative |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| BTS - PyTorch  | 10.80  | 3.08  | 0.11  | 0.07  | 0.32  |
| BTS - Official  | 9.16  | 2.79  | 0.09  | 0.06  | 0.25  |

As can be seen on table above, this implementation performs slightly worse than original implementation, which is very likely due some additional hyper-parameter tuning done by authors, due computational reasons I couldnt fine tune training parameters further.

## Videos

[![Screenshot](https://i.ytimg.com/vi/ekezJiGaiQk/hqdefault.jpg?sqp=-oaymwEZCNACELwBSFXyq4qpAwsIARUAAIhCGAFwAQ==&rs=AOn4CLC9PTHPb2ykWP6x6HnV7tGOxfTJrw)](https://youtu.be/ekezJiGaiQk)

## Setup
```
pip install -r requirements.txt
```
[Download pretrained model](https://drive.google.com/file/d/1OcY7-j7yp8yh47X-A9-ZF-ulxHSincFi/view?usp=sharing) and put it under models directory


*You need a GPU with CUDA support to run this implementation*


## Prediction

Please refer to prediction_example.ipynb

## Dataset Preperation
Kitti:
Preperation process is same as the official tensorflow implementation. But use "kitti_archives_to_download.txt" provided in this reposity which contains more runs.

## Evaluation
Change following lines at the start of the Test.py
```
model_path = "models/btspytorch"
dataset_path = "e://Code/Tez/bts_eren/kitti"
```
Run Test.py

## Training
Change following lines at the start of the Train.py
```
experiment_name = "Balatkan"  # This determines folder names used for saving tensorboard logs and model files
dataset_path = "e://Code/Tez/bts_eren/kitti"
```

Takes around 100 hours to train on GTX 1080.

