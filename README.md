# BTS - PyTorch
This repository contains the unofficial PyTorch implementation of From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation

[Paper](https://arxiv.org/abs/1907.10326)  
[Official Tensorflow Implementation](https://github.com/cogaplex-bts/bts)

The biggest difference between original implementation and this repository, aside from being PyTorch based that is, is that this implementation uses Numba library for Cuda accelerated implementation Local Planar Guidance layer. This saves us from bothersome setup process, but it also requires you to have a GPU with Cuda support.

This repository is tested on Windows and Ubuntu on PyTorch 1.2, 1.3 and 13.1, installed from pip and built from source.

Validation results:

## Setup
```
pip install -r requirements.txt
```


## Prediction

Please refer to prediction_example.ipynb

## Dataset Preperation
Coming very soon!

## Evaluation
Coming very soon!

## Training
Coming very soon!

