# A Full-Convolutional Neural Networks-Based Chinese Speech Recognition System
基于全卷积神经网络的中文语音识别系统

[![GPL-3.0 Licensed](https://img.shields.io/badge/License-GPL3.0-blue.svg?style=flat)](https://opensource.org/licenses/GPL-3.0) [![TensorFlow Version](https://img.shields.io/badge/Tensorflow-1.4+-blue.svg)](https://www.tensorflow.org/) [![Keras Version](https://img.shields.io/badge/Keras-2.0+-blue.svg)](https://keras.io/) [![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/) 

## Introduction 简介

本项目使用TensorFlow基于全深度卷积神经网络实现。
通过git克隆仓库以后，需要将datalist目录下的文件全部拷贝到dataset目录下，也就是将其跟数据集放在一起。
```shell
$ cp -rf datalist/* dataset/
```

目前可用的模型有24、25和251

本项目开始训练请执行：
```shell
$ python3 train_mspeech.py
```
本项目开始测试请执行：
```shell
$ python3 test_mspeech.py
```
测试之前，请确保代码中填写的模型文件路径存在。

ASRT API服务器启动请执行：
```shell
$ python3 asrserver.py
```

## Model 模型

### Speech Model 语音模型

CNN + LSTM/GRU + CTC

### Language Model 语言模型

基于概率图的最大熵隐马尔可夫模型

## About Accuracy 关于准确率

当前，最好的模型在测试集上基本能达到80%的汉语拼音正确率

不过由于目前国际和国内的部分团队能做到97%，所以正确率仍有待于进一步提高

## Python Import
Python的依赖库

* python_speech_features
* TensorFlow
* Keras
* Numpy
* wave
* matplotlib
* math
* Scipy
* h5py

## Data Sets 数据集
* 清华大学THCHS30中文语音数据集

data_thchs30.tgz 
<http://cn-mirror.openslr.org/resources/18/data_thchs30.tgz>
<http://www.openslr.org/resources/18/data_thchs30.tgz>

test-noise.tgz 
<http://cn-mirror.openslr.org/resources/18/test-noise.tgz>
<http://www.openslr.org/resources/18/test-noise.tgz>

resource.tgz 
<http://cn-mirror.openslr.org/resources/18/resource.tgz>
<http://www.openslr.org/resources/18/resource.tgz>

* Free ST Chinese Mandarin Corpus

ST-CMDS-20170001_1-OS.tar.gz 
<http://cn-mirror.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz>
<http://www.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz>
