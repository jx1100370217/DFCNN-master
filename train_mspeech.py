#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
用于训练语音识别系统语音模型的程序

"""
import platform as plat
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import multi_gpu_model
#from AM import ModelSpeech
from SpeechModel_DFCNN import ModelSpeech


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
#进行配置，使用95%的GPU
config = tf.ConfigProto(allow_soft_placement=True)
#config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
#config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
set_session(tf.Session(config=config))

datapath = ''
modelpath = 'model_speech'

f = open('step_dfcnn.txt','r')
flist = f.readlines()
f.close()
fstr = "".join(flist)
base_count=fstr.split('_')[-1]

if(not os.path.exists(modelpath)): # 判断保存模型的目录是否存在
	os.makedirs(modelpath) # 如果不存在，就新建一个，避免之后保存模型的时候炸掉

system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
if(system_type == 'Windows'):
	datapath = 'E:\\语音数据集'
	modelpath = modelpath + '\\'
elif(system_type == 'Linux'):
	datapath = 'dataset'
	modelpath = modelpath + '/'
else:
	print('*[Message] Unknown System\n')
	datapath = 'dataset'
	modelpath = modelpath + '/'

ms = ModelSpeech(datapath)

#ms.LoadModel(modelpath + 'm_DFCNN/speech_model_DFCNN_e_0_step_84000.model')
ms.LoadModel(modelpath + 'm_dfcnn/speech_model_dfcnn_e_0_step_' + base_count +'.model')
ms.TrainModel(datapath, epoch = 50, batch_size = 64, save_step = 1000)


