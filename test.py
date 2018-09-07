
# -*- coding: utf-8 -*-
"""
@author: nl8590687
用于测试整个一套语音识别系统的程序
语音模型 + 语言模型
"""
import platform as plat
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from SpeechModel_DFCNN import ModelSpeech
#from SpeechModel_old import ModelSpeech
from LanguageModel import ModelLanguage
import sys

EPOCH = sys.argv[1]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
#进行配置，使用70%的GPU
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
set_session(tf.Session(config=config))

datapath = ''
modelpath = 'model_speech'

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

ms.LoadModel(modelpath + 'm_dfcnn/speech_model_dfcnn_e_0_step_' + EPOCH + '000.model')
#ms.LoadModel(modelpath + 'm_DFCNN/speech_model_DFCNN_e_0_step_8000.model')
#ms.LoadModel(modelpath + 'm26/speech_model26_e_0_step_122500.model')

#ms.TestModel(datapath, str_dataset='test', data_count = 64, out_report = True)
#r = ms.RecognizeSpeech_FromFile('myspeech.wav')
#r = ms.RecognizeSpeech_FromFile('/home/speech.AI/github/DFCNN/dataset/data_thchs30/test/D11_750.wav')
#aishell
r = ms.RecognizeSpeech_FromFile('/home/speech.AI/github/DFCNN/dataset/data_aishell/wav/test/S0764/BAC009S0764W0122.wav')
#r = ms.RecognizeSpeech_FromFile('/home/speech.AI/github/DFCNN/dataset/data_aishell/wav/test/S0764/BAC009S0764W0124.wav')
#r = ms.RecognizeSpeech_FromFile('/home/speech.AI/github/DFCNN/dataset/data_aishell/wav/test/S0764/BAC009S0764W0179.wav')
#r = ms.RecognizeSpeech_FromFile('/home/speech.AI/github/DFCNN/dataset/data_aishell/wav/test/S0764/BAC009S0764W0226.wav')
#ST_CMDS
#r = ms.RecognizeSpeech_FromFile('/home/speech.AI/data/ST-CMDS-20170001_1-OS/20170001P00215A0075.wav')
#r = ms.RecognizeSpeech_FromFile('/home/speech.AI/data/ST-CMDS-20170001_1-OS/20170001P00091A0063.wav')
#r = ms.RecognizeSpeech_FromFile('/home/speech.AI/data/ST-CMDS-20170001_1-OS/20170001P00272A0101.wav')
#r = ms.RecognizeSpeech_FromFile('/home/speech.AI/data/ST-CMDS-20170001_1-OS/20170001P00087A0045.wav')
#r = ms.RecognizeSpeech_FromFile('/home/speech.AI/github/DFCNN/dataset/data_thchs30/train/A33_100.wav')
print('*[提示] 语音识别结果：\n',r)


ml = ModelLanguage('model_language')
ml.LoadModel()

#str_pinyin = ['zhe4','zhen1','shi4','ji2', 'hao3','de5']
#str_pinyin = ['jin1', 'tian1', 'shi4', 'xing1', 'qi1', 'san1']
#str_pinyin = ['ni3', 'hao3','a1']
str_pinyin = r
#str_pinyin =  ['su1', 'bei3', 'jun1', 'de5', 'yi4','xie1', 'ai4', 'guo2', 'jiang4', 'shi4', 'ma3', 'zhan4', 'shan1', 'ming2', 'yi1', 'dong4', 'ta1', 'ju4', 'su1', 'bi3', 'ai4', 'dan4', 'tian2','mei2', 'bai3', 'ye3', 'fei1', 'qi3', 'kan4', 'zhan4']
r = ml.SpeechToText(str_pinyin)
print('语音转文字结果：\n',r)














