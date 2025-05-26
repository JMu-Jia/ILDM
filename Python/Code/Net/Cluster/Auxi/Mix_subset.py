#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/11/22 9:05
# @Author : JMu
# @ProjectName : 用于将多个数据集导入LSM进行训练

import os
import torch

# ========================subset属性========================
class_dict = {  # 所有subset的class属性
    'AbnormalHeartbeat': 5,
    'ACSF1': 10,
    'ArticularyWordRecognition': 25,
    'BasicMotions': 4,
    'Cricket': 12,
    'Earthquakes': 2,
    'ElectricDevices': 7,
    'ERing': 6,
    'EthanolConcentration': 4,
    'FaceDetection': 2,
    'FaultDetectionA': 3,
    'FordA': 2,
    'FordB': 2,
    'HandMovementDirection': 4,
    'Handwriting': 26,
    'Heartbeat': 2,
    'JapaneseVowels': 9,
    'KeplerLightCurves': 7,
    'LSST': 14,
    'MotionSenseHAR': 6,
    'MotorImagery': 2,
    'NerveDamage': 3,
    'PhonemeSpectra': 39,
    'RacketSports': 4,
    'SelfRegulationSCP1': 2,
    'SelfRegulationSCP2': 2,
    'SpokenArabicDigits': 10,
    'Trace': 4,
    'UWaveGestureLibrary': 8,
    'Wafer': 2,
    'WalkingSittingStanding': 6,
}

len_dict = {  # 所有subset的length属性
    'AbnormalHeartbeat': 3053,
    'ACSF1': 1460,
    'ArticularyWordRecognition': 144,
    'BasicMotions': 100,
    'Cricket': 1197,
    'Earthquakes': 512,
    'ElectricDevices': 96,
    'ERing': 65,
    'EthanolConcentration': 1751,
    'FaceDetection': 62,
    'FaultDetectionA': 5120,
    'FordA': 500,
    'FordB': 500,
    'HandMovementDirection': 400,
    'Handwriting': 152,
    'Heartbeat': 405,
    'JapaneseVowels': 29,
    'KeplerLightCurves': 4767,
    'LSST': 36,
    'MotionSenseHAR': 200,
    'MotorImagery': 3000,
    'NerveDamage': 1500,
    'PhonemeSpectra': 217,
    'RacketSports': 30,
    'SelfRegulationSCP1': 896,
    'SelfRegulationSCP2': 1152,
    'SpokenArabicDigits': 93,
    'Trace': 275,
    'UWaveGestureLibrary': 315,
    'Wafer': 152,
    'WalkingSittingStanding': 206,
}

chan_dict = {  # 所有subset的feature_dimensions属性
    'AbnormalHeartbeat': 1,
    'ACSF1': 1,
    'ArticularyWordRecognition': 9,
    'BasicMotions': 6,
    'Cricket': 6,
    'Earthquakes': 1,
    'ElectricDevices': 1,
    'ERing': 4,
    'EthanolConcentration': 3,
    'FaceDetection': 144,
    'FaultDetectionA': 1,
    'FordA': 1,
    'FordB': 1,
    'HandMovementDirection': 10,
    'Handwriting': 3,
    'Heartbeat': 61,
    'JapaneseVowels': 12,
    'KeplerLightCurves': 1,
    'LSST': 6,
    'MotionSenseHAR': 12,
    'MotorImagery': 64,
    'NerveDamage': 1,
    'PhonemeSpectra': 11,
    'RacketSports': 6,
    'SelfRegulationSCP1': 6,
    'SelfRegulationSCP2': 7,
    'SpokenArabicDigits': 13,
    'Trace': 1,
    'UWaveGestureLibrary': 3,
    'Wafer': 1,
    'WalkingSittingStanding': 2,
}

# ========================调用subset===============
def read_set(path):
    """ 读取所有的subset名字
    """
    folders = []
    files = os.listdir(path)
    for file in files:
        if os.path.isdir(os.path.join(path, file)):
            folders.append(file)
    check_dict(folders, class_dict) # 检测是否所有的subset都有class属性
    return folders

def check_dict(folders, dict=class_dict):
    """ 检查是否所有的subset都在class_dict里
    """
    s_list = list(dict.keys())
    for i in folders:
        if i not in s_list:
            raise Exception(f'Class attribute of {i} subsets are losses')

def set_sta(list, class_dict=class_dict):
    """ 返回当前subet在所有待导入subset中的class开始序号
    Args:
        list: 需要导入的所有subset名字
        class_dict: 包含所有subsetclass属性的字典，默认为当前文件中的class_dict
    Returns:
        rep_dict: 包含旧标签和新标签的对应表
    """
    sta, rep_dict, class_all =0,{},0
    for i, na in enumerate(list):  #遍历所有的list
        rep_dict[na] = {}
        for j in range(0, class_dict[na]):   # 遍历当前subset的所有label
            rep_dict[na][j] = sta + j
        sta += class_dict[na]
        class_all += class_dict[na]
    return rep_dict, class_all

def inf_rep(na, class_dict=class_dict):
    ''' 初始化对应表
    '''
    rep_dict = {}
    for j in range(0, class_dict[na]):  # 遍历当前subset的所有label
        rep_dict[j] = j
    return rep_dict

def max_len(su, di=len_dict):
    ''' 返回目前调用数据集中的长度最大值
    '''
    va=[]
    for i in su:
        va.append(di[i])
    return max(va)

def max_chan(su, di=chan_dict):
    ''' 返回目前调用数据集中的特征维度最大值
    '''
    va=[]
    for i in su:
        va.append(di[i])
    return max(va)

def pad_dim(te, ta):
    ''' 将输入tensor的维度补充到指定维度
    '''
    pa = torch.zeros([te.shape[0], te.shape[1], ta-te.shape[2]])
    feature = torch.cat((te, pa), dim=2)
    return feature


