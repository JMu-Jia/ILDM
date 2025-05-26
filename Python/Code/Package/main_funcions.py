#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/11/13 9:57
# @Author : JMu
# @ProjectName : 子函数的放置文件

import scipy.io as scio
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

def motor_mata_load(filepath,labelname):  # 导入电机的mat文件
    data = scio.loadmat(filepath)
    data = data['train_data']
    return data

def load_txt(path):  # 读取txt文件
    """ 读取指定路径下txt文件, 返回里面数据
    Args:
        path: txt文件所在路径
    Returns:
        data: 保存的txt读取结果
    """

    da=[]
    for dal in open(path, "r"):
        da.append(e_str2array(dal)) #读取科学计数法数据
    data=np.array(da)
    return data

def e_str2array(data_str):
    """ txt文件中的科学计数法保存数据转为数组保存
    Args:
        data_str: 科学技术法数据的str格式
    Returns:
        np.array(lists)：转换后的数组格式
    """
    lists = []
    x=data_str.strip().split('\t')
    for data in x:
        lists.append(eval(data))    #科学计数法转数字
    return np.array(lists)

def sample_divid(da, len):
    """ 将长信号切分
    Args:
        da: 数据
        len: 切分长度
    Returns:
        data：切分后的数组
    """
    data = []
    range_max = da.shape[0]
    for j in range(0, range_max, len):  # 信号段切分
        if j + len > range_max: # 不够长度则删除
            break
        dat = da[j:(j + len),:]
        data.append(dat)  # 填充信号段
    return np.array(data)

def save_numpy(data,name,path,save=False):
    """ 保存numpy数组到指定路径
    Args:
        data: 需要保存的数数组
        name: 保存的文件名
        path: 文件保存路径
        save=False: 默认不保存
    """
    if save==True:
        os.chdir(path)  #定义保存路径
        np.save(name, data)

def load_numpy(name,path,read=True):
    """ 读取numpy数组
    Args:
        name: 保存数组的文件名
        path: 文件所在文件夹
        save=False: 默认不读取
    Returns:
        data: 读取出的数组
    """
    if read == True:
        os.chdir(path)  #定义工作路径
        data = np.load(name, allow_pickle=True)
        return data

class signalDataset(Dataset):   #制作自己的TensorDataset数据集
    def __init__(self,data,label):  #输入数据和标签
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx],self.label[idx]

def plot_confusion_matrix(matrix, labels,path, title=' 1DCNN Confusion matrix',save=True):
    """ 画混淆矩阵
    Args:
        matrix: 矩阵
        labels: 数据标签
        title: 图片标题
        path: 保存路径
        save=True: 是否保存，默认True为保存
    """
    fig, ax = plt.subplots()
    ax = sns.heatmap(matrix, cmap=plt.cm.Blues, annot=True)
    ax.set_xticks([x for x in range(len(labels))])
    ax.set_yticks([y for y in range(len(labels))])
    # 在小刻度上放置标签
    ax.set_xticks([x + 0.5 for x in range(len(labels))], minor=True)
    ax.set_xticklabels(labels,fontsize=10, minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(labels)-1,-1,-1)], minor=True)
    ax.set_yticklabels(labels[::-1], fontsize=10, minor=True)
    # 隐藏主要刻度标签
    ax.tick_params(which='major', labelbottom=False, labelleft=False)
    # 最后，隐藏小刻度线
    ax.tick_params(which='minor', width=0)
    # Add finishing touches
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.grid(True, linestyle=':')
    # ax.set_title(title)
    fig.tight_layout()
    plt.ylabel('The label of true category')
    plt.xlabel('The label of predicted category')
    if save:
        path=os.path.join(path,(title+".jpg"))
        plt.savefig(path,bbox_inches = 'tight', dpi=300)