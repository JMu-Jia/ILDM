import os
import numpy as np
import pandas as pd
import glob
import torch
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
import pickle
import os

warnings.filterwarnings('ignore')


def data_load(args):
    # 导入数据集
    save_path = os.path.abspath(os.path.join(args.current_path, './Results/Data/Dataset'))
    os.chdir(save_path)
    with open('{}_Dataset.pkl'.format(args.deploy), 'rb') as f:  # 读取本地端文件
        data_set = pickle.load(f)
    os.chdir(args.current_path) # 切回原路径

    # 划分训练集和测试集
    train_da, train_la, test_da, test_la = [], [], [], []
    Data = {}
    for di in data_set:  # 遍历所有哦
        if 'da' in di:  # 判断是样本的dict
            data = data_set[di]
            np.random.seed(args.seq)  # 不同的客户端使用不同的seed
            index = np.random.permutation(np.arange(data.shape[0]))  # 生成索引序列
            data_ind = data[index, :, :]  # 样本随机打乱
            lab_ind = data_set[di.replace('d', 'l')][index]  # 标签随机打乱 # 虽然没啥用，但为了保持吃一致

            # 划分数据集
            tr_da = data_ind[:int(args.par_per * args.train_per * len(lab_ind)), :, :]
            tr_la = lab_ind[:int(args.par_per * args.train_per * len(lab_ind))]
            te_da = data_ind[int(args.train_per * len(lab_ind)):, :, :]
            te_la = lab_ind[int(args.train_per * len(lab_ind)):]

            train_da.append(tr_da)
            train_la.append(tr_la)
            test_da.append(te_da)
            test_la.append(te_la)

    Data['train_da'] = np.concatenate(train_da, axis=0)[:,:int(args.len*(data.shape[1])),:]  # list转numpy
    Data['train_la'] = np.concatenate(train_la, axis=0)
    Data['test_da'] = np.concatenate(test_da, axis=0)[:,:int(args.len*(data.shape[1])),:]
    Data['test_la'] = np.concatenate(test_la, axis=0)

    return Data

