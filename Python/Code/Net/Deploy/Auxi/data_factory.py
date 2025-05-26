from Python.Code.Net.Deploy.Auxi import collate_fn
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np
import Code.Package.main_funcions as MF
import torch


def data_provider(args, da, la, flag):
    data_set = MF.signalDataset(data=torch.from_numpy(da), label=torch.from_numpy(la))

    # 用于判断是否是训练/测试
    if flag == 'test':  # 加载测试数据    #
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size[args.seq]

    else:
        shuffle_flag = True # 用于训练
        drop_last = False
        batch_size = args.batch_size[args.seq]  # bsz for train and valid

    # collate_fn(data_set, max_len=args.model_seq_len)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last,
        collate_fn=lambda x: collate_fn(x, max_len=args.model_seq_len)     # 补充到大模型的长度
    )                               # collate_fn 处理动态变化的批处理长度，把batch长度调整一致
                                        # 同时padding_mask矩阵也是这个函数生成的
    return data_loader
