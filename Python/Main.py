#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/11/9 10:44
# @Author : JMu
# @ProjectName : SCT框架主程序

import argparse
import sys
from Code.Auxiliary import make_dataset as MD
import torch
import numpy as np
import os
import subprocess
from Code.Net.Cluster.run import run as RunLSM


# ====================测试程序=====================
# from Code.Auxiliary.acc_polar import run as RunAP
#
# args.model_name = 'SCT'
# RunAP(args)
# exit()

# ====================画图=====================
# path = sys.path[0]
# from Code.Auxiliary.plot_fig import run as RunPF
# RunPF(path)
# exit()

# =====================建立不同的数据集=======================
# path = sys.path[0]
# MD.HIT_dataset_generator(path)  # 航发轴承故障
# MD.HST_dataset_generator(path)  # 高速列车故障
# MD.Motor_dataset_generator(path)    # 电机故障
# MD.WT_dataset_generator(path)    # 风机齿轮箱故障
# exit()

# ====================从服务器上下载模型并部署================
# from Code.Net.Deploy.run import run as RunDP
# RunDP()

# ====================从服务器上下载模型并部署================
from Code.CompareModel.run import run as RunCM
RunCM()

# =====================调用Cluster的大模型==================
# from sktime.datasets import load_from_tsfile_to_dataframe
# filepath = 'D:\\Research\\SCT\\Dataset\\UEA\\InsectWingbeat\\InsectWingbeat_TRAIN.ts'
# df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
# lengths = df.applymap(lambda x: len(x)).values

# RunLSM()    # 调用LSM大模型
# exit()


# tensorboard --logdir=

print('end')