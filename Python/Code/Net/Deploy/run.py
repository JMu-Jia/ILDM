# 将在上游任务学习到的模型部署到下游任务中

import argparse
import os
import torch
from Python.Code.Net.Deploy.Auxi import Exp_Classification as Exp
from Python.Code.Net.Deploy.Auxi import read_set
import random
import numpy as np
import Python.Code.Net.Deploy.Auxi.Mix_subset as MS
import sys

def run():  # 包装成函数，方便debug
    # ===================================deployment定义参数===============================
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--debug', type=bool, default=False, help='debug model (less set and iter)')
    parser.add_argument('--save_path', type=str, default='./Results/Model/', help='location of model checkpoints')
    parser.add_argument('--cluster_name', type=str, default='LTM', help='cluster model name')
    parser.add_argument('--model_name', type=str, default='Deploy', help='cluster model name')
    parser.add_argument('--current_path', type=str, default=sys.path[0], help='current work path')

    # 基础参数
    parser.add_argument('--model_abla', type=str, default='NonFC-NonSL-LoRA', choices=['NonNonFC','NonFC','NonFC-LoRA','UEA-L','UEA-UL','UEA-UL-LoRA'], help='ablation model name')
    parser.add_argument('--max_iter', type=int, default=2, help='max iter for in debug model')
    parser.add_argument('--terminal_name', type=str, default=['HIT', 'Motor', 'WT'], help='order of the terminal')
    parser.add_argument('--input_feature', type=int, default=[6, 6, 3], help='the number of channels of each terminal')
    parser.add_argument('--output_class', type=int, default=[4, 6, 5], help='the number of class of each terminal')
    parser.add_argument('--exp_per', type=float, default=[0, 0.01, 0.1, 0.5, 1.0], help='the percentage of participating training sampel of each terminal')
    parser.add_argument('--len_per', type=float, default=[0.01, 0.1, 0.5, 1.0], help='the percentage of participating training sampel of each terminal')
    parser.add_argument('--train_per', type=float, default=.7, help='the percentage of training sampel of each terminal')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience') # Ealystopping # 如果连续下降那么多次则停止  #！ 重点要调整
    parser.add_argument('--vali_out', type=int, default=1, help='vali interval')  # 验证集的间隔
    parser.add_argument('--lradj', type=str, default='type0', help='adjust learning rate')  # 可变学习率


    parser.add_argument('--train_epochs', type=int, default=1000, help='train epochs') #！ 重点要调整
    parser.add_argument('--batch_size', type=int, default=[60, 60, 40], help='batch size')
    parser.add_argument('--learning_rate', type=float, default=[0.0001, 0.0001, 0.0001], help='learning rate of all loss')

    parser.add_argument('--lora_r', type=int, default=4, help='LoRA attention dimension')
    parser.add_argument('--lora_alpha', type=int, default=1000, help='The alpha parameter for LoRA scaling')
    parser.add_argument('--lora_drop', type=float, default=0.001, help='dropout rate of LoRA')
    parser.add_argument('--sm_a', type=float, default=0.5, help='OnlineLabelSmoothing Term for balancing soft_loss and hard_loss')  # OnlineLabelSmoothing 相关参数
    parser.add_argument('--sm_f', type=float, default=0.1, help='OnlineLabelSmoothing Smoothing factor to be used during first epoch in soft_loss')  # OnlineLabelSmoothing 相关参数

    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')  # 多线程参数，在只有CPU或者1个GPU的情况下，非0会报错
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()


    # ------------------------设备挂载定义--------------------------------
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu: # 定义使用的GPU
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # ======================开始部署训练============================
    for i, na in enumerate(args.terminal_name):    # 遍历所有的终端：
        if na == 'HIT' :
            args.exp_per = [0, 0.01, 0.1, 0.5, 1.0]
            args.len_per = [0.001, 0.01, 0.1, 0.5, 1.0]
            args.lora_r = 2
            args.lora_alpha = 1000
            args.lora_drop = 0
        if na == 'Motor':
            args.exp_per = [0, 0.01, 0.1, 0.5, 1.0]
            args.len_per = [0.001, 0.01, 0.1, 0.5, 1.0]
            args.lora_r = 1
            args.lora_alpha = 1000
            args.lora_drop = 0.001
        elif na == 'WT':
            args.exp_per = [0, 0.001, 0.01, 0.1, 0.5, 1.0]
            args.len_per = [0.01, 0.1, 0.5, 1.0]
            args.lora_r = 3
            args.lora_alpha = 1000
            args.lora_drop = 0.0001

        for j in args.exp_per:  # 遍历所有的零样本或小样本集设置
            for k in args.len_per:  # 遍历所有的信号长度
                if j == 1.0 or k == 1.0:
                    exp = []
                    args.deploy = na    # 部署到的终端名称
                    args.par_per = j    # 参与到训练的样本占训练集的比例
                    args.seq = i    # 该终端在所有终端中的顺序
                    args.len = k    # 参与训练的信号长度

                    # fine best hyper-para
                    # args.deploy = 'HIT'  # 部署到的终端名称
                    # args.seq = 0   # 该终端在所有终端中的顺序
                    # args.par_per = 1  # 参与到训练的样本占训练集的比例
                    # args.len = 1

                    setting = '{}_per{}_len{}'.format(args.deploy,args.par_per,args.len) # 更新当前终端和数据集

                    exp = Exp(args)  # set experiments

                    print('>>>>>>>deploying : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    # exp.train(setting)
                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    # exp.test(setting)
                    torch.cuda.empty_cache()    # 释放显存

                    # exit()  # find the best hyper-para

                    if i ==0 and j == 0:  # 只会在零样本的时候运行一次，以保证每个复杂度计算旨在HIT的数据集上计算一次
                        exp = Exp(args)
                        exp.model_cal()
                        torch.cuda.empty_cache()  # 释放显
                        exit()



                    if args.debug == True:  # 如果进入debug模式，只用在一个终端
                        exit()


