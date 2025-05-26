import argparse
import os
import torch
from Code.Net.Cluster.Auxi.exp_classification import Exp_Classification as Exp
from Code.Net.Cluster.Auxi.Mix_subset import read_set
import random
import numpy as np
import Code.Net.Cluster.Auxi.Mix_subset as MS

def run():  # 包装成函数，方便debug
    # ===================================BigModel定义参数===============================
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--debug', type=bool, default=False, help='debug model (less set and iter)')
    parser.add_argument('--task_name', type=str,  default='classification', help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--model', type=str, default='TimesNet', help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--model_name', type=str, default='LTM', help='large model name')

    # data loader
    parser.add_argument('--data', type=str,  default='UEA', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../Dataset/UEA/', help='root path of the data file')
    parser.add_argument('--save_path', type=str, default='./Results/Model/', help='location of model checkpoints')

    # model define
    parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--scale', type=float, default=1.0, help='scaling factor')
    parser.add_argument('--sm_a', type=float, default=0.5, help='OnlineLabelSmoothing Term for balancing soft_loss and hard_loss')
    parser.add_argument('--sm_f', type=float, default=0.1, help='OnlineLabelSmoothing Smoothing factor to be used during first epoch in soft_loss')

    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=500, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=80, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--lradj', type=str, default='type0', help='adjust learning rate')

    # output
    parser.add_argument('--vali_out', type=int, default=1, help='vali interval')
    # GPU
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    print(os.path.abspath(args.root_path))

    # ------------------------设备挂载定义--------------------------------
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu: # 定义使用的GPU
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

# ===========================LLM+===================================
#     from Code.Net.InternVL import run as RunLM
#     RunLM(args)
#     exit()

# =======================subset读取================
# =======================开始训练===================
    if args.is_training:    # 从头开始训练
        for ii in range(args.itr):  # 实验重复次数
            # setting record of experiments
            setting = '{}'.format(ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            torch.cuda.empty_cache()    # 释放显存
    else:   # 加载训练后的模型
        ii = 0
        setting = '{}'.format(ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
