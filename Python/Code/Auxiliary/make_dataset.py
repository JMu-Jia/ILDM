#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/11/13 9:55
# @Author : JMu
# @ProjectName : 用来根据不同的原始数据集制作可用的数据集

import os
from Code.Package import main_funcions as MF
import numpy as np
from sklearn import preprocessing

def HIT_dataset_generator(args):    # 航空发动机轴承数据集
    """
        每个mate中的data数据[504*8*20480]，504为样本数，8为通道数['Displacement_0','Displacement_1','Accelerate_0','Accelerate_1',
                                     'Accelerate_2','Accelerate_3','Speed','Label']，20480为信号长度
        数据集的整体标签为 0~3 [Normal,内圈故障（小）,内圈故障（大）,外圈故障]
    """
    set_path = os.path.abspath(os.path.join(args.current_path, '../Dataset\HIT\HIT-dataset'))

    filepath = os.path.join(set_path, 'data1.npy')  # 正常
    d1 = np.load(filepath)[:,:6,:]
    l1 = 0 * np.ones(d1.shape[0])

    filepath = os.path.join(set_path, 'data2.npy')  # 正常
    d2 = np.load(filepath)[:,:6,:]
    l2 = 0 * np.ones(d2.shape[0])

    filepath = os.path.join(set_path, 'data3.npy')  # 内圈小故障
    d3 = np.load(filepath)[:,:6,:]
    l3 = 1 * np.ones(d3.shape[0])

    filepath = os.path.join(set_path, 'data4.npy')  # 内圈大故障
    d4 = np.load(filepath)[:,:6,:]
    l4 = 2 * np.ones(d4.shape[0])

    filepath = os.path.join(set_path, 'data5.npy')  # 外圈故障
    d5 = np.load(filepath)[:,:6,:]
    l5 = 3 * np.ones(d5.shape[0])

    da = np.concatenate((d1, d2, d3, d4, d5), axis=0)  # 合并数据集
    la = np.concatenate((l1, l2, l3, l4, l5), axis=0)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 归一化
    da_norm = scaler.fit_transform(da.reshape(da.shape[0] * da.shape[1], -1).T).T.reshape(da.shape)

    index = np.random.permutation(np.arange(len(la)))  # 打乱所有样本
    data = da_norm[index, :, :]
    label = la[index]

    save_path = os.path.abspath(os.path.join(args.current_path, './Results\Data\HITSet'))
    os.chdir(save_path)  # 定义保存路径
    np.save('HIT_x', data)
    np.save('HIT_y', label)


def Motor_dataset_generator(args):    # 电机数据集
    """
    每个mate中的data数据[703*10000*9]，703为样本数，10000为信号长度，9为通道数['Voltage_0','Voltage_1','Voltage_2','Current_0',
                                 'Current_1','Current_2','Rotor_Current','Speed'，'Failde']
    数据集的整体标签为 0~5 [NF,1PSC,2PSC,VREC,OP,REVD]
    """
    set_path = os.path.abspath(os.path.join(args.current_path, '../Dataset\Motor\motor-faults-main'))

    filepath = os.path.join(set_path, 'Preprocessed_No_failed.mat') # 没有故障，NF
    labelname = 'No_failed'
    NF = MF.motor_mata_load(filepath, labelname)
    NF_l = 0 * np.ones(NF.shape[0])

    filepath = os.path.join(set_path, 'Preprocessed_Test_Data_Short_phases_Ln_G_.mat')  # 匝间短路 1PSC
    labelname = 'Test_Data_Short_phases_Ln_G_'
    onePSC = MF.motor_mata_load(filepath, labelname)
    onePSC_l = 1 * np.ones(onePSC.shape[0])

    filepath = os.path.join(set_path, 'Preprocessed_Short_between_two_phases_.mat')  # 两匝间短路 2PSC
    labelname = 'Short_between_two_phases_'
    twoPSC = MF.motor_mata_load(filepath, labelname)
    twoPSC_l = 2 * np.ones(twoPSC.shape[0])

    filepath = os.path.join(set_path, 'Preprocessed_Rotor_Current_Failed_R_.mat')  # 转子激发电压波动 VREC
    labelname = 'Rotor_Current_Failed_R_'
    VREC = MF.motor_mata_load(filepath, labelname)
    VREC_l = 3 * np.ones(VREC.shape[0])

    filepath = os.path.join(set_path, 'Preprocessed_Disconnect_Phase_10_11_21_.mat')    # 匝间断路 OP
    labelname = 'Disconnect_Phase_10_11_21_'
    OP = MF.motor_mata_load(filepath, labelname)
    OP_l = 4 * np.ones(OP.shape[0])

    filepath = os.path.join(set_path,'Preprocessed_Test_Data_Rotor_Current_Faild.mat')  # 转子激发电压断路 REVD
    labelname = 'Test_Data_Rotor_Current_Faild'
    REVD = MF.motor_mata_load(filepath, labelname)
    REVD_l = 5 * np.ones(REVD.shape[0])

    da = np.concatenate((NF,onePSC,twoPSC,VREC,OP,REVD),axis=0)   #合并数据集
    da = da.transpose(0, 2, 1)  # 数据二三维转置
    la = np.concatenate((NF_l,onePSC_l,twoPSC_l,VREC_l,OP_l,REVD_l),axis=0)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 归一化
    da_norm = scaler.fit_transform(da.reshape(da.shape[0] * da.shape[1], -1).T).T.reshape(da.shape)

    index = np.random.permutation(np.arange(len(la)))    # 打乱所有样本
    data = da_norm[index,:,:]
    label = la[index]

    save_path = os.path.abspath(os.path.join(args.current_path, './Results\Data\MotorSet'))
    os.chdir(save_path)  # 定义保存路径
    np.save('Motor_x', data)
    np.save('Motor_y', label)

def Gear_dataset_generator(args): # 齿轮数据集
    """
        每个txt中的data数据[312288*14]，10000为信号长度，14为通道数[时间点,12个加速度传感器，时间脉冲]
        数据集的整体标签为 0~4 [00,02,06,10,14]，数值为齿轮的缺陷大小
        """
    set_path = os.path.abspath(os.path.join(args.current_path, '../Dataset\Gear\XJTU_Spurgear/0-20-0Hz'))
    sample_len = 1000   # 设置切片长度

    filepath = os.path.join(set_path, 'spurgear00.txt') # 正常
    d1 = MF.load_txt(filepath)[:,1:13]
    d1 = MF.sample_divid(d1, sample_len)    # 信号段切片
    l1 = 0 * np.ones(d1.shape[0])

    filepath = os.path.join(set_path, 'spurgear02.txt')  # 故障大小02
    d2 = MF.load_txt(filepath)[:, 1:13]
    d2 = MF.sample_divid(d2, sample_len)  # 信号段切片
    l2 = 1 * np.ones(d2.shape[0])

    filepath = os.path.join(set_path, 'spurgear06.txt')  # 故障大小06
    d3 = MF.load_txt(filepath)[:, 1:13]
    d3 = MF.sample_divid(d3, sample_len)  # 信号段切片
    l3 = 2 * np.ones(d3.shape[0])

    filepath = os.path.join(set_path, 'spurgear10.txt')  # 故障大小10
    d4 = MF.load_txt(filepath)[:, 1:13]
    d4 = MF.sample_divid(d4, sample_len)  # 信号段切片
    l4 = 3 * np.ones(d4.shape[0])

    filepath = os.path.join(set_path, 'spurgear14.txt')  # 故障大小14
    d5 = MF.load_txt(filepath)[:, 1:13]
    d5 = MF.sample_divid(d5, sample_len)  # 信号段切片
    l5 = 4 * np.ones(d5.shape[0])

    da = np.concatenate((d1, d2, d3, d4, d5), axis=0)  # 合并数据集
    da = da.transpose(0, 2, 1)  # 数据二三维转置
    la = np.concatenate((l1, l2, l3, l4, l5), axis=0)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 归一化
    da_norm = scaler.fit_transform(da.reshape(da.shape[0] * da.shape[1], -1).T).T.reshape(da.shape)

    index = np.random.permutation(np.arange(len(la)))  # 打乱所有样本
    data = da_norm[index, :, :]
    label = la[index]

    save_path = os.path.abspath(os.path.join(args.current_path, './Results\Data\GearSet'))
    os.chdir(save_path)  # 定义保存路径
    np.save('Gear_x', data)
    np.save('Gear_y', label)

    print(set_path)