from Code.Net.Cluster.Auxi.data_factory import data_provider
from Code.Net.Cluster.Auxi.exp_basic import Exp_Basic
from Code.Net.Cluster.Auxi.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from Code.Net.Cluster.Auxi.Mix_subset import read_set, set_sta, max_len, max_chan, pad_dim   # 读取混合数据集用到的数据
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import shutil
import random
import pickle
from openpyxl import Workbook
from torch.utils.tensorboard import SummaryWriter



warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        if self.args.debug == True: # 如果是debug模式
            self.set_list = ['AbnormalHeartbeat', 'ERing']
            # self.set_list = ['Handwriting', 'Earthquakes', 'ERing']
        else:
            self.set_list = read_set(self.args.root_path)    # 读取文件夹下所有的subset
        self.sta_dict, self.args.num_class = set_sta(self.set_list)
        self.args.seq_len = max_len(self.set_list)    # 返回所有subset中的最大信号长度
        self.args.pred_len = 0
        self.args.enc_in = max_chan(self.set_list)   # 返回所有subset中的最大通道数
        # self.criterion = OnlineLabelSmoothing(alpha=self.args.sm_a, n_classes=self.args.num_class, smoothing=self.args.sm_f).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.train_set, self.vali_set, self.test_set = locals(), locals(), locals()
        for i in self.set_list:  # 遍历所有的subset
            self.args.lab_rep = self.sta_dict[i]  # 标签替换指标
            self.args.model_id = i  # 导入的数据集
            self.train_set[f'train_data_{i}'], self.train_set[f'train_loader_{i}'] = self._get_data(flag='TRAIN')
            self.vali_set[f'vali_data_{i}'], self.vali_set[f'vali_loader_{i}'] = self._get_data(flag='TEST') # 验证集和测试集一样
            self.test_set[f'test_data_{i}'], self.test_set[f'test_loader_{i}'] = self._get_data(flag='TEST') # 导入测试集

        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def vali(self):
        preds = []
        trues = []
        # self.criterion.eval()
        self.model.eval()
        with torch.no_grad():
            for set in self.set_list:  # 每个epoch遍历所有的subet # 之后需要增加随机功能
                vali_data, vali_loader = self.vali_set.get(f'vali_data_{set}'), self.vali_set.get(f'vali_loader_{set}')

                for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                    batch_x = pad_dim(batch_x, self.args.enc_in)  # batch_x的channel补充至最大channel
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device) # padding_mask因为batch不能动态改变，所以同一batch需要padding到同一大小
                    label = label.to(self.device)

                    outputs = self.model(batch_x, padding_mask, None, None)

                    preds.append(outputs.detach())
                    trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        total_loss = self.criterion(preds, trues.long().squeeze())
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        # self.criterion.train()
        return total_loss, accuracy

    def train(self, setting):
        path = os.path.join(self.args.save_path, self.args.model_name, setting) # 创建文件保存路径
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.mkdir(path)
        time_sta = time.time()  # 记录训练开始时间

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()  # 定义优化器

        torch.save(self.model, path + '/' + 'inimodel.pth')
        print('save initial model ...')
        writer = SummaryWriter(log_dir=path + '/logs')  # 记录训练过程

        for epoch in range(self.args.train_epochs): # 开始训练
            iter_count = 0
            train_loss, train_acc = [],[]

            shuf_set = self.set_list # 每个epoch打乱sub_set的顺序
            random.shuffle(shuf_set)

            self.model.train()
            # self.criterion.train()

            for set in shuf_set:  # 每个epoch遍历所有的subet # 之后需要增加随机功能
                # print(set)  # 输出哪个数据集出错
                train_data, train_loader = self.train_set.get(f'train_data_{set}'), self.train_set.get(f'train_loader_{set}')

                for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                    # batch_x[16,1751,3](EthanolConcentration)  # 16: batch_size 1751: 信号长度 3：channel
                    iter_count += 1
                    model_optim.zero_grad()

                    batch_x = pad_dim(batch_x, self.args.enc_in)    # batch_x的channel补充至最大channel
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                    label = label.to(self.device)

                    outputs = self.model(batch_x, padding_mask, None, None)

                    loss = self.criterion(outputs, label.long().squeeze(-1))
                    train_loss.append(loss.item())

                    probs = torch.nn.functional.softmax(outputs.detach())  # (total_samples, num_classes) est. prob. for each class and sample
                    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
                    trues = label.flatten().cpu().numpy()
                    acc = cal_accuracy(predictions, trues)
                    train_acc.append(acc.item())

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0) # 解决梯度爆炸问题
                    model_optim.step()

            # self.criterion.next_epoch()  # 每个epoch后更新soft labels
            train_loss = np.average(train_loss)
            train_acc = np.average(train_acc)
            print("Epoch: {0} | Train Loss: : {1:.3f} | Train Acc: {2:.3f}".format(epoch, train_loss, train_acc))  # 输出当前epoch花费时间

            # 运行完所有的subset后输出一次
            if epoch % self.args.vali_out == 0 or epoch == 0 or epoch == self.args.train_epochs - 1:
                vali_loss, vali_acc = self.vali()  #  计算vali和test集的输出
                print("Epoch: {0} | Train Loss: {1:.3f} | Train Acc: {2:.3f} | Vali Loss: {3:.3f} | Vali Acc: {4:.3f} "
                    .format(epoch, train_loss, train_acc, vali_loss, vali_acc))
                early_stopping(-vali_acc, self.model, path) # 检测是否过拟合

            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('vali_loss', vali_loss, epoch)
            writer.add_scalar('vali_acc', vali_acc, epoch)

            # -------------------early_stopping判断条件（每个epoch会进行检查）----------------------
            if early_stopping.early_stop:   # 验证集连续下降则会触发early stopping
                print("Early stopping")
                break
            if (epoch + 1) % 10 == 0:    # 变学习率
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # -------------------循环结束的输出---------------------
        time_end = time.time()
        print('End training! All time: {:.4f}s\n'.format(time_end-time_sta))

        folder_path = os.path.join(self.args.save_path, self.args.model_name, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = 'train_result.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write('train time:{} | train loss:{} | train accuracy:{} | vali loss:{} | vali accuracy:{}'
                .format(time_end-time_sta, train_loss, train_acc, vali_loss, vali_acc))

        torch.save(self.model, path + '/' + 'bestmodel.pth')

        return self.model

    def test(self, setting, test=0):
        folder_path = os.path.join(self.args.save_path, self.args.model_name, setting)  # 定义保存路径
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, trues, datas = [], [], {}
        wb = Workbook()
        ws = wb.active
        ws.append(['Set name', 'Accuracy'])
        self.model.eval()
        # self.criterion.eval()
        with torch.no_grad():
            for set in self.set_list:  # 每个epoch遍历所有的subet # 之后需要增加随机功能
                test_data, test_loader = self.test_set.get(f'test_data_{set}'), self.test_set.get(f'test_loader_{set}')

                for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                    batch_x = pad_dim(batch_x, self.args.enc_in)
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                    label = label.to(self.device)

                    outputs = self.model(batch_x, padding_mask, None, None)

                    preds.append(outputs.detach())
                    trues.append(label)

                datas[f'{set}_real'] = torch.cat(trues, 0)
                datas[f'{set}_pred'] = torch.cat(preds, 0)

                # 包括每个subset的准确率
                pre = torch.cat(preds, 0)
                tru = torch.cat(trues, 0)
                prob = torch.nn.functional.softmax(pre)
                pred = torch.argmax(prob, dim=1).cpu().numpy()
                true = tru.flatten().cpu().numpy()
                accu = cal_accuracy(pred, true)
                ws.append([set, accu])

        wb.save(os.path.join(folder_path, 'subset_acc.xlsx'))
        file_name = 'test_record.pkl'  # 分开保存每个数据集
        f = open(os.path.join(folder_path, file_name), 'wb')
        pickle.dump(datas, f, -1)
        f.close()

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        file_name = 'test_output.pkl'   # 记录测试结果
        f = open(os.path.join(folder_path, file_name), 'wb')
        datas = {'preds': preds, 'trues': trues}
        pickle.dump(datas, f, -1)
        f.close()

        loss = self.criterion(preds, trues.long().squeeze(-1))  # 计算整体loss值
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # b = np.unique(trues)
        # auc_score = roc_auc_score(trues, preds.cpu().numpy(), multi_class='ovr')   # 计算AUC值
        # ValueError: Number of classes in y_true not equal to the number of columns in 'y_score'
        # 是因为测试集中真实样本类型比训练集少    # 改为‘ovo’模式即可

        # result save
        print('accuracy:{} | loss:{}'.format(accuracy, loss))
        file_name='test_result.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write('test accuracy:{} | test loss:{}'.format(accuracy, loss))
        f.write('\n')
        f.write('\n')
        f.close()
        return
