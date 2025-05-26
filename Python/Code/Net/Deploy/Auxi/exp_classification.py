from Python.Code.Net.Deploy.Auxi import data_provider
from Python.Code.Net.Deploy.Auxi import Exp_Basic
from Python.Code.Net.Deploy.Auxi import EarlyStopping, adjust_learning_rate, cal_accuracy
from Python.Code.Net.Deploy.Auxi import pad_dim   # 读取混合数据集用到的数据
from Python.Code.Net.Deploy.Auxi import data_load
from Python.Code.Net.Cluster.Auxi.label_smooth import OnlineLabelSmoothing
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import shutil
import pickle
from openpyxl import Workbook
from peft import LoraConfig, get_peft_model
import copy
from sklearn.metrics import recall_score
# from thop import profile
# from thop import clever_format
from ptflops import get_model_complexity_info


from torchsummary import summary
from memory_profiler import profile # 记录内存占用率
from Code.Auxiliary.gpu_mem_track import MemTracker # 记录显存占用率
import inspect


warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # 加载在服务器上训练好的最佳模型
        model_path = os.path.join('./Results/BestModel', self.args.cluster_name, '0')  # 定义保存路径
        basic_model = torch.load(os.path.join(model_path, 'bestmodel.pth'), map_location=self.device).float()  # 会导入服务器的所有路径，所以cluster的Auxi也要下载
        # 需要知道输入的信号长度通道和输出类别数
        self.args.model_seq_len = basic_model.configs.seq_len  # model的输入信号长度
        self.args.model_enc_in = basic_model.configs.enc_in # model的输入通道数
        self.args.model_num_class = basic_model.configs.num_class # model的输出类别数

        model = self._change_model(basic_model)

        # 导入该客户端的数据集
        self.data_set = data_load(self.args)    # 加载数据集并划分训练和测试
        if self.args.par_per == 0:  # 零样本时无训练集
            self.train_loader = None
        else:
            self.train_loader = self._get_data(self.data_set['train_da'], self.data_set['train_la'], flag='TRAIN')
        self.vali_loader = self._get_data(self.data_set['test_da'], self.data_set['test_la'], flag='TEST')
        self.test_loader = self._get_data(self.data_set['test_da'], self.data_set['test_la'], flag='TEST')

        if self.args.debug == True: # 如果是debug模式 # 防止进入零样本模式
            self.train_loader = self.vali_loader

        if 'NonFC' in self.args.model_abla or self.args.model_abla == 'NonFC-NonSL-LoRA':  # 在没有FC的情况下，会导致预测标签与真实标签不一致
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = OnlineLabelSmoothing(alpha=self.args.sm_a, n_classes=self.args.output_class[self.args.seq], smoothing=self.args.sm_f).to(self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, da, la, flag):
        # 将本地数据集调整为culser模型能读取的尺寸
        data_loader = data_provider(self.args, da, la, flag)
        return data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate[self.args.seq])
        model_optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.learning_rate[self.args.seq])
        return model_optim

    def _change_model(self, basic_model):    # 部署时根据不同模型调整
        # ['NonFC','Full','UEA-L','UEA-UL','UEA-LoRA']
        if self.args.model_abla == 'NonNonFC':  # 如果把non-UEA训练的模型直接拿来用，没有LOCK和LORA
            model_path = os.path.join('./Results/InitialModel')  # 定义保存路径
            model = torch.load(os.path.join(model_path, 'InitialModel.pth'), map_location=self.device).float()

        if self.args.model_abla == 'NonFC': # 如果直接把UEA训练后的模型直接拿来用，没有LOCK和LORA
            model = copy.deepcopy(basic_model)

        if self.args.model_abla == 'NonUEA-FC': # 如果把non-UEA训练的模型直接拿来用,增加一层随机初始化FC
            model_path = os.path.join('./Results/InitialModel')  # 定义保存路径
            ini_model = torch.load(os.path.join(model_path, 'InitialModel.pth'), map_location=self.device).float()
            model = New_Net1(ini_model, self.args.model_num_class, self.args.output_class[self.args.seq]).float()

        if self.args.model_abla == 'NonFC-LoRA' or self.args.model_abla == 'NonFC-NonSL-LoRA': # 在FC的基础上增加LoRA后
            name = read_line_name(basic_model)
            peft_config = LoraConfig(inference_mode=False, r=self.args.lora_r, lora_alpha=self.args.lora_alpha, lora_dropout=self.args.lora_drop, target_modules=name)
            model = get_peft_model(basic_model, peft_config) # Lora 模型
            model.print_trainable_parameters()  # 输出可学习参数结果

        if self.args.model_abla == 'UEA-L':  # UEA训练好LOCK后，增加一层随机初始化FC
            model = New_Net1(basic_model, self.args.model_num_class, self.args.output_class[self.args.seq]).float() # lock UEA

        if self.args.model_abla == 'UEA-UL':  # UEA训练好后UNLOCK，增加一层随机初始化FC
            model = New_Net2(basic_model, self.args.model_num_class, self.args.output_class[self.args.seq]).float() # unlock UEA

        if self.args.model_abla == 'UEA-UL-LoRA':  # UEA训练好后UNLOCK使用LoRA，增加一层随机初始化FC
            name = read_line_name(basic_model)
            peft_config = LoraConfig(inference_mode=False, r=self.args.lora_r, lora_alpha=self.args.lora_alpha, lora_dropout=self.args.lora_drop, target_modules=name)
            model = get_peft_model(basic_model, peft_config)  # Lora 模型
            model.print_trainable_parameters()
            model = New_Net1(model, self.args.model_num_class, self.args.output_class[self.args.seq]).float()

        return model

    def vali(self):
        preds = []
        trues = []
        if 'NonFC' not in self.args.model_abla or self.args.model_abla != 'NonFC-NonSL-LoRA':
            self.criterion.eval()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(self.vali_loader):
                batch_x = pad_dim(batch_x, self.args.model_enc_in)  # batch_x的channel补充至最大channel
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device) # padding_mask因为batch不能动态改变，所以同一batch需要padding到同一大小
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

                if self.args.debug == True: # 如果是debug模式，跳出循环
                    break

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        total_loss = self.criterion(preds, trues.long().squeeze())
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        if 'NonFC' not in self.args.model_abla or self.args.model_abla != 'NonFC-NonSL-LoRA':
            self.criterion.train()
        return total_loss, accuracy

    def train(self, setting):   # 非零样本时调用该程序进行训练
        if self.train_loader == None:   # 如果是零样本学习，则自动结束该train
            return

        path = os.path.join(self.args.save_path, self.args.model_name, self.args.model_abla, setting) # 创建文件保存路径
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.mkdir(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()  # 定义优化器
        time_sta = time.time()  # 记录训练开始时间

        for epoch in range(self.args.train_epochs): # 开始训练
            iter_count = 0
            train_loss, train_acc = [],[]

            self.model.train()
            if 'NonFC' not in self.args.model_abla or self.args.model_abla != 'NonFC-NonSL-LoRA':
                self.criterion.train()

            for i, (batch_x, label, padding_mask) in enumerate(self.train_loader):
                # batch_x[10,5120,6](HIT)  # 16: batch_size 5120: 信号长度 6：channel
                iter_count += 1
                model_optim.zero_grad()

                batch_x = pad_dim(batch_x, self.args.model_enc_in)    # batch_x的channel补充至cluster模型的通道
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None) # 大模型的输出结果

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

                if self.args.debug == True: # 如果是debug模式，跳出循环
                    break

            if 'NonFC' not in self.args.model_abla or self.args.model_abla != 'NonFC-NonSL-LoRA':
                self.criterion.next_epoch()  # 每个epoch后更新soft labels
            train_loss = np.average(train_loss)
            train_acc = np.average(train_acc)
            print("Epoch: {0} | Train Loss: : {1:.3f} | Train Acc: {2:.3f}".format(epoch, train_loss, train_acc))  # 输出当前epoch花费时间

            # 运行完所有的subset后输出一次
            if epoch % self.args.vali_out == 0 or epoch == 0 or epoch == self.args.train_epochs - 1:
                vali_loss, vali_acc = self.vali()  # 计算vali和test集的输出
                print("Epoch: {0} | Train Loss: {1:.3f} | Train Acc: {2:.3f} | Vali Loss: {3:.3f} | Vali Acc: {4:.3f} ".format(epoch, train_loss, train_acc, vali_loss, vali_acc))
                early_stopping(-vali_acc, self.model, path)  # 检测是否过拟合


            # -------------------early_stopping判断条件（每个epoch会进行检查）----------------------
            if early_stopping.early_stop:  # 验证集连续下降则会触发early stopping
                print("Early stopping")
                best_epoch = epoch-self.args.patience+1 # record the epoch of best vali
                break
            if (epoch + 1) % 10 == 0:  # 变学习率
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        # -------------------循环结束的输出---------------------
        time_end = time.time()
        print('End training! All time: {:.4f}s\n'.format(time_end - time_sta))

        folder_path = os.path.join(self.args.save_path, self.args.model_name, self.args.model_abla, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = 'train_result.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write('train time:{} | train loss:{} | train accuracy:{} | vali loss:{} | vali accuracy:{} | best epoch:{}'
                .format(time_end - time_sta, train_loss, train_acc, vali_loss, vali_acc, best_epoch))

        torch.save(self.model, path + '/' + 'bestmodel.pth')

        return self.model

    def test(self, setting):    # test 模式
        folder_path = os.path.join(self.args.save_path, self.args.model_name, self.args.model_abla, setting)  # 定义保存路径
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.train_loader != None:   # 如果no是零样本学习，则load trained model
            self.model.load_state_dict(torch.load(os.path.join(folder_path, 'checkpoint.pth')))

        preds, trues, datas = [], [], {}
        wb = Workbook()
        ws = wb.active
        ws.append(['Set name', 'Accuracy'])
        self.model.eval()
        if 'NonFC' not in self.args.model_abla or self.args.model_abla != 'NonFC-NonSL-LoRA':
            self.criterion.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(self.test_loader):
                batch_x = pad_dim(batch_x, self.args.model_enc_in)  # batch_x的channel补充至最大channel
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)  # padding_mask因为batch不能动态改变，所以同一batch需要padding到同一大小
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

                if self.args.debug == True:  # 如果是debug模式，跳出循环
                    break
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        file_name = 'test_output.pkl'  # 记录测试结果
        f = open(os.path.join(folder_path, file_name), 'wb')
        datas = {'preds': preds, 'trues': trues}
        pickle.dump(datas, f, -1)
        f.close()

        loss = self.criterion(preds, trues.long().squeeze(-1))  # 计算整体loss值 #零样本情况下该值无效
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()

        if self.args.par_per == 0:  # 零样本的情况下，需要将pred标签进行重匹配
            if 'NonFC' in self.args.model_abla: # 在没有FC的情况下，会导致预测标签与真实标签不一致
                mo=True
            else: mo=False
            predictions = labels_to_class(trues, predictions, self.args.output_class[self.args.seq], mo)

        accuracy = cal_accuracy(predictions, trues)
        reca = recall_score(predictions, trues, labels=np.arange(self.args.output_class[self.args.seq]), average='micro')

        # result save
        print('accuracy:{} | loss:{} | recall:{}'.format(accuracy, loss, reca))
        file_name = 'test_result.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write('test accuracy:{} | test loss:{} | test recall:{}'.format(accuracy, loss, reca))
        f.write('\n')
        f.write('\n')
        f.close()
        return

    def model_cal(self):  # 计算模型的复杂度和参数量 # 各指标为模型在HIT dataset上的统计结果
        # self.model是在前言里修改后的model，包括可能增加FC或者LoRA的新模型
        bas_model = copy.deepcopy(self.model)
        if self.args.model_abla == 'UEA-L': # part change to eval()
            bas_model.model.eval()

        model = Net_cal(bas_model, self.args.model_seq_len, self.args.model_enc_in)
        if self.args.model_abla == 'UEA-L': # part change to eval()
            model.model.model.eval()
        summary(model.to('cpu'), input_size=(1, self.args.model_seq_len, self.args.model_enc_in), batch_size=-1)    # running on cpu

        input = torch.rand((1, self.args.model_seq_len, self.args.model_enc_in)).to(self.device)
        mask = torch.ones((1, self.args.model_seq_len)).to(self.device)

        flops, params = get_model_complexity_info(model.to('cpu'), (1, self.args.model_seq_len, self.args.model_enc_in),as_strings=True, print_per_layer_stat=True)
        print('para: ', params)
        print('flops: ',flops)

        frame = inspect.currentframe()  # if报错，can not find the txt, 可能是因为不是在GPU上运行导致的
        gpu_tracker = MemTracker(frame)
        gpu_tracker.track() # 显存监测  # print in .txt
        y = record(bas_model.to(self.device), input,mask)   # 内存监测
        gpu_tracker.track()

        # test model .eval()
        bas_model.eval().to(self.device)
        fi_ti = time.time()
        y = bas_model(input, mask, None, None)
        la_ti = time.time()
        print('one sample test time: ', la_ti - fi_ti)

        return


@profile(precision=4)   # 计算内存占用率
def record(model,input, mask):
    y = model(input, mask, None,None)    # 传入1个sample
    return


from collections import Counter # 统计元素出现次数
from itertools import permutations

def labels_to_class(trues, preds, cla, mo=True):   # 将预测标签与零样本标签做对应
    if mo:  # 预测与真实长度不一致，需先调整成一致
        index_t = Counter(trues).most_common(cla)   # 原标签的顺序
        index_p = Counter(preds).most_common(cla-1)   # 预测标签的顺序

        ans = []
        for i in range(len(index_p)):   # 提出预测出的前几个标签
            ans.append(index_p[i][0])

        for i in range(len(preds)): # 遍历所有的预测标签
            if preds[i] not in ans:  # 如果不在预测标签的里面, 则赋值给最有一个标签
                preds[i] = index_t[cla - 1][0]  # inde_t 需要减一位开始索引
            for j in range(len(index_p)):  # 遍历所有的前几
                if preds[i] == index_p[j][0]:   # 存在就替换
                    preds[i] = index_t[j][0]   # 替换为真实标签

    # 对预测标签进行排列组合
    ans = np.unique(trues)  # 返回真实标签的所有标签
    acc0 = float("-inf")
    for i in permutations(ans, int(len(ans))):  # 遍历所有的排序方式
        preds_adj = np.array(i)[preds]  # 遍历所有的标签可能
        acc = cal_accuracy(preds_adj, trues)
        if acc > acc0 : # 寻找最优pred标签
            preds_best = preds_adj  # 保存最优pred标签
            acc0=acc

    return preds_best

class Net_cal(nn.Module):   # 用来计算模型的参数
    def __init__(self, model, model_seq_len, model_enc_in):
        super(Net_cal, self).__init__()
        self.model = model
        self.model_seq_len = model_seq_len
        self.model_enc_in = model_enc_in
    def forward(self,x):
        x = torch.rand(1, self.model_seq_len, self.model_enc_in)
        mark = torch.ones(1, self.model_seq_len)
        y = self.model(x,mark,None,None)
        return y

def read_line_name(model):
    """ 读取模型中所有的可学习参数层，用于LoRA
    Args:
        model: 模型
    Returns:
        name: 可学习参数层名称
    """
    name = []
    print(model)
    rule = ['kernels', 'projection'] # 复合LORA使用的‘nn.Linear、nn.Embedding、nn.Conv2d’的层名
    model_named_modules = [x for x in model.named_modules()]    # 输出所有的层名
    for na, pa in model.named_parameters():   # 后续需要增加只提取line层的
        for j in rule:
            if j in na:    # 判断是否有全连接层的存在，weight和bias只保留一个
                n = na.split('.')
                na = '.'.join(n[0:-1])  #需要删除最后一个“.”后面的内容，不然不是层名
                name.append(na)
    name2=[]
    for i in name:
        if i not in name2:
            name2.append(i)
    return name2


class New_Net1(nn.Module):  # 用来计算模型的参数
    def __init__(self, basic_model, basic_num_class, aim_num_class):
        super(New_Net1, self).__init__()
        self.model = basic_model
        self.classifier = nn.Linear(basic_num_class, aim_num_class)
        nn.init.kaiming_normal_(self.classifier.weight,mode='fan_in', nonlinearity='leaky_relu')  # FC初始化

    def forward(self, x, mark, x_dec, x_mark_dec, mask=None):
        with torch.no_grad():  # 锁定训练好的UEA模型
            x = self.model(x, mark, None, None)
        y = self.classifier(x)

        return y


class New_Net2(nn.Module):  # 用来计算模型的参数
    def __init__(self, basic_model, basic_num_class, aim_num_class):
        super(New_Net2, self).__init__()
        self.model = basic_model
        self.classifier = nn.Linear(basic_num_class, aim_num_class)
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x, mark, x_dec, x_mark_dec, mask=None):
        x = self.model(x, mark, None, None)
        y = self.classifier(x)
        return y
