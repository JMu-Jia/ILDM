import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from Python.Code.Net.Deploy.Auxi import ODConv1d
import math
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):  # max_len默认为5000    # 后期将max_len调整为数据大小？
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):    # 对通道进行压缩
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2

        # 一层结构
        # self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
        #                            kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # self.tokenConv = ODConv1d(in_planes=c_in, out_planes=d_model, kernel_size=3, padding=padding)    # 具有注意力机制
        #
        # 需要写一段逐级通道压缩   # 多地减少维度可能会造成信息的损失(表征性瓶颈)。
        tokenConv = []
        level = math.ceil(c_in/d_model) # 分级    # 自动生成
        cha = np.around(np.linspace(c_in, d_model, num =level+1))
        for i in range(level):
            tokenConv.append(nn.Conv1d(in_channels=int(cha[i]), out_channels=int(cha[i+1]), kernel_size=3, padding=padding, padding_mode='circular', bias=False))
            # tokenConv.append(ODConv1d(in_planes=int(cha[i]),out_planes=int(cha[i+1]), kernel_size=3, padding=padding))
        self.tokenConv = nn.ModuleList(tokenConv)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)    # 只用单层level

        # 多层level
        out = x.permute(0, 2, 1)
        for i, layer in enumerate(self.tokenConv):
            out = layer(out)
        x = out.transpose(1, 2)

        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:  # 分类任务走的这个分支
            x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


