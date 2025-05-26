import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from Code.Net.Cluster.Auxi.embed import DataEmbedding
from Code.Net.Cluster.Auxi.conv_blocks import Inception_Block_V2, Inception_Block_ResNet_V2   # 训练时间竟然比V1短
from Code.Net.Cluster.Auxi.odconv1d import ODConv1d

def FFT_for_Period(x, k=2):
    # xf shape [B, T, C], denoting the amplitude of frequency(T) given the datapiece at B,N
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k  # 多少个高频信息被考虑到
        # parameter-efficient design

        # Inceprion模块，后续可把里面的卷积层进行替换
        self.conv = nn.Sequential(
            Inception_Block_ResNet_V2(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels, scale=configs.scale),
            # nn.BatchNorm2d(configs.d_ff),
            # nn.GELU(),  # 激活函数 # 高斯误差线性单元 # inception_block输出时已自带GELU
            Inception_Block_ResNet_V2(configs.d_ff, configs.d_ff,
                                      num_kernels=configs.num_kernels, scale=configs.scale),
            Inception_Block_ResNet_V2(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels, scale=configs.scale),
            # nn.BatchNorm2d(configs.d_ff),
        )

    def forward(self, x):
        B, T, N = x.size() #B: batch size  T: length of time series  N:number of features， # N是不是通道数？
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

        # 修改后处理器
        self.post = nn.Conv1d(in_channels=configs.d_model ,out_channels=configs.d_model, kernel_size=1, bias=False, dilation=1, groups=1)
        # self.post = nn.Sequential(
            # nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=1, bias=False, dilation=1, groups=1),
            # ODConv1d(in_planes=configs.d_model, out_planes=configs.d_model, kernel_size=1)
        # )


    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)

        # 后处理层
        output = self.post(output.permute(0, 2, 1))  # 增加一层后处理
        output = output.permute(0, 2, 1)    # 再调换回去，这里是为了后面不用大改

        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
