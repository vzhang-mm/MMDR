import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.utils.rnn as rnn_utils
from torch.nn.parameter import Parameter
import random
import os
from torch.autograd import Variable
'''
将文本特征、音频特征和视觉特征分别输入到各自的线性回归层进行抑郁症分数预测，
并将其和每个模态的特征向量进行拼接，然后使用这些新的特征向量来输入到FC层进行预测。
'''
# python train.py --clip_len 240 --dataset avec --modelName TF --batch_size 16 --lr 1e-4
if torch.cuda.device_count() > 1:
    print("多卡")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # '0,1'指定GPU编号
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 创建GPU对象
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Embedding(nn.Module):
    '''将特征映射到隐空间''' #2048=》512
    def __init__(self, inpt_dim = 2048, embed_dim =512):
        super(Embedding, self).__init__()
        self.fc = nn.Sequential(
            # nn.LayerNorm(inpt_dim),#特征维度上归一化
            nn.Linear(inpt_dim, embed_dim),#
            nn.ReLU(),
        )
        # self.fc = nn.Linear(inpt_dim, embed_dim, bias=False)
    def forward(self, x):
        x = self.fc(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, input_dim=512):#64、128
        super(AttentionBlock, self).__init__()

        self.multi_att = nn.Sequential(
            nn.Linear(input_dim, 1), #channel, channel // reduction分别为in out bias=False偏置为0
            nn.ReLU(inplace=True),
            # nn.Sigmoid()  # SE有
        )

    def forward(self,x_v,x_a,x_p):

        x = torch.stack([x_v,x_a,x_p], dim=2)
        # print("x1",self.multi_att(x).size())
        att_m = nn.Softmax(dim=2)(self.multi_att(x).squeeze(-1)).unsqueeze(-1)
        # print("x2",att_m.size())
        x = torch.sum(x * att_m, dim=2)  # 沿着1维度求和
        # print(x.size())
        return x
        # out = self.avg_pool(x) + self.max_pool(x)
        # # print(out.size())
        # att_q = self.seq_att(out.squeeze(-1)).unsqueeze(-1)
        # return att_q*x

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        # self.W = nn.Linear(input_dim, 1)#源代码

        self.W = nn.Sequential(
            nn.Linear(input_dim, 1), #channel, channel // reduction分别为in out bias=False偏置为0
            nn.ReLU(inplace=True),
            # nn.Sigmoid()  # SE有
        )

    def forward(self, x):
        """
        input:batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        attention_weight:att_w : size (N, T, 1)
        return:utter_rep: size (N, H)
        """
        # scores = torch.matmul(Q, K.transpose(-1, -2))
        att_w = nn.Softmax(dim=1)(self.W(x).squeeze(-1)).unsqueeze(-1)#删一维再加一维
        # print("att_w",att_w.size())
        utter_rep = torch.sum(x * att_w, dim=1)#沿着1维度求和
        return utter_rep


class Cnn_a(nn.Module):
    def __init__(self, inpt_dim=1024, embed_dim=512):
        super(Cnn_a, self).__init__()

        self.con1 = nn.Sequential(
            nn.Conv1d(inpt_dim, 128, kernel_size=10, stride=5, bias=False),  # L -> (L - 10)/7 + 1
            nn.ReLU(),
        )
        self.con2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=3, padding=2, bias=False),  # L -> (L + 2*2 - 7)/3 + 1
            nn.ReLU(),
        )
        self.con3 = nn.Sequential(
            nn.Conv1d(256, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),  # L -> (L + 2*1 - 3)/2 + 1
            nn.ReLU(),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling

    def forward(self, x):
        B, T, N, D = x.shape # N为1s内特征，49
        # x shape: (B, T, N, D) -> (B ,T ,D)
        print('x',x.shape)
        x = x.permute(0, 1, 3, 2)  # Change shape to (B, T, D, N)
        # x = x.transpose(1, 2)  # 转换形状为 (B, D, L)
        x = x.view(-1, D, N)
        x = self.con1(x)
        x = self.con2(x)
        x = self.con3(x)
        # print(x.shape)
        x = self.avg_pool(x).squeeze(dim=-1)
        # print(x.shape)
        x = x.view(B, T, -1)  # 转换回形状为 (B, L, D)
        return x

########################################################################################################################

if __name__ == '__main__':
   pass
