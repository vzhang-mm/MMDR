import copy
from typing import Optional
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from network import functions

class Transformer_encoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,#nhead=8
                dim_feedforward=2048, dropout=0.1,
                activation="relu", normalize_before=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)#1024是拼接后的维度
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        #SelfAttentionPooling
        self.sp = functions.SelfAttentionPooling(d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, v,mask_v):#, query_embed, pos_embed

        mask_v = mask_v.data.eq(0)
        # mask_B = mask_B.data.eq(0)
        # mask_P = mask_A.data.eq(0)#mask_P

        v = self.encoder(v, src_key_padding_mask=mask_v)
        v = self.sp(v)

        # a = self.encoder(a, src_key_padding_mask=mask_v)
        # a = self.sp(a)

        return v


class TransformerEncoder(nn.Module):

    def __init__(self, d_model, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        #位置编码
        self.pos_emb = PositionalEncoding_(d_model)###

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                ):
        output = src
        #位置编码
        pos = self.pos_emb(src)  # self.pos_emb(src.transpose(0, 1)).transpose(0, 1)
        for layer in self.layers:
            output = layer(output, src_mask=mask,src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)#
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


############位置编码
class PositionalEncoding_(nn.Module):
    def __init__(self, d_embed, seq_len=5000):
        super(PositionalEncoding_, self).__init__()
        self.d_model = d_embed
        pe = torch.zeros(seq_len, d_embed)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_embed, 2).float()
            * (-math.log(10000.0) / d_embed)
        )
        pe[:, 0::2] = torch.sin(position * div_term)# 字嵌入维度为偶数时
        pe[:, 1::2] = torch.cos(position * div_term)# 字嵌入维度为奇数时
        pe = pe.unsqueeze(0)## 在指定维度0上增加维度大小为1[3,4] -> [1,3,4]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # print(self.pe.size())
        x = x * math.sqrt(self.d_model)# sqrt() 方法返回数字x的平方根 适应多头注意力机制的计算，它说“这个数值的选取略微增加了内积值的标准差，从而降低了内积过大的风险，可以使词嵌入张量在使用多头注意力时更加稳定可靠”
        # print("x",x.size())
        x_pos = self.pe[:, : x.size(1), :]# 变为x.size(1)长度，torch.Size([1, 4, 512])
        # print("x_pos",x_pos.size())
        # x = x + x_pos#torch.Size([1, 4, 512])+torch.Size([4, 4, 512]) 每一个特征都加上位置信息
        # print("x",x.size())
        return x_pos#layer层会再加上位置信息


if __name__ == '__main__':
    pass

