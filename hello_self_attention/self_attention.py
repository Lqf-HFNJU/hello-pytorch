"""
实现原始 self attention
"""
import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    # 注意力机制头数  输入特征维度 输出特征维度（也就是q k v的维度）
    def __init__(self, num_attention_heads, input_size, output_size):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(output_size / num_attention_heads)  # 每个头要多少维
        self.all_head_size = output_size  # 所有头的长度(输出层长度)
        if output_size % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (output_size, num_attention_heads)
            )

        self.key_layer = nn.Linear(input_size, output_size)
        self.query_layer = nn.Linear(input_size, output_size)
        self.value_layer = nn.Linear(input_size, output_size)

    # 多头机制的最终维度为：(batch_size,num_attention_heads,seq_len,attention_head_size)
    def trans_to_multiple_heads(self, x):
        # (batch_size, seq_len, num_heads, heads_size)
        new_size = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        # 维度换位
        return x.permute(0, 2, 1, 3)

    # x: (batch_size, seq_len, feature_dim)
    def forward(self, x):
        # 得到 k q v 矩阵 (batch_size, seq_len, output_size)
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        # 转换为多头形式 (batch_size, head_num, seq_len, heads_size)
        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        # (batch_size, head_num, seq_len, heads_size) * (batch_size, head_num, heads_size, seq_len)
        # => (batch_size, head_num, seq_len, seq_len)
        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        # 对a矩阵做一下变换
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 对注意力矩阵归一化, 保持原有维度
        attention_probs = F.softmax(attention_scores, dim=-1)

        # => (batch_size, head_num, seq_len, heads_size)
        context = torch.matmul(attention_probs, value_heads)
        # 将各头的注意力矩阵进行拼接
        # contiguous() 将Tensor的内存变成连续的，否则view的时候会报错
        # => (batch_size, seq_len, head_num, heads_size)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[:-2] + (self.all_head_size,)  # 为了弄成元组好拼接
        context = context.view(new_size)
        return context


if __name__ == '__main__':
    features = torch.rand((32, 20, 10))  # (batch_size, seq_len, feature_dim)
    attention = SelfAttention(2, 10, 30)  # (head_num, feature_dim, output_dim)
    result = attention.forward(features)
    print(result.shape)
