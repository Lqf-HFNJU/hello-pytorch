"""
transformer

Seq2Seq 的一个强大的实现
由 Encoder 和 Decoder 组成

Encoder
    由多个 block 组成， 每个 block 有一个 self - attention 求和、归一化 fc

Decoder

"""

"""
构造函数参数：
    d_model: Encoder和 Decoder输入参数的特征维度。也就是词向量的维度。默认为512
    nhead: 多头注意力机制中，head的数量。注意该值并不影响网络的深度和参数数量。默认值为8。
    num_encoder_layers: EncoderLayer的数量。该值越大，网络越深，网络参数量越多，计算量越大。默认值为 6
    num_decoder_layers：DecoderLayer的数量。该值越大，网络越深，网络参数量越多，计算量越大。默认值为 6
    dim_feedforward：Feed Forward层（Attention后面的全连接网络）的隐藏层的神经元数量。该值越大，网络参数量越多，计算量越大。默认值为 2048
    dropout：dropout值。默认值为 0.1
    activation： Feed Forward层的激活函数。取值可以是string(“relu” or “gelu”)或者一个一元可调用的函数。默认值是 relu
    custom_encoder：自定义Encoder。若不想用官方实现的TransformerEncoder，你可以自己实现一个。默认值为 None
    custom_decoder: 自定义Decoder。若不想用官方实现的TransformerDecoder，你可以自己实现一个。
    layer_norm_eps: Add&Norm层中，BatchNorm的 eps参数值。默认为 1e-5
    batch_first：batch维度是否是第一个。如果为True，则输入的shape应为(batch_size, 词数，词向量维度)，否则应为(词数, batch_size, 词向量维度)。默认为False。
    norm_first – 是否要先执行 norm。例如，在图中的执行顺序为 Attention -> Add -> Norm。若该值为True，则执行顺序变为：Norm -> Attention -> Add。

forward参数:
    src: Encoder的输入。也就是将 token进行 Embedding并 Positional Encoding之后的 tensor。必填参数。Shape为(batch_size, 词数, 词向量维度)
    tgt: 与src同理，Decoder的输入。 必填参数。Shape为(词数, 词向量维度)
    src_mask: 对src进行 mask。不常用。Shape为(词数, 词数)
    tgt_mask：对tgt进行 mask。常用。Shape为(词数, 词数)
    memory_mask – 对 Encoder的输出 memory进行 mask。 不常用。Shape为(batch_size, 词数, 词数)
    src_key_padding_mask：对 src的 token进行 mask. 常用。Shape为(batch_size, 词数)
    tgt_key_padding_mask：对 tgt的 token进行 mask。常用。Shape为(batch_size, 词数)
    memory_key_padding_mask：对 tgt的 token进行 mask。不常用。Shape为(batch_size, 词数)
"""
