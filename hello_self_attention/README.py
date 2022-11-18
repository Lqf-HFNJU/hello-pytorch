"""
self - attention 带来的问题
    attention 太大，算出来要太久

解决
    人为设定
        Local/Truncated Attention
            选择 k 旁边的两个来做 attention ，其他全部置为 0
        Stride Attention
            选择一个步长 stride ，每个 k 和该步长的来做 attention ，其他置为 0
        Global Attention
            每个 seq 选几个或者另外设置几个作为全局 attention ，其他都只和这个做 attention ，剩下置为 0
        Long transformer
            attention 里面的每个 head 分别做上面的事，一个综合
        Big Bird
            Long transformer + Random Attention(随机搞几个点做 attention)
        Clustering
            使用快速的聚类方法来对每个 k 和 q 做一个聚类， 然后计算 attention 的时候只选择同一类的来计算， 其他置为 0

    机器自己设定(data driven)
        Sinkhorn Sorting Network
            另设一个 network 来选择保留哪个 attention (值小的就直接设为 0)
        Informer
            k矩阵是低秩的，在原先的 k 里面选择/抽取代表性的 k' 来参与 attention 计算
            只选 k 而不选 q 因为 q 的个数决定了输出向量的个数
            把原先 N x N 的 attention 降维成 N x d (k' 的个数) 矩阵
        compressed attention
            用 CNN 选择提取 k
        Synthesizer
            直接将 attention 视为网络参数，让机器自己学习
            所以有个问题就是 attention 的意义在哪里？
        Linear Transformer
            如果不对原始 attention 使用 softmax 的时候，交换矩阵运算顺序， 先算 V x KT , 再右乘 Q ,可以将计算复杂度从 NxN 降到 N
            如果使用 softmax ，可以近似找一个核函数 f , 使得 e^(k·q) = f(k)·f(q), 然后在计算的时候做矩阵转换，发现有大量重复运算，可以保存中间计算步骤来加速
"""
