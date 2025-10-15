import mlx.core as mx
from .basics import softmax, linear

# 单头简单注意力机制: (N, H, L, D)Q、K、V -> (N, H, L, D)
def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        scores = scores + mask
    return mx.matmul(softmax(scores, axis=-1), value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int, # E(H * D)
        num_heads: int, # H
        wq: mx.array, # 查询投影矩阵shape=(H*D, H*D)
        wk: mx.array, # 键投影矩阵shape=(H*D, H*D)
        wv: mx.array, # 值投影矩阵shape=(H*D, H*D)
        wo: mx.array, # 输出投影矩阵shape=(H*D, H*D)
    ):
        self.hidden_size = hidden_size # E总体维度
        self.num_heads = num_heads # H头数
        self.head_dim = hidden_size // num_heads # D单头维度
        self.scale = mx.rsqrt(self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array, # (N, L, H * D)
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None, # (L, L)
    ) -> mx.array:
        # 投影，混合权重
        query = linear(query, self.wq) # (N, L, H * D) + (H * D, H * D) -> (N, L, H * D)
        key = linear(key, self.wk) # 同上
        value = linear(value, self.wv) # 同上
        # reshape
        N, L, _ = query.shape
        query = query.reshape(N, L, self.num_heads, self.head_dim)
        key = key.reshape(N, L, self.num_heads, self.head_dim)
        value = value.reshape(N, L, self.num_heads, self.head_dim)
        # 转置，方便后续计算
        query = query.transpose(0, 2, 1, 3) # (N, H, L, D)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        # 计算注意力
        output = scaled_dot_product_attention_simple(query, key, value, self.scale, mask) # (N, H, L, D)
        output = output.transpose(0, 2, 1, 3) # (N, L, H, D)
        output = output.reshape(N, L, self.hidden_size) # (N, L, H * D)
        return linear(output, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
