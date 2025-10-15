import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array, # (N, I)
    w: mx.array, # (O, I)
    bias: mx.array | None = None, # (O)
) -> mx.array: # (N, O)
    if bias is not None:
        return mx.matmul(x, w.T) + bias
    else:
        return mx.matmul(x, w.T)


def silu(x: mx.array) -> mx.array:
    pass
