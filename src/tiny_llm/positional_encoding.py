import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims # head数目
        self.seq_len = seq_len # 序列长度
        self.base = base # base to what
        self.traditional = traditional
        self.half_dims = dims // 2

        # angles[i, j] 就是“第 i 个 token 用在第 j 个频率上的旋转角度”
        inv_freq = mx.arange(0, self.half_dims, dtype=mx.float32) / self.half_dims # [0., 1/3, 2/3]
        inv_freq = mx.power(base, -inv_freq) # 取指数后取倒数, [0, 1]
        positions = mx.arange(seq_len, dtype=mx.float32) # [0, 1, 2, ..., seq_len-1]
        angles = mx.outer(positions, inv_freq) # (L, D // 2), 不同的旋转角度, C[i, j] = A[i] * B[j]
        self.cos_freqs = mx.cos(angles) # cos波形
        self.sin_freqs = mx.sin(angles) # sin波形


    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None # x: (N, L, H, D)
    ) -> mx.array:
        N, L, H, D = x.shape

        # handle offest: 1.no offest, entire seq; 2.single slice; 3.list of slices; no3
        # (1, L, half_dims) -> (N, L, half_dims)
        if offset is None:
            cos_basis = self.cos_freqs[:L, :]
            sin_basis = self.sin_freqs[:L, :]
            cos_basis = mx.broadcast_to(cos_basis[None, :, :], (N, L, self.half_dims))
            sin_basis = mx.broadcast_to(sin_basis[None, :, :], (N, L, self.half_dims))
        elif isinstance(offset, slice):
            cos_basis = self.cos_freqs[offset, :]
            sin_basis = self.sin_freqs[offset, :]
            cos_basis = mx.broadcast_to(cos_basis[None, :, :], (N, L, self.half_dims))
            sin_basis = mx.broadcast_to(sin_basis[None, :, :], (N, L, self.half_dims))
        else: # list of slices
            pass

        if self.traditional:
            x = x.reshape(N, L, H, self.half_dims, 2)
            x_even = x[..., 0]
            x_odd = x[..., 1]
        else:
            x_even = x[..., : self.half_dims] # first half
            x_odd = x[..., self.half_dims :] # second half

        # (N, L, half_dims) -> (N, L, 1, half_dims)
        cos_basis = cos_basis[:, :, None, :]
        sin_basis = sin_basis[:, :, None, :]

        # formula: output[i] = x[i] * cos_freqs[i] + x[i+1] * -sin_freqs[i]
        # not matrix multiplication, but element-wise multiplication
        real = mx.multiply(x_even, cos_basis) - mx.multiply(x_odd, sin_basis)
        imag = mx.multiply(x_odd, cos_basis) + mx.multiply(x_even, sin_basis)

        # 
        if self.traditional:
            y = mx.stack([real, imag], axis=-1)
            y = y.reshape(N, L, H, D)
        else:
            y = mx.concat([real, imag], axis=-1)
            y = y.reshape(N, L, H, D)
        return y.astype(x.dtype)
