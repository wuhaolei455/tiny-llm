import mlx.core as mx

tokens = ["我", "爱", "学习"]

# shape (1, 2, 3, 4) (batch, head, token, dim)
tensor = mx.array([
    [  # batch 0
        [  # head 0
            [ 0.10,  0.25,  0.55,  0.15],  # “我”
            [ 0.07,  0.42,  0.18,  0.33],  # “爱”
            [ 0.22,  0.31,  0.48,  0.12],  # “学习”
        ],
        [  # head 1
            [ 0.34,  0.08,  0.27,  0.39],  # “我”
            [ 0.58,  0.14,  0.05,  0.23],  # “爱”
            [ 0.21,  0.47,  0.16,  0.30],  # “学习”
        ],
    ]
])

print("shape:", tensor.shape)  # (1, 2, 3, 4)
print("tensor.shape", tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3])

for head in range(tensor.shape[1]):
    print(f"\nHead {head}:")
    for idx, token in enumerate(tokens):
        print(f"  token '{token}': {tensor[0, head, idx]}")

swapaxes_tensor = tensor.swapaxes(-2, -1)
print("swapaxes_tensor.shape", swapaxes_tensor.shape[0], swapaxes_tensor.shape[1], swapaxes_tensor.shape[2], swapaxes_tensor.shape[3])

for head in range(swapaxes_tensor.shape[1]):
    print(f"\nHead {head}:")
    for idx, token in enumerate(tokens):
        print(f"  token '{token}': {swapaxes_tensor[0, head, idx]}")