import os
import json

import mlx.core as mx
from safetensors import safe_open
from transformers import AutoTokenizer

from tiny_llm.attention import SimpleMultiHeadAttention
from tiny_llm.quantize import QuantizedWeights


def load_quantized_weights(model_dir: str) -> dict[str, mx.array]:
    weight_file = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"未找到模型权重文件: {weight_file}")

    weights: dict[str, mx.array] = {}
    with safe_open(weight_file, framework="np") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            weights[key] = mx.array(tensor)
    return weights


def inspect_attention_output(text: str):
    model_dir = "/Users/wuhaolei/code/demos/llm/tiny-llm/models/Qwen/Qwen2-0___5B-Instruct-MLX"

    config_file = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"未找到模型配置文件: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    vocab_size = config["vocab_size"]
    group_size = config["quantization"]["group_size"]
    bits = config["quantization"]["bits"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    inputs = tokenizer(text, return_tensors="np")
    input_ids_np = inputs["input_ids"].astype("int64")

    weights = load_quantized_weights(model_dir)

    def dequantize_weight(prefix: str) -> mx.array:
        quant = QuantizedWeights(
            scales=weights[f"{prefix}.scales"],
            biases=weights[f"{prefix}.biases"],
            group_size=group_size,
            bits=bits,
            weight=weights[f"{prefix}.weight"],
        )
        return mx.dequantize(
            quant.weight,
            quant.scales,
            quant.biases,
            quant.group_size,
            quant.bits,
        )

    embed_weight = dequantize_weight("model.embed_tokens")
    token_embeddings = mx.take(embed_weight, mx.array(input_ids_np), axis=0)

    layer_prefix = "model.layers.0.self_attn"
    wq = dequantize_weight(f"{layer_prefix}.q_proj")
    wk = dequantize_weight(f"{layer_prefix}.k_proj")
    wv = dequantize_weight(f"{layer_prefix}.v_proj")
    wo = dequantize_weight(f"{layer_prefix}.o_proj")

    print("wq shape:", wq.shape)
    print("wk shape:", wk.shape)
    print("wv shape:", wv.shape)
    print("wo shape:", wo.shape)

    print("head_dim:", wq.shape[1] // num_heads)
    print("kv head dim:", wk.shape[1] // (wk.shape[0] // num_heads if wk.shape[0] < hidden_size else num_heads))

    attention = SimpleMultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        wq=wq,
        wk=wk,
        wv=wv,
        wo=wo,
    )

    print("输入token:", text)
    print("input_ids:", input_ids_np)
    print("token_embeddings shape:", token_embeddings.shape)
    print("token_embeddings:", token_embeddings)

    output = attention(token_embeddings, token_embeddings, token_embeddings)
    print("SimpleMultiHeadAttention输出shape:", output.shape)
    print("SimpleMultiHeadAttention输出:", output)


if __name__ == "__main__":
    inspect_attention_output("我爱学习")
