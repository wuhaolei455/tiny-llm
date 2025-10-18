import json
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open
from transformers import AutoTokenizer

from tiny_llm.attention import SimpleMultiHeadAttention
from tiny_llm.quantize import QuantizedWeights


def load_quantized_weights(model_dir: Path) -> dict[str, mx.array]:
    weight_file = model_dir / "model.safetensors"
    if not weight_file.exists():
        raise FileNotFoundError(f"未找到模型权重文件: {weight_file}")

    weights: dict[str, mx.array] = {}
    with safe_open(str(weight_file), framework="np") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            weights[key] = mx.array(tensor)
    return weights


def inspect_attention_output(texts: str | list[str]):
    if isinstance(texts, str):
        texts = [texts]

    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models" / "Qwen" / "Qwen2-0___5B-Instruct-MLX"

    config_file = model_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"未找到模型配置文件: {config_file}")
    with config_file.open("r", encoding="utf-8") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    vocab_size = config["vocab_size"]
    group_size = config["quantization"]["group_size"]
    bits = config["quantization"]["bits"]

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    inputs = tokenizer(
        texts,
        return_tensors="np",
        padding=True,
    )
    input_ids_np = inputs["input_ids"].astype("int64")
    tokens_batch = [
        tokenizer.convert_ids_to_tokens(ids.tolist())
        for ids in input_ids_np
    ]


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

    attention = SimpleMultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        wq=wq,
        wk=wk,
        wv=wv,
        wo=wo,
    )

    output = attention(token_embeddings, token_embeddings, token_embeddings)
    print(f"SimpleMultiHeadAttention输出shape: {output.shape}")

    for sample_idx, (text, ids, tokens) in enumerate(zip(texts, input_ids_np, tokens_batch)):
        ids_list = ids.tolist()
        print(f"样本[{sample_idx}] 原始文本: {text}")
        print(f"样本[{sample_idx}] input_ids: {ids_list}")
        print(f"样本[{sample_idx}] tokens: {tokens}")

        decoded_tokens = [
            tokenizer.convert_tokens_to_string([token]).strip()
            for token in tokens
        ]
        dog_indices = [
            idx for idx, decoded in enumerate(decoded_tokens)
            if decoded.lower() == "dog"
        ]
        if len(dog_indices) >= 2:
            first_idx, second_idx = dog_indices[:2]
            dog1_out = output[sample_idx, first_idx]
            dog2_out = output[sample_idx, second_idx]
            identical = mx.allclose(dog1_out, dog2_out, atol=1e-6)
            print(
                f"样本[{sample_idx}] token索引 {first_idx} 与 {second_idx} (token='{tokens[first_idx]}', '{tokens[second_idx]}') 输出是否一致: {identical}"
            )
        else:
            print(f"样本[{sample_idx}] 未找到重复 token 用于比较。")


if __name__ == "__main__":
    inspect_attention_output(["The dog chased another dog"])
