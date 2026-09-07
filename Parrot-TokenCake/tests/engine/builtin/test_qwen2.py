import parrot  # noqa: F401
from transformers import LlamaConfig

from parrot.engine.builtin.models import MODEL_ARCH_MAP, LlamaForCausalLM
from parrot.engine.config import BuiltinConfig


def test_qwen2_architecture_alias():
    assert MODEL_ARCH_MAP["Qwen2ForCausalLM"] is LlamaForCausalLM


def test_qwen2_gqa_projection_shape():
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=128,
        vocab_size=256,
        pad_token_id=0,
    )
    config.model_type = "qwen2"
    builtin_config = BuiltinConfig(
        num_kv_cache_blocks=16,
        attn_func="xformers_with_buffer",
        device="cpu",
    )

    model = LlamaForCausalLM(config, builtin_config)
    attn = model.model.layers[0].self_attn

    assert attn.num_heads == 8
    assert attn.num_key_value_heads == 2
    assert attn.num_key_value_groups == 4
    assert attn.q_size == 64
    assert attn.kv_size == 16
    assert attn.qkv_proj.weight.shape == (96, 64)
    assert attn.qkv_proj.bias is not None


def test_llama_projection_shape_is_unchanged():
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=8,
        num_hidden_layers=2,
        max_position_embeddings=128,
        vocab_size=256,
        pad_token_id=0,
    )
    builtin_config = BuiltinConfig(
        num_kv_cache_blocks=16,
        attn_func="xformers_with_buffer",
        device="cpu",
    )

    model = LlamaForCausalLM(config, builtin_config)
    attn = model.model.layers[0].self_attn

    assert attn.num_key_value_heads == 8
    assert attn.num_key_value_groups == 1
    assert attn.qkv_proj.weight.shape == (192, 64)
