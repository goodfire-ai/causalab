"""Compare Qwen and Llama model architectures"""

import torch
from neural.pipeline import LMPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*80)
print("Qwen/Qwen2.5-0.5B")
print("="*80)
qwen_pipeline = LMPipeline("Qwen/Qwen2.5-0.5B", max_new_tokens=1, device=device, dtype=torch.float16, max_length=32)
print(f"Vocab size: {qwen_pipeline.model.config.vocab_size}")
print(f"Hidden size: {qwen_pipeline.model.config.hidden_size}")
print(f"Num heads: {qwen_pipeline.get_num_attention_heads()}")
print(f"Num layers: {qwen_pipeline.get_num_layers()}")
print(f"Head dim: {qwen_pipeline.model.config.hidden_size // qwen_pipeline.get_num_attention_heads()}")
print(f"Model dtype: {qwen_pipeline.model.dtype}")

# Check if model has specific attributes that might affect NaN
if hasattr(qwen_pipeline.model.config, 'rms_norm_eps'):
    print(f"RMS norm eps: {qwen_pipeline.model.config.rms_norm_eps}")
if hasattr(qwen_pipeline.model.config, 'rope_theta'):
    print(f"RoPE theta: {qwen_pipeline.model.config.rope_theta}")

del qwen_pipeline
torch.cuda.empty_cache()

print("\n" + "="*80)
print("meta-llama/Meta-Llama-3.1-8B-Instruct")
print("="*80)
llama_pipeline = LMPipeline("meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=1, device=device, dtype=torch.float16, max_length=32)
print(f"Vocab size: {llama_pipeline.model.config.vocab_size}")
print(f"Hidden size: {llama_pipeline.model.config.hidden_size}")
print(f"Num heads: {llama_pipeline.get_num_attention_heads()}")
print(f"Num layers: {llama_pipeline.get_num_layers()}")
print(f"Head dim: {llama_pipeline.model.config.hidden_size // llama_pipeline.get_num_attention_heads()}")
print(f"Model dtype: {llama_pipeline.model.dtype}")

if hasattr(llama_pipeline.model.config, 'rms_norm_eps'):
    print(f"RMS norm eps: {llama_pipeline.model.config.rms_norm_eps}")
if hasattr(llama_pipeline.model.config, 'rope_theta'):
    print(f"RoPE theta: {llama_pipeline.model.config.rope_theta}")
