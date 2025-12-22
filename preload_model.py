import os
import gc
import torch
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

# ---- CONFIG ----
MODEL_ID = os.getenv("MODEL_ID", "Wan-AI/Wan2.2-I2V-14B")


print("Downloading Wan 2.2 models...")

# Avoid CUDA during build
torch.set_default_device("cpu")

transformer = WanTransformer3DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)

transformer_2 = WanTransformer3DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer_2",
    torch_dtype=torch.bfloat16
)

pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer,
    transformer_2=transformer_2,
    torch_dtype=torch.bfloat16
)

# ---- Download LoRA weights ----
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    adapter_name="lightx2v"
)

pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    adapter_name="lightx2v_2",
    load_into_transformer_2=True
)

# Cleanup
del pipe, transformer, transformer_2
gc.collect()

print("✅ Wan 2.2 models & LoRA cached successfully")
