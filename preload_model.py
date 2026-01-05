# premodel_download.py

def ensure_models():
    from huggingface_hub import snapshot_download
    from pathlib import Path

    WAN_REPO_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    LORA_REPO_ID = "Kijai/WanVideo_comfy"

    BASE_MODEL_DIR = Path("/runpod-volume/models/Wan2.2-I2V-A14B-Diffusers")
    LORA_DIR = Path("/runpod-volume/models/lora/WanVideo_comfy")

    WAN_SENTINEL = BASE_MODEL_DIR / "model_index.json"
    LORA_SENTINEL = (
        LORA_DIR
        / "Lightx2v"
        / "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"
    )

    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LORA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Base Wan 2.2 model ----
    if not WAN_SENTINEL.exists():
        print("⬇️ Downloading Wan 2.2 base model...")
        snapshot_download(
            repo_id=WAN_REPO_ID,
            repo_type="model",
            local_dir=str(BASE_MODEL_DIR),
            cache_dir="/runpod-volume/huggingface",
            local_dir_use_symlinks=False,
            allow_patterns=[
                "model_index.json",
                "scheduler/*",
                "text_encoder/*",
                "tokenizer/*",
                "transformer/*",
                "transformer_2/*",
                "vae/*",
            ],
        )
    else:
        print("✅ Wan 2.2 base model already present")

    # ---- LoRA ----
    if not LORA_SENTINEL.exists():
        print("⬇️ Downloading Wan LoRA...")
        snapshot_download(
            repo_id=LORA_REPO_ID,
            repo_type="model",
            local_dir=str(LORA_DIR),
            cache_dir="/runpod-volume/huggingface",
            local_dir_use_symlinks=False,
            allow_patterns=[
                "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"
            ],
        )
    else:
        print("✅ Wan LoRA already present")
