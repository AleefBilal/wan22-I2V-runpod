import os
import gc
import torch
import numpy as np
from PIL import Image

from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video

from torchao.quantization import quantize_
from torchao.quantization import Int8WeightOnlyConfig




# =========================
# Global config
# =========================
MODEL_ID = "/runpod-volume/models/Wan2.2-I2V-A14B-Diffusers"
LORA_ID = "/runpod-volume/models/lora/WanVideo_comfy"
DEVICE = "cuda"

FIXED_FPS = 16
MAX_DIM = 832
MIN_DIM = 480
MULTIPLE_OF = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 7720

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, motion artifacts, unstable motion, jitter, frame jitter, "
    "wobbling limbs, motion distortion, inconsistent movement, robotic movement, "
    "animation-like motion, awkward transitions, incorrect body mechanics, unnatural posing, "
    "off-balance poses, broken motion paths, frozen frames, duplicated frames, frame skipping, "
    "warped motion, stretching artifacts bad anatomy, incorrect proportions, deformed body, "
    "twisted torso, broken joints, dislocated limbs, distorted neck, unnatural spine curvature, "
    "malformed hands, extra fingers, missing fingers, fused fingers, distorted legs, extra limbs, "
    "collapsed feet, floating feet, foot sliding, foot jitter, backward walking, unnatural gait "
    "blurry details, ghosting, compression noise, jpeg artifacts, cartoon texture"
)

# =========================
# Global pipeline handle
# =========================
_PIPE = None


# =========================
# Utilities
# =========================
def get_num_frames(duration_seconds: float) -> int:
    frames = int(round(duration_seconds * FIXED_FPS))
    return int(np.clip(frames + 1, MIN_FRAMES_MODEL, MAX_FRAMES_MODEL))


def resize_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    scale = min(MAX_DIM / max(w, h), 1.0)
    w, h = int(w * scale), int(h * scale)
    w = (w // MULTIPLE_OF) * MULTIPLE_OF
    h = (h // MULTIPLE_OF) * MULTIPLE_OF
    w = max(MIN_DIM, w)
    h = max(MIN_DIM, h)
    return image.resize((w, h), Image.LANCZOS)


# =========================
# Pipeline loader (ONE TIME)
# =========================
def load_pipe():
    global _PIPE

    if _PIPE is not None:
        return _PIPE

    torch.cuda.empty_cache()
    gc.collect()

    print("🚀 Loading Wan 2.2 transformers...")

    transformer = WanTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    transformer_2 = WanTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer_2",
        torch_dtype=torch.bfloat16,
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        transformer_2=transformer_2,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)

    print("⚡ Loading Lightning LoRA...")

    pipe.load_lora_weights(
        "/runpod-volume/models/lora/WanVideo_comfy",
        weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
        adapter_name="lightx2v",
    )

    pipe.load_lora_weights(
        "/runpod-volume/models/lora/WanVideo_comfy",
        weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
        adapter_name="lightx2v_2",
        load_into_transformer_2=True,
    )

    pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1.0, 1.0])

    pipe.fuse_lora(
        adapter_names=["lightx2v"],
        lora_scale=3.0,
        components=["transformer"],
    )

    pipe.fuse_lora(
        adapter_names=["lightx2v_2"],
        lora_scale=1.0,
        components=["transformer_2"],
    )

    pipe.unload_lora_weights()

    print("🔒 Applying safe quantization...")
    quantize_(pipe.text_encoder, Int8WeightOnlyConfig())

    pipe.transformer.to(torch.bfloat16)
    pipe.transformer_2.to(torch.bfloat16)

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    _PIPE = pipe
    print("✅ Pipeline ready")

    return _PIPE


# Video generation
def generate_video(
    image_path: str,
    prompt: str,
    output_path: str,
    duration_sec: float = 3.0,
    steps: int = 4,
    seed: int = 42,
    guidance_scale: float = 1.0,
    guidance_scale_2: float = 1.0,
):
    pipe = load_pipe()

    torch.cuda.empty_cache()
    gc.collect()

    image = Image.open(image_path).convert("RGB")
    image = resize_image(image)

    num_frames = get_num_frames(duration_sec)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    out = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        height=image.height,
        width=image.width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
        generator=generator,
    )

    frames = out.frames[0]
    del out
    torch.cuda.empty_cache()

    export_to_video(frames, output_path, fps=FIXED_FPS)
    return output_path