import runpod
import uuid
import os
import logging
import time
import shutil
import torch
from pathlib import Path
from utils.s3 import download_image, upload_video
from utils.utllity import (load_environment,
                           reset_runtime_env,
                           classify_env,
                           extract_last_clear_frame,)
from utils.video import load_pipe, generate_video
from moviepy import VideoFileClip, concatenate_videoclips

logging.basicConfig(level=logging.INFO)




_ = load_pipe() #loading wan2.2
SEED = 123


def handler(event):
    workdir = None
    clips = []

    try:
        reset_runtime_env()
        inp = event["input"]

        # ---- Info-only mode ----
        if inp.get("aleef") is True:
            return {
                "service": "wan2.2-i2v",
                "version": "1.0",
                "inputs": ["prompts", "clip_sec", "img_path", "level"],
            }

        prompts = inp["prompts"]
        clip_sec = inp.get("clip_sec", 5)
        img_path = inp["img_path"]
        level = inp.get("level")

        # ---- Environment selection ----
        try:
            if level:
                load_environment(level)
            else:
                _, _, bucket, *_ = img_path.split("/")
                level = classify_env(bucket)
                load_environment(level)
        except Exception:
            load_environment() # default = stag

        # ---- Working directory ----
        workdir = Path("/tmp") / str(uuid.uuid4())
        workdir.mkdir(parents=True, exist_ok=True)

        input_img = workdir / "input.png"
        download_image(img_path, str(input_img))

        current_image = str(input_img)
        video_paths = []
        total_time = 0

        # ---- Generate clips ----
        for idx, prompt in enumerate(prompts):
            start = time.time()
            logging.info(f"🎬 Generating clip {idx + 1}")

            video_path = workdir / f"clip_{idx + 1}.mp4"
            frame_path = workdir / f"clip_{idx + 1}_last.jpg"

            generate_video(
                image_path=current_image,
                prompt=prompt,
                output_path=str(video_path),
                duration_sec=clip_sec,
                steps=8,
                seed=SEED + idx,
            )

            video_paths.append(str(video_path))

            current_image = extract_last_clear_frame(
                video_path=str(video_path),
                output_path=str(frame_path),
                sharpness_threshold=75.0,
            )

            total_time += time.time() - start

        logging.info(f"⏱️ Total generation time: {total_time:.2f}s")

        # ---- Concatenate ----
        final_video_path = workdir / "final.mp4"

        clips = [VideoFileClip(v) for v in video_paths]
        final = concatenate_videoclips(clips, method="chain")

        final.write_videofile(
            str(final_video_path),
            codec="libx264",
            audio=False,
            fps=clips[0].fps,
            logger=None,
        )

        # ---- Upload ----
        key = f"video_gen/wan2/{uuid.uuid4()}.mp4"
        s3_path = upload_video(str(final_video_path), key)

        return {"video_path": s3_path}

    except Exception as e:
        logging.exception("❌ Generation failed")
        return {"error": str(e)}

    finally:
        # ---- Close MoviePy handles ----
        for c in clips:
            try:
                c.close()
            except Exception:
                pass

        try:
            final.close()
        except Exception:
            pass

        # ---- Cleanup files ----
        if workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)

        # ---- Cleanup GPU ----
        torch.cuda.empty_cache()


runpod.serverless.start({"handler": handler})
