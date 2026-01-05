import os
import cv2
from dotenv import load_dotenv, find_dotenv

def load_environment(env_key: str = "stag"):
    if env_key not in ("stag", "prod"):
        raise ValueError("env_key must be 'stag' or 'prod'")

    env_file = find_dotenv(f"{env_key}.env", usecwd=True)
    if env_file:
        load_dotenv(env_file, override=False)
        print(f"🟢 Loaded local {env_key}.env")
    else:
        print("🟡 Using injected RunPod env vars")

    if env_key == "stag":
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["STAG_AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["STAG_AWS_SECRET_ACCESS_KEY"]
        os.environ["LAMBDA_BUCKET"] = os.environ["LAMBDA_BUCKET"]
    else:
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["PROD_AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["PROD_AWS_SECRET_ACCESS_KEY"]
        os.environ["LAMBDA_BUCKET"] = os.environ["LAMBDA_BUCKET"]

    os.environ.setdefault("AWS_REGION", "us-east-2")

    print(f"✅ Runtime environment configured: {env_key}")
    return env_key

def classify_env(value: str, default: str = "stag") -> str:
    if not value:
        return default

    val = value.lower()
    if "prod" in val or "production" in val:
        return "prod"
    if "stag" in val or "staging" in val:
        return "stag"
    return default



def extract_last_clear_frame(
    video_path: str,
    output_path: str,
    sharpness_threshold: float = 75.0,
    min_brightness: float = 30.0,   # <— NEW
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_clear_frame = None
    last_valid_frame = None

    for i in range(total_frames - 1, -1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        if last_valid_frame is None:
            last_valid_frame = frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = gray.mean()
        if brightness < min_brightness:
            continue  # ⬅️ skip fade-out / black frames

        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness >= sharpness_threshold:
            last_clear_frame = frame
            break

    cap.release()

    final_frame = last_clear_frame if last_clear_frame is not None else last_valid_frame

    if final_frame is None:
        raise RuntimeError("No readable frame found in video.")

    cv2.imwrite(output_path, final_frame)
    return output_path
