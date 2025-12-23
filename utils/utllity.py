import os
import cv2
from dotenv import load_dotenv, dotenv_values, find_dotenv

# ---- Global state ----
_ENV_LOADED = False
_ENV_NAME = None
_ENV_VARS = {}

def load_environment(env_key: str = "stag", force_reload: bool = False) -> dict:
    """
    Load environment variables based on env_key ('stag' or 'prod').

    Args:
        env_key (str): 'stag' or 'prod'
        force_reload (bool): reload even if already loaded

    Returns:
        dict: loaded environment variables
    """
    global _ENV_LOADED, _ENV_NAME, _ENV_VARS

    if _ENV_LOADED and not force_reload:
        return _ENV_VARS

    if env_key == "stag":
        env_path = find_dotenv("stag.env", usecwd=True)
    elif env_key == "prod":
        env_path = find_dotenv("prod.env", usecwd=True)
    else:
        raise ValueError("env_key must be 'stag' or 'prod'")

    if not env_path:
        raise FileNotFoundError(f"Environment file not found for '{env_key}'")

    load_dotenv(dotenv_path=env_path, override=True)
    _ENV_VARS = dotenv_values(env_path)

    if not _ENV_VARS:
        raise RuntimeError(f"No environment variables found in {env_path}")

    _ENV_LOADED = True
    _ENV_NAME = env_key

    print(f"✅ Loaded '{env_key}' environment from: {env_path}")
    return _ENV_VARS


def get_env(key: str, default=None):
    """Safe env getter after environment is loaded."""
    return os.getenv(key, default)


def current_env():
    """Returns currently loaded environment name."""
    return _ENV_NAME


def classify_env(value: str, default: str = "stag") -> str:
    """
    Classify environment from a string.

    Examples:
        'messproof-staging'    -> 'stag'
        'messproof-production' -> 'prod'
    """
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
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_clear_frame = None

    for i in range(total_frames - 1, -1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        if sharpness >= sharpness_threshold:
            last_clear_frame = frame
            break

    cap.release()

    if last_clear_frame is None:
        raise RuntimeError("No clear frame found, lower threshold.")

    cv2.imwrite(output_path, last_clear_frame)
    return output_path

