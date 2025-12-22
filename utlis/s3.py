import boto3
import os
from utlis.utllity import get_env

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=get_env("LAMBDA_ACCESS_KEY_ID"),
        aws_secret_access_key=get_env("LAMBDA_SECRET_ACCESS_KEY"),
        region_name=get_env("LAMBDA_DEFAULT_REGION"),
    )

def download_image(s3_uri, local_path):
    if s3_uri.startswith("http"):
        import requests
        r = requests.get(s3_uri, timeout=30)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        return

    # s3://bucket/key
    _, _, bucket, *key = s3_uri.split("/")
    key = "/".join(key)

    s3 = get_s3_client()
    s3.download_file(bucket, key, local_path)

def upload_video(local_path, bucket, key):
    s3 = get_s3_client()
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"
