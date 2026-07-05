#!/usr/bin/env python3
"""
notify_ready.py — Writes ready.json to R2.
The poll endpoint checks R2 (fast HEAD) — no GitHub API calls per poll request.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import boto3


def get_r2():
    return boto3.client(
        's3',
        endpoint_url          = os.environ['R2_ENDPOINT'],
        aws_access_key_id     = os.environ['R2_ACCESS_KEY_ID'],
        aws_secret_access_key = os.environ['R2_SECRET_ACCESS_KEY'],
        region_name           = 'auto',
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hash",        required=True)
    p.add_argument("--client-slug", default=None)  # kept for compat, unused in paths
    args = p.parse_args()

    bucket = os.environ['R2_BUCKET_NAME']
    r2     = get_r2()
    prefix = f"user_masks/user_{args.hash}"

    # Discover which masks were written to R2
    surfaces = []
    for s in ['floor', 'wall']:
        try:
            r2.head_object(Bucket=bucket, Key=f"{prefix}/{s}_mask.png")
            surfaces.append(s)
        except Exception:
            pass

    ready = {
        "hash":      args.hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "surfaces":  surfaces,
    }

    body = json.dumps(ready, indent=2).encode()
    r2.put_object(Bucket=bucket, Key=f"{prefix}/ready.json", Body=body, ContentType="application/json")
    print(f"[READY] Uploaded to R2: {prefix}/ready.json")
    print(f"[READY] Surfaces: {surfaces}")
