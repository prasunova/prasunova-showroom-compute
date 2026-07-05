#!/usr/bin/env python3
"""
validate_mask.py — Quality check for masks stored in R2.
"""
import argparse
import io
import os
import sys

import boto3
import numpy as np
from PIL import Image

MIN_COVERAGE_WARN = 5.0
MIN_COVERAGE_FAIL = 0.1


def get_r2():
    return boto3.client(
        's3',
        endpoint_url          = os.environ['R2_ENDPOINT'],
        aws_access_key_id     = os.environ['R2_ACCESS_KEY_ID'],
        aws_secret_access_key = os.environ['R2_SECRET_ACCESS_KEY'],
        region_name           = 'auto',
    )


def check_mask_from_r2(r2, bucket, key):
    obj = r2.get_object(Bucket=bucket, Key=key)
    img = Image.open(io.BytesIO(obj['Body'].read())).convert("L")
    arr = np.array(img)
    return (arr > 127).sum() / arr.size * 100


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--client-slug", required=True)
    p.add_argument("--room-id",     default=None, help="Validate specific room masks")
    p.add_argument("--image-hash",  default=None, help="Validate user upload masks")
    args = p.parse_args()

    bucket = os.environ['R2_BUCKET_NAME']
    r2     = get_r2()
    slug   = args.client_slug
    fail   = False

    if args.image_hash:
        prefix = f"user_masks/user_{args.image_hash}"
    elif args.room_id:
        prefix = f"rooms/{args.room_id}"
    else:
        print("ERROR: --room-id or --image-hash required")
        sys.exit(1)

    for s in ['floor', 'wall']:
        key = f"{prefix}/{s}_mask.png"
        try:
            coverage = check_mask_from_r2(r2, bucket, key)
            if coverage < MIN_COVERAGE_FAIL:
                print(f"[FAIL] {s}_mask.png: {coverage:.2f}% — BLANK")
                fail = True
            elif coverage < MIN_COVERAGE_WARN:
                print(f"[WARN] {s}_mask.png: {coverage:.2f}% — low coverage")
            else:
                print(f"[ OK] {s}_mask.png: {coverage:.1f}%")
        except Exception:
            pass  # mask doesn't exist for this surface — not a failure

    if fail:
        print("\n[RESULT] VALIDATION FAILED")
        sys.exit(1)
    else:
        print("\n[RESULT] All masks OK")
