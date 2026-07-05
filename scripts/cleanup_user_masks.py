#!/usr/bin/env python3
"""
cleanup_user_masks.py — Deletes user_masks entries from R2 older than RETENTION_DAYS.
Finds user_masks/user_*/ready.json, checks LastModified, deletes the whole prefix if stale.
"""
import os
import sys
from datetime import datetime, timezone, timedelta

import boto3
from botocore.exceptions import ClientError

RETENTION_DAYS = int(os.environ.get('RETENTION_DAYS', '30'))


def get_r2():
    return boto3.client(
        's3',
        endpoint_url          = os.environ['R2_ENDPOINT'],
        aws_access_key_id     = os.environ['R2_ACCESS_KEY_ID'],
        aws_secret_access_key = os.environ['R2_SECRET_ACCESS_KEY'],
        region_name           = 'auto',
    )


def list_ready_keys(r2, bucket):
    """Yield (hash, last_modified) for every user_masks/user_*/ready.json in R2."""
    paginator = r2.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix='user_masks/', Delimiter='/'):
        for prefix_obj in page.get('CommonPrefixes', []):
            prefix = prefix_obj['Prefix']          # e.g. user_masks/user_abc12345/
            ready_key = f"{prefix}ready.json"
            try:
                head = r2.head_object(Bucket=bucket, Key=ready_key)
                yield prefix, head['LastModified']
            except ClientError:
                pass  # no ready.json yet — skip


def delete_prefix(r2, bucket, prefix):
    """Delete all objects under a given prefix."""
    paginator = r2.get_paginator('list_objects_v2')
    deleted = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = [{'Key': obj['Key']} for obj in page.get('Contents', [])]
        if objects:
            r2.delete_objects(Bucket=bucket, Delete={'Objects': objects})
            deleted += len(objects)
    return deleted


if __name__ == "__main__":
    bucket   = os.environ['R2_BUCKET_NAME']
    r2       = get_r2()
    cutoff   = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    total    = 0

    print(f"[CLEANUP] Deleting user masks older than {RETENTION_DAYS} days (before {cutoff.date()})")

    for prefix, last_modified in list_ready_keys(r2, bucket):
        if last_modified < cutoff:
            count = delete_prefix(r2, bucket, prefix)
            print(f"[DELETED] {prefix} ({count} objects, last modified {last_modified.date()})")
            total += count
        else:
            print(f"[KEEP]    {prefix} (last modified {last_modified.date()})")

    print(f"[DONE] Deleted {total} objects total.")
