#!/usr/bin/env python3
"""
build_catalog.py — Generates per-client rooms.json.
Checks R2 for mask existence to set hasMasks correctly.
Run by precompute workflow after SAM 3 completes.
"""
import argparse
import glob
import json
import os

import boto3


def get_r2():
    return boto3.client(
        's3',
        endpoint_url          = os.environ['R2_ENDPOINT'],
        aws_access_key_id     = os.environ['R2_ACCESS_KEY_ID'],
        aws_secret_access_key = os.environ['R2_SECRET_ACCESS_KEY'],
        region_name           = 'auto',
    )


def r2_exists(r2, bucket, key):
    try:
        r2.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def build(client_slug):
    r2     = get_r2()
    bucket = os.environ['R2_BUCKET_NAME']
    rooms  = []

    for meta_path in sorted(glob.glob(f"clients/{client_slug}/rooms/*/meta.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        room_id   = meta['id']
        has_floor = r2_exists(r2, bucket, f"clients/{client_slug}/rooms/{room_id}/floor_mask.png")
        has_wall  = r2_exists(r2, bucket, f"clients/{client_slug}/rooms/{room_id}/wall_mask.png")
        entry = {
            "id":       meta['id'],
            "name":     meta['name'],
            "category": meta['category'],
            "surfaces": meta.get('surfaces', ['floor']),
            "hasMasks": has_floor,
        }
        if has_floor:
            entry["masks"] = {"floor": f"clients/{client_slug}/rooms/{room_id}/floor_mask.png"}
        if has_wall:
            entry["masks"]["wall"] = f"clients/{client_slug}/rooms/{room_id}/wall_mask.png"
        rooms.append(entry)
        print(f"[CATALOG] {room_id}: hasMasks={has_floor}")

    out = f"clients/{client_slug}/rooms.json"
    with open(out, "w") as f:
        json.dump(rooms, f, indent=2)
    print(f"[CATALOG] Wrote {len(rooms)} rooms to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--client-slug", required=True)
    args = p.parse_args()
    build(args.client_slug)
