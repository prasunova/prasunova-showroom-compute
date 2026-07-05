#!/usr/bin/env python3
"""
update_room_status.py — Sets SH_rooms.has_masks = true in Supabase via REST.
Called by precompute workflow after masks are validated in R2.
"""
import argparse
import os
import sys
import requests


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--room-id", required=True)
    args = p.parse_args()

    url  = os.environ['SUPABASE_URL'].rstrip('/') + '/rest/v1/SH_rooms'
    key  = os.environ['SUPABASE_KEY']
    headers = {
        "apikey":        key,
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal",
    }
    res = requests.patch(
        url,
        json={"has_masks": True},
        headers=headers,
        params={"id": f"eq.{args.room_id}"},
    )
    if res.status_code not in (200, 204):
        print(f"[ERROR] Supabase update failed: {res.status_code} {res.text}")
        sys.exit(1)
    print(f"[OK] SH_rooms.has_masks=true for {args.room_id}")