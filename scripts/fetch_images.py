#!/usr/bin/env python3
"""Fetch ONLY the nuScenes images a subset of DriveBench needs — via HTTP range
requests against the gated DriveLM image zip on HuggingFace, so you avoid the
full 705 MB download.

Prereqs (one-time, on your account — cannot be automated):
  1. Create a free HF account, visit https://huggingface.co/datasets/OpenDriveLab/DriveLM
     and click "Agree to access" (gated: auto, instant).
  2. Create a token at https://huggingface.co/settings/tokens and:
       export HF_TOKEN=hf_xxx

Then, e.g. grab images for the first 15 keyframes:
  python scripts/fetch_images.py \
      --qa-file data/raw/drivebench/drivebench-test.json \
      --frames 15 --out-root data/raw/drivebench

Images are written under <out-root>/data/nuscenes/samples/<CAM>/<file>.jpg, the
layout the curation adapter expects (the basename-fallback also covers other
layouts). Re-running skips files already present.
"""

import argparse
import json
import os
import sys

import _bootstrap  # noqa: F401

ZIP_URL = ("https://huggingface.co/datasets/OpenDriveLab/DriveLM/resolve/main/"
           "drivelm_nus_imgs_{split}.zip")


def needed_images(qa_file, n_frames):
    """Return {basename: arena_relpath} for the first n_frames distinct frames."""
    data = json.load(open(qa_file))
    wanted, seen_frames = {}, []
    for rec in data:
        ft = rec.get("frame_token")
        if ft not in seen_frames:
            if len(seen_frames) >= n_frames:
                continue
            seen_frames.append(ft)
        if ft in seen_frames:
            for _cam, relpath in (rec.get("image_path") or {}).items():
                wanted[os.path.basename(relpath)] = relpath
    return wanted


def signed_url(token, split):
    import requests

    url = ZIP_URL.format(split=split)
    headers = {"Authorization": "Bearer {}".format(token)}
    r = requests.get(url, headers=headers, allow_redirects=False, timeout=30)
    if r.status_code in (401, 403):
        sys.exit("Auth failed ({}). Accept the DriveLM gate and enable gated-repo "
                 "access on the token.".format(r.status_code))
    loc = r.headers.get("Location")
    return loc or url  # if no redirect, the resolve URL itself serves bytes


# --- Minimal remote-zip reader using explicit byte ranges only -----------------
# The HF Xet CDN rejects suffix ranges (bytes=-N), so we never use them: we learn
# the total size from a Content-Range probe and address everything explicitly.
import struct  # noqa: E402
import zlib  # noqa: E402


def _get_range(sess, url, start, end):
    import requests  # noqa: F401

    r = sess.get(url, headers={"Range": "bytes={}-{}".format(start, end)}, timeout=60)
    r.raise_for_status()
    return r.content


def _total_size(sess, url):
    r = sess.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
    r.raise_for_status()
    return int(r.headers["Content-Range"].split("/")[1])


def _central_directory(sess, url, total):
    tail = min(1 << 21, total)  # 2 MiB tail holds EOCD + (usually) full CD
    buf = _get_range(sess, url, total - tail, total - 1)
    eocd = buf.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("EOCD not found; CD larger than tail window")
    cd_size = struct.unpack("<I", buf[eocd + 12:eocd + 16])[0]
    cd_off = struct.unpack("<I", buf[eocd + 16:eocd + 20])[0]
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        raise RuntimeError("ZIP64 not supported by this helper")
    win_start = total - tail
    if cd_off >= win_start:
        return buf[cd_off - win_start: cd_off - win_start + cd_size]
    return _get_range(sess, url, cd_off, cd_off + cd_size - 1)


def _parse_cd(cd):
    """basename -> (method, comp_size, local_header_offset)."""
    out, i = {}, 0
    while i + 4 <= len(cd) and cd[i:i + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[i + 10:i + 12])[0]
        comp = struct.unpack("<I", cd[i + 20:i + 24])[0]
        n = struct.unpack("<H", cd[i + 28:i + 30])[0]
        m = struct.unpack("<H", cd[i + 30:i + 32])[0]
        k = struct.unpack("<H", cd[i + 32:i + 34])[0]
        lho = struct.unpack("<I", cd[i + 42:i + 46])[0]
        name = cd[i + 46:i + 46 + n].decode("utf-8", "replace")
        out[os.path.basename(name)] = (method, comp, lho)
        i += 46 + n + m + k
    return out


def _extract_member(sess, url, method, comp_size, lho):
    # Local header is 30 bytes + name + extra; lengths can differ from CD, so read
    # the header first, then the exact compressed payload.
    head = _get_range(sess, url, lho, lho + 29)
    if head[:4] != b"PK\x03\x04":
        raise RuntimeError("bad local header")
    n = struct.unpack("<H", head[26:28])[0]
    m = struct.unpack("<H", head[28:30])[0]
    data_start = lho + 30 + n + m
    raw = _get_range(sess, url, data_start, data_start + comp_size - 1)
    return raw if method == 0 else zlib.decompress(raw, -15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-file", required=True)
    ap.add_argument("--frames", type=int, default=15, help="Distinct keyframes to fetch")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--out-root", default="data/raw/drivebench")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN (https://huggingface.co/settings/tokens) after accepting the DriveLM gate.")

    wanted = needed_images(args.qa_file, args.frames)
    print("Subset needs {} images across {} frames".format(len(wanted), args.frames))

    import requests

    url = signed_url(token, args.split)
    sess = requests.Session()
    total = _total_size(sess, url)
    print("Reading {} central directory ({:.0f} MB zip)...".format(args.split, total / 1e6))
    index = _parse_cd(_central_directory(sess, url, total))
    print("  {} entries in zip".format(len(index)))

    n_ok, n_miss = 0, 0
    for base, relpath in wanted.items():
        dst = os.path.join(args.out_root, relpath)
        if os.path.exists(dst):
            n_ok += 1
            continue
        entry = index.get(base)
        if not entry:
            n_miss += 1
            continue
        method, comp, lho = entry
        data = _extract_member(sess, url, method, comp, lho)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as out:
            out.write(data)
        n_ok += 1
        if n_ok % 20 == 0:
            print("  {} fetched...".format(n_ok))
    print("Done: {} present/fetched, {} not found in {} zip".format(n_ok, n_miss, args.split))
    if n_miss:
        print("  ({} frames not in this split — try --split train)".format(n_miss))


if __name__ == "__main__":
    main()
