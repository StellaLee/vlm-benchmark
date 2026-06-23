#!/usr/bin/env python3
"""Generate a tiny synthetic DriveBench-format fixture so the pipeline runs
end-to-end without downloading the real dataset. NOT real driving data — solid
color images and toy MCQs purely to exercise curate -> infer -> evaluate."""

import json
import os

from PIL import Image

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "sample")
IMG_DIR = os.path.join(OUT_DIR, "images")

CAMERAS = {
    "CAM_FRONT": (90, 110, 140),
    "CAM_FRONT_LEFT": (70, 90, 120),
    "CAM_FRONT_RIGHT": (110, 90, 70),
    "CAM_BACK": (60, 70, 80),
}

TASKS = ["perception", "prediction", "planning", "behavior"]


def make_images():
    os.makedirs(IMG_DIR, exist_ok=True)
    for cam, color in CAMERAS.items():
        img = Image.new("RGB", (320, 180), color)
        img.save(os.path.join(IMG_DIR, "{}.jpg".format(cam)), "JPEG")


def make_records():
    images = {cam: "images/{}.jpg".format(cam) for cam in CAMERAS}
    recs = []
    for i in range(12):
        task = TASKS[i % len(TASKS)]
        recs.append({
            "id": "sample_{:03d}".format(i),
            "task": task,
            "format": "mcq",
            "split": "clean",
            "question": (
                "In scene {sid}, what is the moving status of "
                "<c1,CAM_FRONT,{x},90>?".format(sid=i, x=100 + i * 7)
                if task == "perception"
                else "In scene {sid}, based on the scene, what should the ego "
                     "vehicle do next?".format(sid=i)
            ),
            "options": ["Keep going at the same speed", "Decelerate",
                        "Turn left", "Stop"],
            "answer": ["A", "B", "C", "D"][i % 4],
            "images": images,
            "scene_token": "scene_{:03d}".format(i),
        })
    # a couple of corrupted-split records to verify split filtering
    for i in range(2):
        recs.append({
            "id": "corrupt_{:03d}".format(i), "task": "perception", "format": "mcq",
            "split": "fog", "question": "Toy corrupted question?",
            "options": ["A thing", "Another"], "answer": "A", "images": images,
        })
    return recs


def main():
    make_images()
    recs = make_records()
    path = os.path.join(OUT_DIR, "drivebench_sample.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2)
    print("Wrote {} records + {} images under {}".format(len(recs), len(CAMERAS), OUT_DIR))


if __name__ == "__main__":
    main()
