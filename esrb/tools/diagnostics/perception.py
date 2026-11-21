"""Exercise real SAM3 -> typed ASPIRE client -> RGB-D localization, no LLM.

This checks service integration, not benchmark accuracy or task completion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from cap_harness.api import CapApi
from cap_harness.libero.adapter import LiberoAdapter
from cap_harness.providers.sam3 import Sam3Provider


def main(args=None) -> int:
    from esrb.cli.options import options_tools_diagnose_perception
    parser=argparse.ArgumentParser()
    options_tools_diagnose_perception(parser)
    args=parser.parse_args() if args is None else args
    args.out.mkdir(parents=True, exist_ok=False)
    adapter = LiberoAdapter(camera_width=512, camera_height=512)
    report = {"kind": "real_sam3_libero_perception_smoke", "status": "failed",
              "gpt_api_called": False, "task_success_evaluated": False,
              "esrb_safety_evaluated": False, "segmentation_accuracy_evaluated": False,
              "suite": "libero_object_swap", "task_id": 0, "seed": 1,
              "init_mode": "saved", "query": args.query, "provider_url": args.url}
    try:
        provider = Sam3Provider(base_url=args.url, timeout_s=(10, 240), max_retries=0)
        if not provider.health():
            raise RuntimeError("SAM3 health check failed")
        report["service_health"] = True
        observation = adapter.reset("libero_object_swap:0", seed=1)
        camera = observation.cameras["agentview"]
        Image.fromarray(camera.rgb).save(args.out / "scene.png")
        api = CapApi(adapter, segmentation_provider=provider)
        started = time.monotonic()
        localized = api.localize_object(args.query, camera_name="agentview")
        report["text_and_localization_seconds"] = time.monotonic() - started
        if not localized.ok:
            raise RuntimeError(f"Localization failed: {localized.error}")
        if localized.segmentation is None or localized.point_cloud is None or localized.geometry is None:
            raise RuntimeError("Missing localization intermediates")
        segmentation = localized.segmentation
        points = localized.point_cloud.points
        if points.shape[0] == 0 or not np.all(np.isfinite(points)):
            raise RuntimeError("Invalid 3D points")
        np.save(args.out / "text_mask.npy", segmentation.mask)
        np.save(args.out / "object_points_robot_base.npy", points)
        Image.fromarray(segmentation.mask.astype(np.uint8) * 255).save(args.out / "text_mask.png")
        report["text"] = {"score": segmentation.score, "pixels": int(segmentation.mask.sum()),
                          "box_xyxy": segmentation.box_xyxy.tolist() if segmentation.box_xyxy is not None else None}
        report["localization"] = {"frame": localized.geometry.frame,
                                  "points": len(points),
                                  "center_m": localized.geometry.pose.position.tolist(),
                                  "extents_m": localized.geometry.extents.tolist()}
        if not args.skip_point:
            ys, xs = np.nonzero(segmentation.mask)
            if len(xs) == 0:
                raise RuntimeError("Text mask empty")
            index = len(xs) // 2
            click = np.array([[float(xs[index]), float(ys[index])]])
            started = time.monotonic()
            point_result = api.segment_points("agentview", click)
            report["point_seconds"] = time.monotonic() - started
            if not point_result.ok or not point_result.segmentations:
                raise RuntimeError(f"Point segmentation failed: {point_result.error}")
            best = point_result.segmentations[0]
            if not np.any(best.mask):
                raise RuntimeError("Point mask empty")
            np.save(args.out / "point_mask.npy", best.mask)
            Image.fromarray(best.mask.astype(np.uint8) * 255).save(args.out / "point_mask.png")
            report["point"] = {"click_xy": click[0].tolist(), "score": best.score,
                               "pixels": int(best.mask.sum())}
        report["status"] = "passed"
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        try:
            adapter.close()
        except Exception:
            report["status"] = "failed"
            report["close_error"] = traceback.format_exc()
        (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1
