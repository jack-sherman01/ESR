"""Start ASPIRE's SAM3 HTTP service from a pinned, locally verified checkpoint.

No GPT calls. Uses upstream text and point request handlers without copying
their implementation. CPU checkpoint loading avoids allocating weights on the
GPU before the chosen storage precision has been applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import uvicorn
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from cap_harness.providers.sam3 import service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--startup-report", type=Path, required=True)
    parser.add_argument("--precision", choices=("float32", "bf16-backbone", "float32-offload"), default="float32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8114)
    parser.add_argument("--retain-public-topk", action="store_true")
    args = parser.parse_args()
    if args.startup_report.exists():
        parser.error("Startup report already exists; choose a new path")
    receipt = json.loads(args.receipt.read_text())
    info = receipt["files"]["sam3.pt"]
    checkpoint = Path(info["path"])
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != info["sha256"]:
        raise RuntimeError("Checkpoint does not match download receipt")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This deployment requires an accessible CUDA GPU")
    torch.cuda.set_device(torch.device(args.device))
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    started = time.monotonic()
    model = build_sam3_image_model(device="cpu", checkpoint_path=str(checkpoint),
                                  load_from_HF=False, enable_inst_interactivity=True)
    if args.precision == "bf16-backbone":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support bfloat16")
        # Upstream decoder FFNs deliberately disable autocast and need FP32
        # weights. Reduce storage only for backbone linear/convolution layers;
        # retain normalization, embeddings, decoder and heads in their original
        # precision. This is an explicitly recorded local memory adaptation.
        for module in model.backbone.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d,
                                   torch.nn.Conv2d, torch.nn.Conv3d)):
                module.to(dtype=torch.bfloat16)
    model = model.eval()
    offload = None
    if args.precision == "float32-offload":
        from aspire_sam3_offload import stage_backbone
        offload = stage_backbone(model, args.device)
        model._backbone_offload = offload
    else:
        model = model.to(args.device)
    service._DEVICE = args.device
    service._MODEL = model
    if args.retain_public_topk:
        from aspire_sam3_topk import retain_public_topk
        original_grounding = model.forward_grounding
        def bounded_grounding(*positional, **named):
            return retain_public_topk(original_grounding(*positional, **named), k=5)
        model.forward_grounding = bounded_grounding
    service._PROCESSOR = Sam3Processor(model, device=args.device, confidence_threshold=0.0)
    dtypes = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        dtypes[key] = dtypes.get(key, 0) + parameter.numel()
    report = {"status": "loaded", "model_repo": receipt["repo_id"],
              "model_revision": receipt["revision"], "checkpoint_sha256": info["sha256"],
              "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "parameter_storage": args.precision,
              "parameter_dtypes": dtypes,
              "parameter_devices": {device: sum(p.numel() for p in model.parameters() if p.device.type == device)
                                    for device in ("cpu", "cuda")},
              "staged_parameter_elements": 0 if offload is None else offload.parameter_elements,
              "inference_autocast": "bfloat16 (upstream handlers)",
              "resolution": 1008, "instance_interactivity": True,
              "retain_public_topk_before_upsample": args.retain_public_topk,
              "parameters": sum(p.numel() for p in model.parameters()),
              "load_seconds": time.monotonic() - started,
              "cuda_allocated_bytes": torch.cuda.memory_allocated(),
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "bind": f"127.0.0.1:{args.port}", "gpt_api_called": False}
    args.startup_report.parent.mkdir(parents=True, exist_ok=True)
    with args.startup_report.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)
    uvicorn.run(service.app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
