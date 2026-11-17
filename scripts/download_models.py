#!/usr/bin/env python3
"""
Helper utility to pre-download and cache NVIDIA Parakeet-TDT and Sortformer
weights for 100% offline execution.
"""

from __future__ import annotations

import argparse
from aslytics.config import MODEL_REGISTRY
from aslytics.models.parakeet_engine import ParakeetOnDeviceEngine


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-download on-device NVIDIA Parakeet weights for offline group captioning."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="parakeet-tdt-0.6b-v3",
        help="Model key to download (default: parakeet-tdt-0.6b-v3)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "onnx", "mlx", "nemo"],
        default="auto",
        help="Runtime backend to prepare",
    )
    args = parser.parse_args()

    spec = MODEL_REGISTRY[args.model]
    backends = ParakeetOnDeviceEngine.detect_available_backends()

    print(f"Model      : {spec.name} ({spec.parameters})")
    print(f"NeMo ID    : {spec.nemo_id}")
    print(f"ONNX ID    : {spec.onnx_id}")
    print(f"MLX ID     : {spec.mlx_id or 'N/A'}")
    print("\nDetected Local Runtimes:")
    for name, info in backends.items():
        status = "INSTALLED" if info["installed"] else f"Not installed ({info['install_cmd']})"
        print(f"  - {name:10s}: {status}")

    engine = ParakeetOnDeviceEngine()
    if engine.active_backend == "simulation":
        print(
            "\nNote: Running in zero-download simulation mode. "
            "Install one of the hardware runtimes above (`pip install -e '.[onnx]'` or "
            "`pip install -e '.[mlx]'`) to download neural weights."
        )
    else:
        print(f"\nDownloading and initializing {spec.key} via [{engine.active_backend}]...")
        engine.load_model()
        print("Model cached and ready for 100% offline use!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
