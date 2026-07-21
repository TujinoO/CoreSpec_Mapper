from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import sys
import traceback


def _self_test(output_path: str) -> int:
    record: dict[str, object] = {"status": "failed"}
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import numpy as np
        import torch
        from PIL import Image
        from PySide6.QtWidgets import QApplication

        from corespec_mapper import __version__
        from corespec_mapper.foreground_model import inspect_foreground_model, run_foreground_model
        from corespec_mapper.spectral_db import V5SpectralDatabase
        from corespec_mapper.spectral_v5 import load_v5_catalog

        app = QApplication.instance() or QApplication([])
        catalog = load_v5_catalog()
        with V5SpectralDatabase() as database:
            reference_counts = database.candidate_counts()
        model = inspect_foreground_model()
        if not model.get("available"):
            raise RuntimeError("Foreground model unavailable: " + "; ".join(model.get("missing", ())))
        with TemporaryDirectory() as temporary:
            rgb_path = Path(temporary) / "self_test_rgb.png"
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[..., 0] = np.arange(32, dtype=np.uint8)[None, :] * 8
            image[..., 1] = np.arange(32, dtype=np.uint8)[:, None] * 8
            image[..., 2] = 128
            Image.fromarray(image).save(rgb_path)
            mask, inference = run_foreground_model(rgb_path, (32, 32))
        record = {
            "status": "passed",
            "application_version": __version__,
            "qt": "available",
            "torch_version": torch.__version__,
            "torch_device": inference.get("device"),
            "catalog_minerals": len(catalog.get("minerals", {})),
            "reference_minerals": len(reference_counts),
            "foreground_model": {
                "available": True,
                "model_version": model.get("model_version"),
                "inference_shape": list(mask.shape),
                "foreground_pixels": int(np.count_nonzero(mask)),
            },
        }
        app.quit()
    except Exception as exc:
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc()
    Path(output_path).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if record.get("status") == "passed" else 1


def main() -> int:
    if "--self-test" in sys.argv:
        try:
            index = sys.argv.index("--self-test-output")
            output_path = sys.argv[index + 1]
        except (ValueError, IndexError):
            output_path = str(Path.cwd() / "corespec_mapper_self_test.json")
        return _self_test(output_path)
    from corespec_mapper.desktop_v5 import main as desktop_main

    return desktop_main()


if __name__ == "__main__":
    raise SystemExit(main())
