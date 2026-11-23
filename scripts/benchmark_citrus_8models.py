#!/usr/bin/env python3
"""Evaluate Citrus ONNX models on Raspberry Pi 5 and organize results.

Expected source files in ~/:
  8 ONNX model files (FP32 / INT8 / structured pruning / pruning+INT8)
  Deploy JSON files for each model, or its matching non-INT8 parent
  pi_test/dataset/metadata/splits.csv
  pi_test/dataset/processed/... (for image-using models)

Usage (after source files are copied to ~/):
  source ~/citrus-venv/bin/activate
  python ~/benchmark_citrus_8models.py

Outputs in ~/citrus_benchmark_8models/; existing source files are never modified.
Eight variants of lightweight image CNN and image+weight fusion are evaluated.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

MODEL_CHOICES = (
    "custom_image", "custom_image_int8", "custom_image_pruned", "custom_image_pruned_int8",
    "custom_fusion", "custom_fusion_int8", "custom_fusion_pruned", "custom_fusion_pruned_int8",
)
DEFAULT_HOME = Path.home()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Citrus ONNX models on Raspberry Pi")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES,
                        default=list(MODEL_CHOICES))
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_HOME / "pi_test")
    parser.add_argument("--output", type=Path, default=DEFAULT_HOME / "citrus_benchmark_8models")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--single", choices=MODEL_CHOICES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.threads < 1:
        parser.error("--warmup >= 0, --repeats >= 1, --threads >= 1 required")
    return args


def write_csv(path: Path, fields, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for item in items:
            wr.writerow(item)


def prepare(args):
    root = args.output.expanduser().resolve()
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    test_dir = args.test_dir.expanduser().resolve()
    csv_path = test_dir / "dataset/metadata/splits.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Test split CSV does not exist: {csv_path}")
    link = root / "data/pi_test"
    if not link.exists() and not link.is_symlink():
        link.symlink_to(test_dir, target_is_directory=True)

    for name in args.models:
        model_file = f"{name}.onnx"
        src = DEFAULT_HOME / model_file
        dst = models_dir / model_file
        if src.is_file():
            shutil.copy2(src, dst)
        elif not dst.is_file():
            raise FileNotFoundError(f"Missing {src}. Copy it to ~/ first.")

        # INT8 shares preprocessing with its corresponding FP32 model.
        # Prefer explicit <name>_deploy.json if it exists, otherwise the parent JSON.
        config_file = f"{name}_deploy.json"
        parent_name = name.removesuffix("_int8")
        config_candidates = [DEFAULT_HOME / config_file]
        if parent_name != name:
            config_candidates.append(DEFAULT_HOME / f"{parent_name}_deploy.json")
        config_dst = models_dir / config_file
        config_src = next((p for p in config_candidates if p.is_file()), None)
        if config_src is not None:
            shutil.copy2(config_src, config_dst)
        elif not config_dst.is_file():
            raise FileNotFoundError(
                f"Missing {config_file}; also checked {parent_name}_deploy.json in ~/"
            )
    return root


def load_test_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        expected = {"image_path", "fruit_id", "class", "weight_g", "split"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"splits.csv missing columns: {expected - set(reader.fieldnames or [])}")
        rows = [row for row in reader if row["split"] == "test"]
    if not rows:
        raise ValueError("splits.csv has no test rows")
    fruit_map = {}
    for row in rows:
        key = row["fruit_id"]
        if key in fruit_map:
            first = fruit_map[key]
            if (first["class"], first["weight_g"]) != (row["class"], row["weight_g"]):
                raise ValueError(f"Inconsistent fruit label/weight: {key}")
        else:
            fruit_map[key] = row
    return rows, len(fruit_map)


def run_one(args, root: Path, name: str):
    # Keep model evaluations isolated in separate Python processes so peak RSS
    # of one model does not include model sessions from the others.
    import numpy as np
    import onnxruntime as ort

    model_path = root / "models" / f"{name}.onnx"
    config_path = root / "models" / f"{name}_deploy.json"
    test_dir = args.test_dir.expanduser().resolve()
    test_rows, n_fruits = load_test_rows(test_dir / "dataset/metadata/splits.csv")
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    classes = config["class_names"]
    if len(classes) != 3:
        raise ValueError("Expected exactly 3 classes")
    w_mean = float(config["weight_mean"])
    w_std = float(config["weight_std"])
    if w_std <= 0:
        raise ValueError("weight_std must be positive")
    uses_standardized_weight = bool(config.get("weight_is_standardized", True))
    image_size = tuple(config.get("image_size", [224, 224]))
    image_mean = np.array(config["image_mean"], dtype=np.float32).reshape(1, 1, 3)
    image_std = np.array(config["image_std"], dtype=np.float32).reshape(1, 1, 3)

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Initial peak-RSS measurement: Python libraries + configuration are counted.
    # This is deliberately labeled "process peak RSS", not memory for the model alone.
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    inputs = {tensor.name for tensor in session.get_inputs()}
    expected_inputs = {"image", "weight"} if name.startswith("custom_fusion") else {"image"}
    if inputs != expected_inputs:
        raise ValueError(f"{name} ONNX inputs {inputs} != expected {expected_inputs}")

    def make_input(row):
        feed = {}
        if "image" in inputs:
            from PIL import Image
            file_path = test_dir / row["image_path"]
            if not file_path.is_file():
                raise FileNotFoundError(f"Missing test image: {file_path}")
            with Image.open(file_path) as image:
                image = image.convert("RGB")
                if image.size != image_size:
                    raise ValueError(
                        f"Expected processed {image_size} image, got {image.size}: {file_path}"
                    )
                array = np.array(image, dtype=np.float32) / np.float32(255.0)
            array = (array - image_mean) / image_std
            array = np.transpose(array, (2, 0, 1))[None, :, :, :].copy()
            feed["image"] = array
        if "weight" in inputs:
            value = float(row["weight_g"])
            if uses_standardized_weight:
                value = (value - w_mean) / w_std
            feed["weight"] = np.array([[value]], dtype=np.float32)
        return feed

    first = make_input(test_rows[0])
    for _ in range(args.warmup):
        session.run(None, first)

    prediction_rows = []
    latency_rows = []
    correct = 0
    # Class order from deploy JSON is authoritative.
    matrix = [[0 for _ in classes] for _ in classes]
    timings = []

    for repetition in range(args.repeats):
        for index, row in enumerate(test_rows):
            feed = make_input(row)  # excluded from pure ONNX latency
            t0 = time.perf_counter_ns()
            logits = session.run(None, feed)[0]
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            timings.append(elapsed_ms)
            latency_rows.append({
                "repeat": repetition + 1,
                "sample_index": index + 1,
                "fruit_id": row["fruit_id"],
                "latency_ms": round(elapsed_ms, 6),
            })
            if repetition != 0:
                continue
            predicted_index = int(np.argmax(logits[0]))
            predicted_class = classes[predicted_index]
            truth = row["class"]
            if truth not in classes:
                raise ValueError(f"Unknown class '{truth}' in splits.csv")
            okay = int(predicted_class == truth)
            correct += okay
            matrix[classes.index(truth)][predicted_index] += 1
            prediction_rows.append({
                "fruit_id": row["fruit_id"],
                "image_path": row["image_path"],
                "weight_g": row["weight_g"],
                "true_class": truth,
                "predicted_class": predicted_class,
                "correct": okay,
            })

    sorted_t = sorted(timings)
    p95_index = max(0, int(np.ceil(0.95 * len(sorted_t))) - 1)
    result = {
        "model": name,
        "model_family": "image+weight" if name.startswith("custom_fusion") else "image-only",
        "optimization": (
            "pruning+INT8" if name.endswith("_pruned_int8") else
            "INT8" if name.endswith("_int8") else
            "pruning" if name.endswith("_pruned") else "FP32 baseline"
        ),
        "accuracy_unit": "image",
        "n_accuracy_samples": len(prediction_rows),
        "n_unique_test_fruits": n_fruits,
        "n_test_images": len(test_rows),
        "correct": correct,
        "accuracy_percent": round(100.0 * correct / len(prediction_rows), 4),
        "n_latency_runs": len(timings),
        "warmup_runs": args.warmup,
        "repeats": args.repeats,
        "batch_size": 1,
        "cpu_threads": args.threads,
        "avg_latency_ms": round(statistics.mean(timings), 6),
        "median_latency_ms": round(statistics.median(timings), 6),
        "p95_latency_ms": round(sorted_t[p95_index], 6),
        "min_latency_ms": round(min(timings), 6),
        "max_latency_ms": round(max(timings), 6),
        "peak_process_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3),
        "onnx_model_size_mb": round(model_path.stat().st_size / 1024**2, 4),
        "avg_power_w": None,
        "power_note": "Not measured: external power meter required",
        "python": sys.version.split()[0],
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
    }

    folder = root / "results" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(folder / "predictions.csv",
              ["fruit_id", "image_path", "weight_g", "true_class", "predicted_class", "correct"],
              prediction_rows)
    write_csv(folder / "latencies.csv", ["repeat", "sample_index", "fruit_id", "latency_ms"],
              latency_rows)
    cm_rows = []
    for i, true_class in enumerate(classes):
        cm_rows.append({"true_class": true_class,
                        **{f"predicted_{classes[j]}": matrix[i][j] for j in range(len(classes))}})
    write_csv(folder / "confusion_matrix.csv",
              ["true_class"] + [f"predicted_{c}" for c in classes], cm_rows)
    print(f"{name}: accuracy={result['accuracy_percent']:.2f}% "
          f"({result['n_accuracy_samples']} {result['accuracy_unit']}s), "
          f"latency={result['avg_latency_ms']:.3f} ms, "
          f"peak process RSS={result['peak_process_rss_mb']:.1f} MiB", flush=True)


def main():
    args = parse_args()
    root = args.output.expanduser().resolve()
    if args.single:
        run_one(args, root, args.single)
        return

    root = prepare(args)
    for name in args.models:
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--single", name, "--test-dir", str(args.test_dir),
                   "--output", str(root), "--warmup", str(args.warmup),
                   "--repeats", str(args.repeats), "--threads", str(args.threads)]
        subprocess.run(command, check=True)

    metrics = []
    for name in args.models:
        path = root / "results" / name / "metrics.json"
        metrics.append(json.loads(path.read_text(encoding="utf-8")))

    fields = [
        "model", "model_family", "optimization", "accuracy_unit", "n_accuracy_samples", "accuracy_percent",
        "n_latency_runs", "avg_latency_ms", "median_latency_ms", "p95_latency_ms",
        "peak_process_rss_mb", "onnx_model_size_mb", "avg_power_w"
    ]
    write_csv(root / "results" / "comparison.csv", fields, metrics)
    (root / "results" / "README.txt").write_text(
        "Eight Citrus ONNX variants evaluated on Raspberry Pi 5\n"
        "- comparison.csv: one row per model; avg_power_w blank because not measured\n"
        "- Each <model>/metrics.json: all model metrics and measurement context\n"
        "- All 8 variants: FP32, INT8, structural pruning, structural pruning+INT8\n"
        "- Each <model>/predictions.csv: one row per evaluated sample\n"
        "- Each <model>/latencies.csv: pure ONNX inference times (no image I/O or preprocessing)\n"
        "- Each <model>/confusion_matrix.csv: class confusion counts\n"
        "- Image CNN/fusion accuracy: all test image rows (typically 210)\n"
        "- Latency: each test image once per repeat at batch size 1\n"
        "- Memory: peak RSS of isolated Python process, not model-only RAM\n"
        "- Fusion receives both image and standardized weight; image CNN receives only image\n"
        "- CPU frequency, thermal state, and system load may affect timing\n",
        encoding="utf-8"
    )
    print(f"\nAll results saved to: {root / 'results'}")
    print("Summary:", root / "results" / "comparison.csv")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
