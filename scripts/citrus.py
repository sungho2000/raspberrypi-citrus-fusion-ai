from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ("mandarin", "orange", "hwanggeumhyang")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ARCHITECTURE = "custom_depthwise_separable_cnn_v1"
MODEL_NAMES = ("image", "weight", "fusion", "custom_image", "custom_fusion")

# ---------- 1. Data validation, ID matching and fruit-level split ----------

def prepare(root: Path, seed: int = 42):
    weight_csv = root / "loadcell" / "fruit_weights.csv"
    if not weight_csv.is_file():
        raise FileNotFoundError(f"Weight CSV missing: {weight_csv}")
    with weight_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"fruit_id", "class", "weight"}.issubset(reader.fieldnames or []):
            raise ValueError("Weight CSV columns must be fruit_id,class,weight")
        weights = {}
        for row in reader:
            fid, cls = row["fruit_id"].strip(), row["class"].strip()
            value = float(row["weight"])
            if cls not in CLASSES or not fid.startswith(cls + "_") or fid in weights:
                raise ValueError(f"Invalid/duplicate CSV fruit ID: {fid}")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid weight for {fid}: {value}")
            weights[fid] = (cls, value)

    folder_ids, rows, errors = set(), [], []
    for cls in CLASSES:
        folder = root / "dataset" / "processed" / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"No processed class folder: {folder}")
        for fruit_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            fid = fruit_dir.name
            folder_ids.add(fid)
            if fid not in weights or weights[fid][0] != cls:
                errors.append(f"Unmatched fruit ID: {fid}")
                continue
            images = sorted(fruit_dir.glob("*.png"))
            if len(images) != 14:
                errors.append(f"Expected 14 PNGs for {fid}, found {len(images)}")
            for path in images:
                try:
                    with Image.open(path) as img:
                        img.load()
                        if img.size != (224, 224) or img.mode != "RGB" or img.format != "PNG":
                            errors.append(f"Invalid PNG (must be RGB 224x224): {path}")
                except (OSError, ValueError) as e:
                    errors.append(f"Unreadable PNG: {path}: {e}")
                rows.append({"image_path": path.relative_to(root).as_posix(), "fruit_id": fid,
                             "class": cls, "weight_g": weights[fid][1]})
    for fid in sorted(set(weights) - folder_ids):
        errors.append(f"CSV weight has no image folder: {fid}")
    if errors:
        raise ValueError("Dataset validation failed:\n  " + "\n  ".join(errors[:25]) +
                         (f"\n... ({len(errors)} errors total)" if len(errors) > 25 else ""))
    if len(weights) != 90 or len(rows) != 1260:
        raise ValueError(f"Expected 90 fruit weights and 1260 images, got {len(weights)}, {len(rows)}")

    rng = random.Random(seed)
    split_of = {}
    for cls in CLASSES:
        ids = sorted(fid for fid in folder_ids if weights[fid][0] == cls)
        if len(ids) != 30:
            raise ValueError(f"Expected 30 fruits for {cls}, got {len(ids)}")
        rng.shuffle(ids)
        for split, subset in (("train", ids[:21]), ("val", ids[21:25]), ("test", ids[25:])):
            for fid in subset:
                split_of[fid] = split
    for row in rows:
        row["split"] = split_of[row["fruit_id"]]
    rows.sort(key=lambda r: (r["class"], r["fruit_id"], r["image_path"]))
    target = root / "dataset" / "metadata" / "splits.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Prepared {len(rows)} individual photos / {len(folder_ids)} fruits; "
          "Train=882, Val=168, Test=210")
    print(f"Saved: {target}")


def read_split(root: Path):
    path = root / "dataset" / "metadata" / "splits.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; first run: py scripts/citrus.py prepare")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"image_path", "fruit_id", "class", "weight_g", "split"}.issubset(rows[0]):
        raise ValueError("Missing columns in splits.csv")
    groups, labels, weights = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in rows:
        fid = row["fruit_id"]
        if row["class"] not in CLASSES or row["split"] not in ("train", "val", "test"):
            raise ValueError(f"Invalid class/split: {row}")
        groups[fid].add(row["split"])
        labels[fid].add(row["class"])
        weights[fid].add(float(row["weight_g"]))
    if any(len(groups[k]) != 1 or len(labels[k]) != 1 or len(weights[k]) != 1 for k in groups):
        raise ValueError("Data leakage or inconsistent labels/weights for a fruit ID")
    if set(r["split"] for r in rows) != {"train", "val", "test"}:
        raise ValueError("All 3 splits are required")
    return rows


def weight_stats(rows):
    unique = {r["fruit_id"]: float(r["weight_g"]) for r in rows if r["split"] == "train"}
    values = np.asarray(list(unique.values()), dtype=np.float64)
    if not len(values):
        raise ValueError("Training set has no weights")
    std = float(values.std())
    return float(values.mean()), std if std > 1e-6 else 1.0


# ---------- 2. Models (original 3 + self-designed light CNN 2) ----------

class OriginalNet(nn.Module):
    """Identical module names/shapes to old CitrusNet: works with old .pth."""
    def __init__(self, kind: str, pretrained=False):
        super().__init__()
        self.model_type = kind
        if kind == "weight":
            self.backbone = nn.Identity()
            self.head = nn.Sequential(nn.Linear(1, 32), nn.ReLU(inplace=True),
                                      nn.Linear(32, 16), nn.ReLU(inplace=True), nn.Linear(16, 3))
        else:
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights)
            d = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
            self.head = nn.Sequential(nn.Linear(d + (kind == "fusion"), 128), nn.ReLU(inplace=True),
                                      nn.Dropout(p=.2), nn.Linear(128, 3))

    def forward(self, image, weight):
        if self.model_type == "weight":
            return self.head(weight.view(-1, 1))
        features = self.backbone(image)
        if self.model_type == "fusion":
            features = torch.cat((features, weight.view(-1, 1)), dim=1)
        return self.head(features)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class LightBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            DepthwiseSeparableConv(24, 48, 2), DepthwiseSeparableConv(48, 96, 2),
            DepthwiseSeparableConv(96, 128, 2), DepthwiseSeparableConv(128, 192, 2),
            DepthwiseSeparableConv(192, 192, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, image):
        return self.layers(image)


class LightImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = LightBackbone()
        self.classifier = nn.Sequential(nn.Linear(192, 64), nn.ReLU(inplace=True),
                                        nn.Dropout(p=.2), nn.Linear(64, 3))

    def forward(self, image):
        return self.classifier(self.backbone(image))


class LightFusionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = LightBackbone()
        self.classifier = nn.Sequential(nn.Linear(193, 64), nn.ReLU(inplace=True),
                                        nn.Dropout(p=.2), nn.Linear(64, 3))

    def forward(self, image, weight):
        features = self.backbone(image)
        if weight.numel() != image.shape[0]:
            raise ValueError("Expected 1 standardized weight per image")
        return self.classifier(torch.cat((features, weight.reshape(-1, 1).to(features.dtype)), dim=1))


def build_model(name, pretrained=False):
    if name in ("image", "fusion", "weight"):
        return OriginalNet(name, pretrained)
    if name == "custom_image":
        return LightImageCNN()
    if name == "custom_fusion":
        return LightFusionCNN()
    raise ValueError(f"Unknown model {name}")


def forward(model, name, image, weight):
    if name == "custom_image":
        return model(image)
    return model(image, weight)


def choose_models(selection):
    return {"custom": ["custom_image", "custom_fusion"],
            "original": ["image", "weight", "fusion"],
            "all": list(MODEL_NAMES)}.get(selection, [selection])


# ---------- 3. Training, image-by-image evaluation and CSV ----------

def make_transform(augment):
    steps = []
    if augment:
        steps = [transforms.RandomHorizontalFlip(p=.5), transforms.RandomRotation(8),
                 transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1)]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class CitrusData(Dataset):
    def __init__(self, rows, root, split, mean, std, *, weight_only=False, augment=False,
                 unique_weight=False):
        self.rows = [r for r in rows if r["split"] == split]
        if unique_weight:
            self.rows = list({r["fruit_id"]: r for r in self.rows}.values())
        self.root, self.weight_only = root, weight_only
        self.mean, self.std = mean, std
        self.transform = make_transform(augment)
        if not self.rows:
            raise ValueError(f"Empty split: {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        if self.weight_only:
            image = torch.empty(0, dtype=torch.float32)
        else:
            with Image.open(self.root / row["image_path"]) as img:
                image = self.transform(img.convert("RGB"))
        weight = (float(row["weight_g"]) - self.mean) / self.std
        return (image, torch.tensor([weight], dtype=torch.float32),
                torch.tensor(CLASSES.index(row["class"]), dtype=torch.long))


def scores(y_true, y_pred):
    matrix = np.zeros((3, 3), dtype=np.int64)
    for a, b in zip(y_true, y_pred):
        matrix[int(a), int(b)] += 1
    f1 = []
    for c in range(3):
        tp = matrix[c, c]; fp = sum(matrix[:, c]) - tp; fn = sum(matrix[c, :]) - tp
        denom = 2 * tp + fp + fn
        f1.append(float(2 * tp / denom) if denom else 0.0)
    return {"accuracy": float(np.trace(matrix) / matrix.sum()),
            "macro_f1": float(np.mean(f1)), "confusion_matrix": matrix.tolist()}


@torch.inference_mode()
def evaluate(model, name, dl, device):
    model.eval()
    total_loss, count, labels, preds = 0.0, 0, [], []
    loss_fn = nn.CrossEntropyLoss()
    for x, w, y in dl:
        pred = forward(model, name, x.to(device), w.to(device))
        total_loss += float(loss_fn(pred, y.to(device))) * len(y)
        count += len(y)
        labels.extend(y.tolist()); preds.extend(pred.argmax(1).cpu().tolist())
    return {"loss": total_loss / count, "image": scores(labels, preds), "n_images": count}


@torch.inference_mode()
def predictions_csv(model, name, rows, root, device, mean, std, batch, path):
    model.eval()
    result = []
    for split in ("train", "val", "test"):
        ds = CitrusData(rows, root, split, mean, std, weight_only=name == "weight")
        dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
        offset = 0
        for x, w, y in dl:
            probs = torch.softmax(forward(model, name, x.to(device), w.to(device)), 1).cpu().numpy()
            for j in range(len(y)):
                row = ds.rows[offset + j]
                true = int(y[j]); prediction = int(probs[j].argmax())
                result.append({"split": split, "image_path": row["image_path"],
                               "fruit_id": row["fruit_id"], "weight_g": row["weight_g"],
                               "true_class": CLASSES[true], "predicted_class": CLASSES[prediction],
                               "correct": int(true == prediction),
                               "confidence": float(probs[j, prediction]),
                               **{f"prob_{c}": float(probs[j, i]) for i, c in enumerate(CLASSES)}})
            offset += len(y)
    if len(result) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} per-image predictions, got {len(result)}")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0])); writer.writeheader(); writer.writerows(result)
    print(f"Per-image CSV: {path.name} ({len(result)} rows)")


def load_checkpoint(root, name, device):
    path = root / "model" / f"{name}_best.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(ckpt["class_names"]) != CLASSES:
        raise ValueError(f"Class order mismatch in {path}")
    if name.startswith("custom_") and ckpt.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Custom architecture mismatch in {path}")
    model = build_model(name, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def train_one(root, rows, name, args, device, mean, std):
    weight_only = name == "weight"
    custom = name.startswith("custom_")
    train_ds = CitrusData(rows, root, "train", mean, std, weight_only=weight_only,
                          unique_weight=weight_only, augment=not args.no_augment and not weight_only)
    val_ds = CitrusData(rows, root, "val", mean, std, weight_only=weight_only, unique_weight=weight_only)
    test_ds = CitrusData(rows, root, "test", mean, std, weight_only=weight_only, unique_weight=weight_only)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    pretrained = not custom and not weight_only and not args.no_pretrained
    model = build_model(name, pretrained=pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    lr = args.lr if args.lr else (3e-3 if weight_only else 1e-3 if custom else 1e-4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3 if custom else 2, factor=.5)
    ckpt_path = root / "model" / f"{name}_best.pth"
    ckpt_path.parent.mkdir(exist_ok=True, parents=True)
    history, best_loss, bad_epochs = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = 0.0; count = 0
        for x, w, y in train_dl:
            x, w, y = x.to(device), w.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward(model, name, x, w)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.item()) * len(y); count += len(y)
        val = evaluate(model, name, val_dl, device)
        scheduler.step(val["loss"])
        history.append({"epoch": epoch, "train_loss": loss_sum / count, "val_loss": val["loss"],
                        "val_image_accuracy": val["image"]["accuracy"],
                        "val_image_macro_f1": val["image"]["macro_f1"]})
        print(f"{name} | Epoch {epoch:02d}/{args.epochs} | train loss={loss_sum/count:.4f} | "
              f"val loss={val['loss']:.4f} | val accuracy={val['image']['accuracy']:.1%}", flush=True)
        if val["loss"] < best_loss - 1e-4:
            best_loss, bad_epochs = val["loss"], 0
            torch.save({"state_dict": model.state_dict(),
                        "architecture": ARCHITECTURE if custom else "mobilenetv3_small_or_weight_mlp",
                        "model_type": name.replace("custom_", "") if custom else name,
                        "class_names": list(CLASSES), "image_size": [224, 224],
                        "image_mean": list(MEAN), "image_std": list(STD),
                        "weight_mean": mean, "weight_std": std, "selected_epoch": epoch,
                        "seed": args.seed, "pretrained": pretrained, "parameters": n_params}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: {name} at epoch {epoch}")
                break
    with (root / "model" / f"{name}_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    model, ckpt = load_checkpoint(root, name, device)
    test = evaluate(model, name, test_dl, device)
    result = {"model": name, "best_epoch": ckpt["selected_epoch"],
              "pretrained": pretrained, "train_fruits": len({r['fruit_id'] for r in rows if r['split'] == 'train'}),
              "weight_mean_train_only": mean, "weight_std_train_only": std,
              "parameters": n_params, "test_unit": "unique_fruit_weight" if weight_only else "individual_image",
              "test": test}
    with (root / "model" / f"{name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    predictions_csv(model, name, rows, root, device, mean, std, args.batch_size,
                    root / "model" / f"{name}_predictions.csv")
    print(f"TEST {name}: {test['image']['accuracy']:.2%}, macro F1={test['image']['macro_f1']:.4f}, "
          f"samples={test['n_images']} ({result['test_unit']})")
    for line in test["image"]["confusion_matrix"]:
        print(" ", line)


def evaluate_existing(root, rows, names, device, batch):
    for name in names:
        try:
            model, ckpt = load_checkpoint(root, name, device)
        except FileNotFoundError as e:
            print(f"SKIP {name}: {e}")
            continue
        mean, std = float(ckpt["weight_mean"]), float(ckpt["weight_std"])
        unique = name == "weight"
        ds = CitrusData(rows, root, "test", mean, std, weight_only=unique, unique_weight=unique)
        dl = DataLoader(ds, batch_size=batch, num_workers=0)
        test = evaluate(model, name, dl, device)
        print(f"TEST {name}: {test['image']['accuracy']:.2%}, F1={test['image']['macro_f1']:.4f}, "
              f"n={test['n_images']} ({'fruits' if unique else 'images'})")
        predictions_csv(model, name, rows, root, device, mean, std, batch,
                        root / "model" / f"{name}_predictions.csv")


def compare(root):
    descriptions = {"image": "MobileNet image", "weight": "Weight MLP", "fusion": "MobileNet fusion",
                    "custom_image": "Custom lightweight CNN", "custom_fusion": "Custom lightweight CNN fusion"}
    table = []
    for name in MODEL_NAMES:
        path = root / "model" / f"{name}_metrics.json"
        if not path.is_file():
            print(f"SKIP {name}: metrics not found")
            continue
        with path.open(encoding="utf-8-sig") as f:
            data = json.load(f)
        metric = data["test"]["image"]
        table.append({"model": name, "description": descriptions[name],
                      "test_unit": data.get("test_unit", "individual_image"),
                      "n_test_samples": data["test"].get("n_images", ""),
                      "test_accuracy": metric["accuracy"], "test_macro_f1": metric["macro_f1"],
                      "test_loss": data["test"]["loss"], "best_epoch": data.get("best_epoch", ""),
                      "pretrained": data.get("pretrained", ""), "parameters": data.get("parameters", ""),
                      "confusion_matrix": json.dumps(metric["confusion_matrix"], ensure_ascii=False)})
        print(f"{name:14s} | {table[-1]['test_unit']:18s} | "
              f"Accuracy {metric['accuracy']:.2%} | F1 {metric['macro_f1']:.4f}")
    if not table:
        raise FileNotFoundError("No *_metrics.json found under model/")
    out = root / "model" / "all_five_models_comparison.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    print(f"Saved: {out}")
    print("Note: weight-only is evaluated per unique fruit; image/fusion per image.")


# ---------- 4. ONNX export ----------

class OldImageExport(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, image):
        return self.model(image, image.new_zeros((image.shape[0], 1)))

class OldWeightExport(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, weight):
        return self.model(None, weight)


def export_one(root, name, verify=False):
    model, ckpt = load_checkpoint(root, name, "cpu")
    model.eval()
    example = torch.randn(1, 3, 224, 224)
    if name == "weight":
        wrapped, inputs, names = OldWeightExport(model), (torch.zeros(1, 1),), ["weight"]
    elif name == "image":
        wrapped, inputs, names = OldImageExport(model), (example,), ["image"]
    elif name in ("fusion", "custom_fusion"):
        wrapped, inputs, names = model, (example, torch.zeros(1, 1)), ["image", "weight"]
    else:
        wrapped, inputs, names = model, (example,), ["image"]
    wrapped.eval()
    target = root / "model" / f"{name}.onnx"
    torch.onnx.export(wrapped, inputs, str(target), input_names=names, output_names=["logits"],
                      opset_version=17,
                      dynamic_axes={**{k: {0: "batch"} for k in names}, "logits": {0: "batch"}},
                      dynamo=False)
    meta = {k: ckpt[k] for k in ("class_names", "image_size", "image_mean", "image_std",
                                   "weight_mean", "weight_std", "model_type") if k in ckpt}
    meta["weight_is_standardized"] = True
    if name.startswith("custom_"):
        meta["architecture"] = ARCHITECTURE
    with (root / "model" / f"{name}_deploy.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Exported {target.name} ({target.stat().st_size/1024**2:.2f} MB)")
    if verify:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("Install ONNX Runtime: py -m pip install onnxruntime") from e
        with torch.inference_mode():
            expected = wrapped(*inputs).cpu().numpy()
        session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
        observed = session.run(None, {k: v.numpy() for k, v in zip(names, inputs)})[0]
        diff = float(np.max(np.abs(expected-observed)))
        print(f"  ONNX vs PyTorch max |diff| = {diff:.6g}")
        if not np.allclose(expected, observed, rtol=1e-3, atol=1e-4):
            raise RuntimeError(f"ONNX parity check failed for {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "compare", "export", "info"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--models", choices=("custom", "original", "all") + MODEL_NAMES,
                        default="custom", help="custom=2 custom CNNs, original=3 old models")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Compare ONNX and PyTorch logits")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing checkpoints")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if args.command == "prepare":
        target = root / "dataset" / "metadata" / "splits.csv"
        if target.exists() and not args.overwrite:
            print(f"Already exists, preserving current split: {target}")
            print("Use --overwrite only if you deliberately want to rebuild splits.csv")
        else:
            prepare(root, args.seed)
        return
    if args.command == "info":
        for name in MODEL_NAMES:
            print(f"{name:14s} {sum(p.numel() for p in build_model(name).parameters()):,} parameters")
        return
    if args.command == "compare":
        compare(root); return
    models = choose_models(args.models)
    if args.command == "export":
        for name in models:
            export_one(root, name, args.verify)
        return
    rows = read_split(root)
    device = torch.device("cuda" if (args.device == "cuda" or
          (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        parser.error("CUDA requested but not available")
    if args.command == "evaluate":
        evaluate_existing(root, rows, models, device, args.batch_size)
        return
    if ((args.epochs is not None and args.epochs < 1) or args.batch_size < 1
            or (args.patience is not None and args.patience < 1)
            or (args.lr is not None and args.lr <= 0)):
        parser.error("epochs, batch-size, patience and learning rate must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    mean, std = weight_stats(rows)
    for name in models:
        path = root / "model" / f"{name}_best.pth"
        if path.exists() and not args.overwrite:
            print(f"SKIP existing checkpoint: {path.name} (use --overwrite to retrain)")
            continue
        from copy import copy
        options = copy(args)
        options.epochs = args.epochs or (60 if name.startswith("custom_") or name == "weight" else 20)
        options.patience = args.patience or (12 if name.startswith("custom_") else 6)
        print(f"\nTraining {name} on {device}; {options.epochs} epochs max, batch size {options.batch_size}")
        train_one(root, rows, name, options, device, mean, std)


if __name__ == "__main__":
    main()
