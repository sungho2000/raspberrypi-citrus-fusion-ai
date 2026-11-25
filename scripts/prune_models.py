from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import citrus

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
NAMES = ("custom_image", "custom_fusion")


class SlimBackbone(nn.Module):
    def __init__(self, widths):
        super().__init__()
        layers = [
            nn.Conv2d(3, widths[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True),
        ]
        for i in range(5):
            stride = 2 if i < 4 else 1
            layers.append(citrus.DepthwiseSeparableConv(widths[i], widths[i + 1], stride))
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SlimImageCNN(nn.Module):
    def __init__(self, widths, hidden):
        super().__init__()
        self.backbone = SlimBackbone(widths)
        self.classifier = nn.Sequential(nn.Linear(widths[-1], hidden), nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.2), nn.Linear(hidden, 3))

    def forward(self, image):
        return self.classifier(self.backbone(image))


class SlimFusionCNN(nn.Module):
    def __init__(self, widths, hidden):
        super().__init__()
        self.backbone = SlimBackbone(widths)
        self.classifier = nn.Sequential(nn.Linear(widths[-1] + 1, hidden), nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.2), nn.Linear(hidden, 3))

    def forward(self, image, weight):
        features = self.backbone(image)
        if weight.numel() != image.shape[0]:
            raise ValueError("Expected one standardized weight per image")
        return self.classifier(torch.cat((features, weight.reshape(-1, 1).to(features.dtype)), dim=1))


def channel_keep(conv: nn.Conv2d, ratio: float) -> torch.Tensor:
    """Keep high-L1 output channels; round kept channel count to multiples of 4."""
    n = conv.out_channels
    target = max(4, min(n, int(round((n * (1.0 - ratio)) / 4)) * 4))
    importance = conv.weight.detach().abs().sum(dim=(1, 2, 3))
    return torch.topk(importance, k=target, sorted=False).indices.sort().values


def hidden_keep(fc: nn.Linear, ratio: float) -> torch.Tensor:
    n = fc.out_features
    target = max(4, min(n, int(round((n * (1.0 - ratio)) / 4)) * 4))
    importance = fc.weight.detach().abs().sum(dim=1)
    return torch.topk(importance, k=target, sorted=False).indices.sort().values


def copy_bn(dst: nn.BatchNorm2d, src: nn.BatchNorm2d, indices: torch.Tensor):
    for field in ("weight", "bias", "running_mean", "running_var"):
        getattr(dst, field).copy_(getattr(src, field)[indices])
    dst.num_batches_tracked.copy_(src.num_batches_tracked)


def structurally_prune(original: nn.Module, model_name: str, ratio: float):
    """Physically shorten stem, 5 depthwise blocks and classifier."""
    stem_old = original.backbone.layers[0]
    blocks_old = [original.backbone.layers[i] for i in range(3, 8)]
    keeps = [channel_keep(stem_old, ratio)]
    keeps += [channel_keep(block.layers[3], ratio) for block in blocks_old]
    fc_old = original.classifier[0]
    keep_h = hidden_keep(fc_old, ratio)
    widths = [len(k) for k in keeps]
    hidden = len(keep_h)
    new_model = (SlimFusionCNN(widths, hidden) if model_name == "custom_fusion"
                 else SlimImageCNN(widths, hidden))

    old_layers = original.backbone.layers
    new_layers = new_model.backbone.layers
    with torch.no_grad():
        # Stem Conv 3->C and its BatchNorm.
        new_layers[0].weight.copy_(old_layers[0].weight[keeps[0]])
        copy_bn(new_layers[1], old_layers[1], keeps[0])

        # Each block: depthwise Conv+BN -> pointwise Conv+BN.
        # Next layer depthwise depends on previous layer's retained channels.
        for i in range(5):
            prev, current = keeps[i], keeps[i + 1]
            old = old_layers[i + 3].layers
            new = new_layers[i + 3].layers
            new[0].weight.copy_(old[0].weight[prev])
            copy_bn(new[1], old[1], prev)
            new[3].weight.copy_(old[3].weight[current][:, prev])
            copy_bn(new[4], old[4], current)

        # Classifier must keep weight-input column for the fusion model.
        previous = keeps[-1]
        if model_name == "custom_fusion":
            selected_columns = torch.cat((previous, torch.tensor([fc_old.in_features - 1])))
        else:
            selected_columns = previous
        new_model.classifier[0].weight.copy_(fc_old.weight[keep_h][:, selected_columns])
        new_model.classifier[0].bias.copy_(fc_old.bias[keep_h])
        new_model.classifier[3].weight.copy_(original.classifier[3].weight[:, keep_h])
        new_model.classifier[3].bias.copy_(original.classifier[3].bias)

    original_params = sum(p.numel() for p in original.parameters())
    new_params = sum(p.numel() for p in new_model.parameters())
    if new_params >= original_params:
        raise RuntimeError("Pruning did not reduce parameter count")
    return new_model, widths, hidden, original_params, new_params


def epoch_loss(model, model_name, loader, device, optimizer=None):
    model.train(optimizer is not None)
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, count = 0.0, 0, 0
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for x, w, y in loader:
            x, w, y = x.to(device), w.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = citrus.forward(model, model_name, x, w)
            loss = criterion(logits, y)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += int((logits.argmax(dim=1) == y).sum())
            count += len(y)
    return total_loss / count, 100.0 * correct / count


def make_loaders(rows, checkpoint, batch_size):
    mean = float(checkpoint["weight_mean"])
    std = float(checkpoint["weight_std"])
    train = citrus.CitrusData(rows, ROOT, "train", mean, std, augment=True)
    val = citrus.CitrusData(rows, ROOT, "val", mean, std, augment=False)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def export_onnx(model, name, checkpoint, widths, hidden, ratio, target):
    model.cpu().eval()
    example_image = torch.randn(1, 3, 224, 224)
    if name == "custom_fusion":
        examples = (example_image, torch.zeros(1, 1))
        input_names = ["image", "weight"]
    else:
        examples = (example_image,)
        input_names = ["image"]

    with torch.inference_mode():
        reference = model(*examples).cpu().numpy()
    torch.onnx.export(
        model, examples, str(target), opset_version=17, dynamo=False,
        input_names=input_names, output_names=["logits"],
        dynamic_axes={**{key: {0: "batch"} for key in input_names}, "logits": {0: "batch"}},
    )
    metadata = {k: checkpoint[k] for k in (
        "class_names", "image_size", "image_mean", "image_std", "weight_mean",
        "weight_std", "model_type") if k in checkpoint}
    metadata.update({
        "architecture": "custom_depthwise_separable_cnn_v1_structurally_pruned",
        "weight_is_standardized": True,
        "structured_pruned": True,
        "target_channel_prune_ratio": ratio,
        "backbone_widths": widths,
        "classifier_hidden": hidden,
    })
    json_file = target.with_name(target.stem + "_deploy.json")
    json_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import onnx
        onnx.checker.check_model(str(target))
        import onnxruntime as ort
        session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
        result = session.run(None, dict(zip(input_names, [e.numpy() for e in examples])))[0]
        maximum_difference = float(np.abs(reference - result).max())
        print(f"  ONNX verification max |diff|={maximum_difference:.7g}")
        if not np.allclose(reference, result, atol=1e-4, rtol=1e-3):
            raise RuntimeError(f"ONNX parity check failed: {name}")
    except ImportError:
        print("  ONNX Runtime verification skipped (onnxruntime not installed)")
    return json_file


def main():
    parser = argparse.ArgumentParser(description="Structured pruning for two custom Citrus CNNs")
    parser.add_argument("--ratio", type=float, default=0.20, help="Channel pruning fraction target")
    parser.add_argument("--epochs", type=int, default=8, help="Fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--models", nargs="+", choices=NAMES, default=list(NAMES))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--overwrite", action="store_true", help="Replace previously pruned outputs")
    args = parser.parse_args()
    if not 0 < args.ratio <= 0.9:
        parser.error("--ratio must be in (0, 0.9]")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("--epochs / --batch-size / --lr must be positive")

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device} / pruning target: {args.ratio:.0%}")
    rows = citrus.read_split(ROOT)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.models:
        print(f"\n=== {name} ===", flush=True)
        out_pth = MODEL_DIR / f"{name}_pruned_best.pth"
        out_onnx = MODEL_DIR / f"{name}_pruned.onnx"
        out_meta = MODEL_DIR / f"{name}_pruned_deploy.json"
        if not args.overwrite and any(p.exists() for p in (out_pth, out_onnx, out_meta)):
            raise FileExistsError(f"Existing pruned outputs for {name}; add --overwrite if intentional")
        original, ckpt = citrus.load_checkpoint(ROOT, name, "cpu")
        original.eval()
        pruned, widths, hidden, before, after = structurally_prune(original, name, args.ratio)
        print(f"Channels: [24, 48, 96, 128, 192, 192] -> {widths}")
        print(f"Classifier hidden: 64 -> {hidden}")
        print(f"Parameters: {before:,} -> {after:,} ({100 * (before - after) / before:.1f}% fewer)")

        train_loader, val_loader = make_loaders(rows, ckpt, args.batch_size)
        pruned = pruned.to(device)
        best_loss, initial_acc = epoch_loss(pruned, name, val_loader, device)
        best_state = copy.deepcopy(pruned.state_dict())
        print(f"Initial VAL: loss={best_loss:.4f}, accuracy={initial_acc:.2f}%", flush=True)
        optimizer = torch.optim.AdamW(pruned.parameters(), lr=args.lr, weight_decay=1e-4)
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = epoch_loss(pruned, name, train_loader, device, optimizer)
            val_loss, val_acc = epoch_loss(pruned, name, val_loader, device)
            print(f"Epoch {epoch:02d}/{args.epochs}: TRAIN loss={train_loss:.4f}, "
                  f"VAL loss={val_loss:.4f}, VAL acc={val_acc:.2f}%", flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(pruned.state_dict())

        pruned.load_state_dict(best_state)
        final_loss, final_acc = epoch_loss(pruned, name, val_loader, device)
        print(f"Best VAL: loss={final_loss:.4f}, accuracy={final_acc:.2f}%")
        torch.save({
            "state_dict": {k: v.cpu() for k, v in pruned.state_dict().items()},
            "source_model": name, "target_prune_ratio": args.ratio,
            "architecture": "custom_depthwise_separable_cnn_v1_structurally_pruned",
            "widths": widths, "hidden": hidden, "best_val_loss": best_loss,
            "parameters_before": before, "parameters_after": after,
            "class_names": ckpt["class_names"],
            "weight_mean": ckpt["weight_mean"], "weight_std": ckpt["weight_std"],
        }, out_pth)
        out_json = export_onnx(pruned, name, ckpt, widths, hidden, args.ratio, out_onnx)
        print(f"Saved: {out_pth.name}, {out_onnx.name} ({out_onnx.stat().st_size / 1024:.1f} KiB), "
              f"{out_json.name}", flush=True)
    print("\nStructured pruning completed. Test data was not used for fine-tuning.")


if __name__ == "__main__":
    main()
