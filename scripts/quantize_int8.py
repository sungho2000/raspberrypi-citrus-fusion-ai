import json
from pathlib import Path

import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
from PIL import Image
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"

# Test 데이터는 제외하고 Train 데이터로 양자화 보정
df = pd.read_csv(ROOT / "dataset/metadata/splits.csv")
train_df = df[df["split"] == "train"]

# 클래스별 70장씩, 총 210장 선택
samples = pd.concat([
    group.sample(n=min(70, len(group)), random_state=42)
    for _, group in train_df.groupby("class")
]).to_dict("records")


class CitrusReader(CalibrationDataReader):
    def __init__(self, model_name):
        self.model_name = model_name
        self.rows = samples
        self.index = 0

        config_path = MODEL_DIR / f"{model_name}_deploy.json"
        with open(config_path, encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.mean = np.array(
            self.cfg["image_mean"], dtype=np.float32
        ).reshape(1, 1, 3)

        self.std = np.array(
            self.cfg["image_std"], dtype=np.float32
        ).reshape(1, 1, 3)

    def get_next(self):
        if self.index >= len(self.rows):
            return None

        row = self.rows[self.index]
        self.index += 1

        image_path = ROOT / row["image_path"]

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if img.size != tuple(self.cfg["image_size"]):
                raise ValueError(f"Image size mismatch: {image_path}")
            image = np.asarray(img, dtype=np.float32) / 255.0

        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1)[None, ...].copy()

        feed = {"image": image}

        if self.model_name == "custom_fusion":
            weight = float(row["weight_g"])

            if self.cfg.get("weight_is_standardized", True):
                weight = (
                    weight - self.cfg["weight_mean"]
                ) / self.cfg["weight_std"]

            feed["weight"] = np.array(
                [[weight]], dtype=np.float32
            )

        return feed

    def rewind(self):
        self.index = 0


for name in ["custom_image", "custom_fusion"]:
    original = MODEL_DIR / f"{name}.onnx"
    quantized = MODEL_DIR / f"{name}_int8.onnx"

    print(f"\n{name}: INT8 양자화 시작")

    reader = CitrusReader(name)

    quantize_static(
        model_input=str(original),
        model_output=str(quantized),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )

    # ONNX 파일 구조 및 실행 가능 여부 확인
    onnx.checker.check_model(str(quantized))
    session = ort.InferenceSession(
        str(quantized),
        providers=["CPUExecutionProvider"]
    )

    reader.rewind()
    session.run(None, reader.get_next())

    original_kb = original.stat().st_size / 1024
    quantized_kb = quantized.stat().st_size / 1024

    print(f"원본 크기: {original_kb:.1f} KB")
    print(f"INT8 크기: {quantized_kb:.1f} KB")
    print(f"생성 완료: {quantized.name}")

print("\n모든 양자화 완료!")
