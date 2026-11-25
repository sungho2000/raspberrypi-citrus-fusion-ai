import argparse
from collections import deque
import threading
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

HOME = Path.home()
MODELS_DIR = HOME / "citrus_final"
CALIBRATION_PATH = HOME / "Loadcell_calibration" / "calibration.json"
DT = 5       # GPIO BCM pin numbers (not physical header pin numbers)
SCK = 6
LABEL_KO = {"mandarin": "귤", "orange": "오렌지", "hwanggeumhyang": "황금향"}


def load_model(stem):
    model_path = MODELS_DIR / (stem + ".onnx")
    config_path = MODELS_DIR / (stem + "_deploy.json")
    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"모델 또는 설정 파일 없음: {model_path} / {config_path}")

    with config_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"],
        sess_options=ort.SessionOptions()
    )
    inputs = session.get_inputs()
    image_input = next((x.name for x in inputs if "image" in x.name.lower()), None)
    if image_input is None:
        image_input = next((x.name for x in inputs if len(x.shape) == 4), None)
    weight_input = next((x.name for x in inputs if "weight" in x.name.lower()), None)
    if weight_input is None and len(inputs) == 2:
        weight_input = next((x.name for x in inputs if x.name != image_input), None)
    if not image_input:
        raise ValueError(f"이미지 입력을 찾지 못했습니다: {stem}")
    return {"session": session, "cfg": cfg, "image_input": image_input,
            "weight_input": weight_input}


def preprocess(frame_rgb, cfg):
    # Uses the per-model deployment settings; input is RGB numpy (H, W, 3).
    height, width = (int(x) for x in cfg.get("image_size", [224, 224]))
    image = Image.fromarray(frame_rgb, mode="RGB").resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(cfg["image_mean"], dtype=np.float32)
    std = np.asarray(cfg["image_std"], dtype=np.float32)
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1))[None].astype(np.float32)


def crop_center_square(frame_rgb, crop_ratio):
    """Center square ROI matching square-cropped 224x224 training inputs.

    crop_ratio=1.0 means maximal centered square; <1.0 zooms into center.
    The final crop is then resized and normalized by preprocess().
    """
    height, width = frame_rgb.shape[:2]
    side = max(16, min(min(height, width), int(min(height, width) * crop_ratio)))
    left = (width - side) // 2
    top = (height - side) // 2
    return frame_rgb[top:top + side, left:left + side]


def predict(model, frame_rgb, weight_g=None):
    cfg = model["cfg"]
    data = {model["image_input"]: preprocess(frame_rgb, cfg)}
    if model["weight_input"]:
        if weight_g is None:
            raise ValueError("융합 모델에 무게 입력이 없습니다")
        value = float(weight_g)
        if cfg.get("weight_is_standardized", True):
            value = (value - float(cfg["weight_mean"])) / float(cfg["weight_std"])
        data[model["weight_input"]] = np.array([[value]], dtype=np.float32)

    logits = np.asarray(model["session"].run(None, data)[0], dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(logits)):
        raise ValueError("모델 출력에 NaN 또는 Inf가 있습니다")
    logits -= logits.max()
    exp = np.exp(logits)
    probabilities = exp / exp.sum()
    classes = cfg["class_names"]
    if len(classes) != len(probabilities):
        raise ValueError("출력 클래스 수와 deploy.json의 class_names가 다릅니다")
    return dict(zip(classes, probabilities * 100.0))


class HX711:
    def __init__(self, dt, sck, calibration_path):
        import lgpio  # Import at runtime; keep camera-only errors easier to diagnose.
        self.gpio = lgpio
        with calibration_path.open(encoding="utf-8") as f:
            cfg = json.load(f)
        self.zero = float(cfg["zero_offset"])
        self.scale = float(cfg["scale_factor"])
        if abs(self.scale) < 1e-10:
            raise ValueError("scale_factor가 0입니다. 400g 감도 교정을 먼저 진행하세요")
        self.dt = dt
        self.sck = sck
        self.h = self.gpio.gpiochip_open(0)
        try:
            self.gpio.gpio_claim_input(self.h, self.dt)
            self.gpio.gpio_claim_output(self.h, self.sck, 0)
        except Exception:
            self.gpio.gpiochip_close(self.h)
            raise

    def read_raw(self):
        start = time.monotonic()
        while self.gpio.gpio_read(self.h, self.dt) == 1:
            if time.monotonic() - start > 2.0:
                raise TimeoutError("HX711 응답 없음 (GPIO 5/6 확인)")
            time.sleep(0.001)

        value = 0
        for _ in range(24):
            self.gpio.gpio_write(self.h, self.sck, 1)
            self.gpio.gpio_write(self.h, self.sck, 0)
            value = (value << 1) | self.gpio.gpio_read(self.h, self.dt)
        self.gpio.gpio_write(self.h, self.sck, 1)  # 25th pulse: channel A, gain 128
        self.gpio.gpio_write(self.h, self.sck, 0)
        if value & 0x800000:
            value -= 1 << 24
        if value == -1:
            raise ValueError("HX711에서 비정상 원시값 -1이 나왔습니다")
        return value

    def grams(self, samples):
        raw = sum(self.read_raw() for _ in range(samples)) / samples
        return (raw - self.zero) / self.scale

    def close(self):
        if getattr(self, "h", None) is not None:
            try:
                self.gpio.gpio_write(self.h, self.sck, 0)
            finally:
                self.gpio.gpiochip_close(self.h)
                self.h = None


class Camera:
    def __init__(self, mode):
        self.kind = None
        self.camera = None
        if mode in ("auto", "picamera2"):
            try:
                from picamera2 import Picamera2
                self.camera = Picamera2()
                # Picamera2's BGR888 returns RGB channel order in capture_array().
                config = self.camera.create_video_configuration(
                    main={"size": (640, 480), "format": "BGR888"},
                    buffer_count=3
                )
                self.camera.configure(config)
                self.camera.start()
                time.sleep(0.6)
                self.kind = "picamera2"
            except (ImportError, ModuleNotFoundError) as exc:
                if mode == "picamera2":
                    raise RuntimeError(
                        "Picamera2가 현재 가상환경에 없습니다. --camera rpicam 옵션을 사용하세요."
                    ) from exc
            except Exception:
                if self.camera:
                    try:
                        self.camera.close()
                    except Exception:
                        pass
                raise
        if self.kind is None:
            if not shutil.which("rpicam-still"):
                raise RuntimeError("Picamera2를 불러올 수 없고 rpicam-still 명령어도 없습니다")
            self.kind = "rpicam"

    def capture(self):
        if self.kind == "picamera2":
            frame = self.camera.capture_array("main")
            return np.asarray(frame, dtype=np.uint8)
        cmd = ["rpicam-still", "--nopreview", "--immediate", "-t", "1",
               "--width", "640", "--height", "480", "-o", "-"]
        done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if done.returncode != 0:
            raise RuntimeError("rpicam-still 촬영 실패: " + done.stderr.decode(errors="replace")[-500:])
        with Image.open(io.BytesIO(done.stdout)) as img:
            return np.asarray(img.convert("RGB"))

    def close(self):
        if self.camera:
            self.camera.stop()
            self.camera.close()




class WeightSampler(threading.Thread):
    """Sample HX711 independently so the camera display is never held up by ADC waits."""
    def __init__(self, hx, window=5, hz=10):
        super().__init__(daemon=True)
        self.hx = hx
        self.window = deque(maxlen=window)
        self.period = 1.0 / hz
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.error = None

    def run(self):
        next_tick = time.monotonic()
        try:
            while not self.stop_event.is_set():
                raw = self.hx.read_raw()
                weight = (raw - self.hx.zero) / self.hx.scale
                with self.lock:
                    self.window.append(weight)
                next_tick += self.period
                delay = max(0.0, next_tick - time.monotonic())
                if delay:
                    self.stop_event.wait(delay)
                elif time.monotonic() - next_tick > self.period:
                    next_tick = time.monotonic()
        except Exception as exc:
            with self.lock:
                self.error = exc
        finally:
            self.hx.close()

    def snapshot(self):
        with self.lock:
            vals = tuple(self.window)
            error = self.error
        if not vals:
            return None, error
        return float(np.mean(vals)), error

    def stop(self):
        self.stop_event.set()
        self.join(timeout=3.0)
        if self.is_alive():
            print("경고: 로드셀 스레드 종료를 기다리는 중입니다.", file=sys.stderr)


def smooth_probs(history, labels):
    if not history:
        return None
    return {name: float(np.mean([p[name] for p in history])) for name in labels}


def label_scores(probs):
    if probs is None:
        return "--"
    return " | ".join(
        f"{LABEL_KO.get(name, name)} {score:5.1f}%" for name, score in probs.items()
    )


def best_label(probs):
    if probs is None:
        return "--"
    key = max(probs, key=probs.get)
    return LABEL_KO.get(key, key)


def show_terminal(weight, image_p, fusion_p, fps, camera_kind, args):
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")
    print("===== 감귤류 실시간 분류 (Ctrl+C 종료) =====")
    w = "측정 중" if weight is None else f"{weight:.1f} g"
    print(f"카메라: {camera_kind} | 영상: {fps:.1f} FPS | 무게: {w}")
    print(f"AI 추론: {1 / args.infer_interval:.1f} Hz 목표 | 최근 {args.smooth_window}회 확률 평균")
    print("\n[CNN 단독 (프루닝 50%)]")
    print(label_scores(image_p), "\n예측:", best_label(image_p))
    print("\n[이미지+무게 융합 (프루닝 40%)]")
    print(label_scores(fusion_p), "\n예측:", best_label(fusion_p))
    sys.stdout.flush()


def show_preview(cv2, frame_rgb, crop_ratio, weight, image_p, fusion_p, fps):
    """Display each camera frame; predictions update on the inference schedule."""
    panel = np.full((560, 1050, 3), 245, dtype=np.uint8)
    camera_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    panel[30:510, :640] = cv2.resize(camera_bgr, (640, 480))

    # Identify the square region used by both models.
    src_h, src_w = frame_rgb.shape[:2]
    side = max(16, min(min(src_h, src_w), int(min(src_h, src_w) * crop_ratio)))
    x0 = int((src_w - side) / 2 * 640 / src_w)
    y0 = 30 + int((src_h - side) / 2 * 480 / src_h)
    x1 = int((src_w + side) / 2 * 640 / src_w)
    y1 = 30 + int((src_h + side) / 2 * 480 / src_h)
    cv2.rectangle(panel, (x0, y0), (x1, y1), (80, 230, 100), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, "LIVE CAMERA (green: CNN input)", (12, 21), font, 0.55, (0, 0, 0), 1)
    cv2.putText(panel, f"Camera FPS: {fps:.1f}", (12, 535), font, 0.60, (0, 0, 0), 1)
    w = "..." if weight is None else f"{weight:.1f} g"
    cv2.putText(panel, f"Weight: {w}", (658, 36), font, 0.66, (0, 0, 0), 2)
    cv2.putText(panel, "CONTINUOUS CLASSIFICATION", (658, 62), font, 0.43, (80, 65, 30), 1)
    labels = [("Mandarin", "mandarin"), ("Orange", "orange"),
              ("Hwanggeumhyang", "hwanggeumhyang")]
    for title, probs, top_y in [
        ("CNN only (Pruning 50%)", image_p, 102),
        ("Image + Weight (Pruning 40%)", fusion_p, 320),
    ]:
        cv2.putText(panel, title, (658, top_y), font, 0.54, (0, 0, 0), 2)
        if probs is None:
            cv2.putText(panel, "Starting inference...", (665, top_y + 40), font,
                        0.48, (110, 110, 110), 1)
            continue
        for idx, (label, key) in enumerate(labels):
            y = top_y + 33 + idx * 53
            value = float(probs.get(key, 0))
            cv2.putText(panel, f"{label}: {value:.1f}%", (665, y), font,
                        0.46, (0, 0, 0), 1)
            cv2.rectangle(panel, (665, y + 8), (1008, y + 22), (220, 220, 220), -1)
            cv2.rectangle(panel, (665, y + 8), (665 + int(343 * value / 100), y + 22),
                          (80, 145, 60), -1)
    cv2.putText(panel, "Q: Quit | Moving average of 5 predictions (default)",
                (655, 536), font, 0.39, (60, 60, 60), 1)
    cv2.imshow("Citrus Real-time Classification", panel)
    return cv2.waitKey(1) & 0xFF == ord("q")


def main():
    parser = argparse.ArgumentParser(
        description="Smooth video + periodic 2-model ONNX citrus inference with probability averaging")
    parser.add_argument("--preview", action="store_true", help="Open video and probabilities GUI")
    parser.add_argument("--camera", choices=["auto", "picamera2", "rpicam"], default="auto")
    parser.add_argument("--crop-ratio", type=float, default=0.95,
                        help="Center square crop, 0.1-1.0 (default: 0.95)")
    parser.add_argument("--infer-interval", "--interval", dest="infer_interval", type=float,
                        default=0.2, help="Seconds between model inference attempts (default: 0.2)")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Rolling average of most recent N probability outputs (default: 5)")
    parser.add_argument("--weight-window", "--samples", dest="weight_window", type=int,
                        default=5, help="Most recent N HX711 weights to average (default: 5)")
    parser.add_argument("--weight-hz", type=float, default=10.0,
                        help="Maximum HX711 polling rate (default: 10 Hz)")
    args = parser.parse_args()
    if not 0.1 <= args.crop_ratio <= 1.0:
        parser.error("--crop-ratio는 0.1~1.0 사이여야 합니다")
    if args.infer_interval <= 0 or args.smooth_window < 1 or args.weight_window < 1:
        parser.error("--infer-interval은 양수, --smooth-window/--weight-window는 1 이상이어야 합니다")
    if args.weight_hz <= 0:
        parser.error("--weight-hz는 양수여야 합니다")

    image_model = load_model("custom_image_pruned")
    fusion_model = load_model("custom_fusion_pruned")
    if not fusion_model["weight_input"]:
        raise ValueError("융합 ONNX 모델에서 weight 입력을 찾지 못했습니다")
    if image_model["cfg"]["class_names"] != fusion_model["cfg"]["class_names"]:
        raise ValueError("두 모델의 class_names 순서가 다릅니다")
    classes = image_model["cfg"]["class_names"]

    camera = worker = cv2 = None
    try:
        hx = HX711(DT, SCK, CALIBRATION_PATH)
        worker = WeightSampler(hx, args.weight_window, args.weight_hz)
        worker.start()
        camera = Camera(args.camera)
        if camera.kind == "rpicam":
            print("참고: rpicam-still 방식은 매번 사진을 촬영하므로 영상이 느릴 수 있습니다.")
            print("부드러운 프리뷰는 Picamera2가 필요합니다.")
        if args.preview:
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                print("그래픽 디스플레이가 없어 터미널 출력만 사용합니다")
            else:
                try:
                    import cv2 as cv2_module
                    if "GUI:                           NONE" in cv2_module.getBuildInformation():
                        print("OpenCV에 GUI 지원이 없어 터미널만 사용합니다")
                    else:
                        cv2 = cv2_module
                except ImportError:
                    print("cv2 모듈이 없어 터미널 출력만 사용합니다")

        image_history = deque(maxlen=args.smooth_window)
        fusion_history = deque(maxlen=args.smooth_window)
        image_p = fusion_p = None
        last_infer = -1e9
        last_terminal = -1e9
        frame_count = 0
        fps = 0.0
        fps_start = time.monotonic()

        print("실시간 분류 시작 (종료: Q 또는 Ctrl+C)")
        while True:
            frame = camera.capture()  # Camera refresh EVERY iteration.
            now = time.monotonic()
            frame_count += 1
            elapsed_fps = now - fps_start
            if elapsed_fps >= 1.0:
                fps = frame_count / elapsed_fps
                frame_count = 0
                fps_start = now

            weight_g, err = worker.snapshot()
            if err is not None:
                raise RuntimeError(f"로드셀 측정 실패: {err}")

            # Run the image-only CNN continuously, without a weight threshold.
            # The fusion model starts as soon as the first HX711 sample arrives.
            if now - last_infer >= args.infer_interval:
                crop = crop_center_square(frame, args.crop_ratio)
                image_history.append(predict(image_model, crop))
                image_p = smooth_probs(image_history, classes)
                if weight_g is not None:
                    fusion_history.append(predict(fusion_model, crop, weight_g))
                    fusion_p = smooth_probs(fusion_history, classes)
                last_infer = time.monotonic()

            if cv2 is not None:
                if show_preview(cv2, frame, args.crop_ratio, weight_g,
                                image_p, fusion_p, fps):
                    break
            elif now - last_terminal >= 0.5:
                show_terminal(weight_g, image_p, fusion_p, fps, camera.kind, args)
                last_terminal = now
    except KeyboardInterrupt:
        print("\n정상 종료했습니다")
    finally:
        if camera is not None:
            camera.close()
        if worker is not None:
            worker.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
