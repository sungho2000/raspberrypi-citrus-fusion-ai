
import lgpio
import time
import json
from pathlib import Path

DT = 5
SCK = 6

REFERENCE_WEIGHT = 400.0
SAMPLE_COUNT = 100
INTERVAL = 1.0

SAVE_PATH = Path(__file__).resolve().parent / "calibration.json"

# 영점 교정값 불러오기
with open(SAVE_PATH, "r") as f:
    calibration = json.load(f)

zero_offset = float(calibration["zero_offset"])

# GPIO 설정
h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_input(h, DT)
lgpio.gpio_claim_output(h, SCK, 0)


def read_raw():
    start = time.monotonic()

    while lgpio.gpio_read(h, DT) == 1:
        if time.monotonic() - start > 2:
            raise TimeoutError("HX711 응답 없음")
        time.sleep(0.001)

    value = 0

    for _ in range(24):
        lgpio.gpio_write(h, SCK, 1)
        lgpio.gpio_write(h, SCK, 0)

        bit = lgpio.gpio_read(h, DT)
        value = (value << 1) | bit

    # A채널, 증폭률 128
    lgpio.gpio_write(h, SCK, 1)
    lgpio.gpio_write(h, SCK, 0)

    if value & 0x800000:
        value -= 1 << 24

    return value


try:
    print("===== 감도 교정 =====")
    print(f"영점값: {zero_offset:.2f}")

    input("400g 분동을 올린 후 Enter를 누르세요.")

    print("5초 동안 안정화합니다.")
    time.sleep(5)

    raw_values = []
    start_time = time.monotonic()

    for i in range(SAMPLE_COUNT):
        next_time = start_time + i * INTERVAL
        wait_time = next_time - time.monotonic()

        if wait_time > 0:
            time.sleep(wait_time)

        raw = read_raw()

        if raw == -1:
            raise ValueError("비정상 원시값 -1 감지")

        raw_values.append(raw)
        print(f"{i+1}/100 | 원시값: {raw}")

    # 400g 평균 원시값 계산
    reference_raw = sum(raw_values) / len(raw_values)

    # 감도 보정계수 계산 (ADC counts/g)
    scale_factor = (
        reference_raw - zero_offset
    ) / REFERENCE_WEIGHT

    if abs(scale_factor) < 1e-9:
        raise ValueError("보정계수가 0에 가까워 계산할 수 없습니다.")

    # 기존 영점값은 유지하고 감도값 추가
    calibration["scale_factor"] = scale_factor
    calibration["reference_weight_g"] = REFERENCE_WEIGHT
    calibration["reference_raw_mean"] = reference_raw

    with open(SAVE_PATH, "w") as f:
        json.dump(calibration, f, indent=4)

    print("\n===== 감도 교정 결과 =====")
    print(f"영점값: {zero_offset:.2f}")
    print(f"400g 평균 원시값: {reference_raw:.2f}")
    print(f"보정계수: {scale_factor:.6f} counts/g")
    print(f"저장 경로: {SAVE_PATH}")
    print("==========================")

except (KeyboardInterrupt, TimeoutError, ValueError) as e:
    print(f"\n측정 중단: {e}")

finally:
    lgpio.gpio_write(h, SCK, 0)
    lgpio.gpiochip_close(h)
