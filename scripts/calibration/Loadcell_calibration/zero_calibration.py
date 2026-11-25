
import lgpio
import time
import json
from pathlib import Path

DT = 5
SCK = 6

SAMPLE_COUNT = 100
INTERVAL = 1.0  # 1초 간격

# 교정 결과 저장 경로
SAVE_PATH = Path(__file__).resolve().parent / "calibration.json"

# GPIO 설정
h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_input(h, DT)
lgpio.gpio_claim_output(h, SCK, 0)


def read_raw():
    # HX711 데이터 준비 대기
    start = time.monotonic()

    while lgpio.gpio_read(h, DT) == 1:
        if time.monotonic() - start > 2:
            raise TimeoutError("HX711 응답 없음")
        time.sleep(0.001)

    value = 0

    # 24비트 원시값 읽기
    for _ in range(24):
        lgpio.gpio_write(h, SCK, 1)
        lgpio.gpio_write(h, SCK, 0)

        bit = lgpio.gpio_read(h, DT)
        value = (value << 1) | bit

    # 25번째 클럭: A채널, 증폭률 128
    lgpio.gpio_write(h, SCK, 1)
    lgpio.gpio_write(h, SCK, 0)

    # 부호 있는 24비트 정수 변환
    if value & 0x800000:
        value -= 1 << 24

    return value


try:
    raw_values = []

    print("===== 영점 교정 시작 =====")
    print("받침대 위에 아무것도 올리지 마세요.")

    start_time = time.monotonic()

    # 원시값 100개 측정
    for i in range(SAMPLE_COUNT):

        # 1초 간격 유지
        next_time = start_time + i * INTERVAL
        wait_time = next_time - time.monotonic()

        if wait_time > 0:
            time.sleep(wait_time)

        raw = read_raw()

        # 비정상 데이터 확인
        if raw == -1:
            raise ValueError(
                "-1 원시값 발생: 교정을 중단합니다."
            )

        raw_values.append(raw)

        print(f"{i+1}/100 | 원시값: {raw}")

    # 평균 계산
    zero_offset = sum(raw_values) / len(raw_values)

    print("\n===== 영점 교정 결과 =====")
    print(f"측정 횟수: {len(raw_values)}")
    print(f"최솟값: {min(raw_values)}")
    print(f"최댓값: {max(raw_values)}")
    print(f"영점 평균값: {zero_offset:.2f}")
    print("==========================")

    # 기존 교정 데이터가 있으면 유지
    calibration = {}

    if SAVE_PATH.exists():
        with open(SAVE_PATH, "r") as f:
            calibration = json.load(f)

    # 영점값 저장
    calibration["zero_offset"] = zero_offset

    with open(SAVE_PATH, "w") as f:
        json.dump(calibration, f, indent=4)

    print(f"영점값 저장 완료: {SAVE_PATH}")

except KeyboardInterrupt:
    print("\n측정 중단")

except (TimeoutError, ValueError) as e:
    print(f"\n오류: {e}")

finally:
    lgpio.gpio_write(h, SCK, 0)
    lgpio.gpiochip_close(h)
