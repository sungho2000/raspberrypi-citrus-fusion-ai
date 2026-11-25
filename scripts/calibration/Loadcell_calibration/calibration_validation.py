
import lgpio
import time
import json

DT = 5
SCK = 6

# 교정값 불러오기
with open("calibration.json", "r") as f:
    calibration = json.load(f)

zero_offset = calibration["zero_offset"]
scale_factor = calibration["scale_factor"]

# GPIO 설정
h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_input(h, DT)
lgpio.gpio_claim_output(h, SCK, 0)

def read_raw():
    # 데이터 준비 대기
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

    # A채널, 증폭률 128
    lgpio.gpio_write(h, SCK, 1)
    lgpio.gpio_write(h, SCK, 0)

    if value & 0x800000:
        value -= 1 << 24

    return value

try:
    while True:
        raw = read_raw()

        if raw == -1:
            print("비정상 원시값 감지")
            continue

        # 보정식을 이용한 무게 계산
        weight = (raw - zero_offset) / scale_factor

        print(f"측정 무게: {weight:.2f}g")
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n측정 종료")

finally:
    lgpio.gpio_write(h, SCK, 0)
    lgpio.gpiochip_close(h)
