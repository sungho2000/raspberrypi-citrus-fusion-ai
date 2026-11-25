import lgpio
import time

DT = 5
SCK = 6

# Raspberry Pi 3 B+ GPIO 칩 열기
h = lgpio.gpiochip_open(0)

# GPIO 설정
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

    # 부호 있는 24비트 정수로 변환
    if value & 0x800000:
        value -= 1 << 24

    return value

try:
    while True:
        raw = read_raw()

        if raw == -1:
            print("경고: -1 원시값 발생")
        else:
            print(f"로드셀 원시값: {raw}")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n측정 종료")

except TimeoutError as e:
    print(f"\n오류: {e}")

finally:
    lgpio.gpio_write(h, SCK, 0)
    lgpio.gpiochip_close(h)
