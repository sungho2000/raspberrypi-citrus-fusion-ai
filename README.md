# Citrus Fusion AI

**Raspberry Pi 5 기반 감귤류 이미지·무게 센서 융합 분류 및 경량화**

Citrus Fusion AI는 카메라 이미지와 로드셀 무게 정보를 이용하여 **귤(Mandarin), 오렌지(Orange), 황금향(Hwanggeumhyang)**을 분류하는 캡스톤디자인 프로젝트입니다. 이미지 단독 경량 CNN과 이미지+무게 융합 경량 CNN을 학습하고, **INT8 양자화와 구조적 프루닝**으로 모델을 최적화한 뒤 Raspberry Pi 5에서 정확도·추론시간·모델 크기를 비교했습니다.

- **팀:** LevelUp
- **최종 채택 모델:** 이미지 단독 CNN **프루닝 50%**, 이미지+무게 융합 CNN **프루닝 40%**
- **분류 대상:** 귤 / 오렌지 / 황금향 (3개 클래스)
- **실행 장치:** Raspberry Pi 5 + Camera Module 3 + HX711 + 1 kg 로드셀

> **주의:** 본 모델은 세 종류만 구분하는 *폐쇄 집합 분류기*입니다. 빈 화면이나 다른 물체를 비춰도 세 종류 중 하나를 높은 확률로 예측할 수 있습니다. UI의 예측 비율은 Softmax 출력이며 검증된 정답 확률을 뜻하지 않습니다.

## 1. 시스템 개요

```text
Camera Module 3 ──> 이미지 (RGB, 224×224) ──> 경량 CNN ───────────> 이미지 단독 분류
                                       │
                                       └──> 이미지 특징 ─────┐
                                                            ├─> 특징 연결 → FC 분류층 → 융합 분류
1 kg 로드셀 ──> HX711 ──> GPIO ──> 영점·감도 교정 ──> 무게(g) ──> 표준화 ──┘
```

융합 모델은 **특징 수준 융합(Feature-level Fusion)**을 사용합니다. 별도 무게 분류기의 예측 확률을 합치는 것이 아니라, **CNN이 추출한 이미지 특징 벡터에 표준화된 무게값 하나를 연결(Concatenation)**한 뒤 완전연결층에서 최종 분류합니다.

| 구성 요소 | 용도 |
| --- | --- |
| Raspberry Pi 5 | GPIO 데이터 수집, ONNX 추론 및 UI 실행 |
| Raspberry Pi Camera Module 3 | 과일 이미지 촬영 및 실시간 영상 |
| HX711 + 1 kg 로드셀 | 과일 무게 측정 |
| Python / PyTorch / torchvision | 모델 학습 및 프루닝 |
| ONNX / ONNX Runtime | 모델 내보내기, INT8 양자화 및 Raspberry Pi 추론 |
| OpenCV / Pillow / NumPy / pandas | 영상·이미지·데이터 처리 |
| lgpio / Picamera2 | GPIO 접근 및 카메라 프레임 취득(사용 환경에 따라) |

## 2. 데이터셋

| 항목 | 내용 |
| --- | --- |
| 대상 과일 | 귤·오렌지·황금향 각 30개, 총 **90개 개체** |
| 이미지 | 개체당 14방향 촬영, 총 **1,260장** |
| 전처리 | 과일을 포함한 정사각형 영역을 잘라 **224×224 RGB** 이미지로 저장 |
| 학습 분할 | Train **882장** / Validation **168장** / Test **210장** |
| 분할 기준 | 동일 과일 개체(`fruit_id`)의 이미지는 하나의 분할에만 배치 |
| 무게 데이터 | 개체별 무게(g)를 이미지와 연결 |

- `dataset/raw/`: 원본 촬영 이미지
- `dataset/processed/`: 전처리된 이미지
- `dataset/metadata/splits.csv`: 분할 정보
- `loadcell/fruit_weights.csv`: 개체별 무게 데이터

**평가 범위:** Test 210장은 서로 다른 **15개 과일 개체**의 14방향 사진입니다. 따라서 210장을 모두 서로 독립적인 과일이라고 해석해서는 안 됩니다.

## 3. 로드셀 캘리브레이션

실제 융합 모델에 입력할 무게값을 얻기 위해 **2점 교정(0g/400g)**을 수행했습니다.

1. **영점 교정:** 아무것도 올리지 않은 상태에서 원시값 100개를 측정하고 평균 `zero_offset`을 구합니다.
2. **감도 교정:** 기준 무게 **400g**을 올린 상태에서 원시값 100개를 측정하고 `scale_factor`를 구합니다.
3. **검증:** 알려진 다른 무게를 올려 g 단위로 올바르게 변환되는지 확인합니다.

```text
scale_factor (ADC counts/g) = (400g 상태의 원시값 평균 - zero_offset) / 400
무게(g) = (현재 원시값 - zero_offset) / scale_factor
```

- GPIO 설정은 **BCM 번호 DT=5, SCK=6**입니다(물리적 핀 번호가 아님).
- 라즈베리파이 실행 코드는 `/home/sunghopark/Loadcell_calibration/calibration.json`에 저장된 `zero_offset`, `scale_factor`를 사용합니다.
- 캘리브레이션 스크립트는 `scripts/calibration/`에 보관하며, **`calibration.json`은 장치별 교정값이므로 Git에 반드시 올릴 필요는 없습니다.**

## 4. 모델 및 경량화

### 4.1 비교 모델

| 모델 | 입력 | 구조 |
| --- | --- | --- |
| `custom_image` | 이미지 | Depthwise Separable CNN + 분류층 |
| `custom_fusion` | 이미지 + 무게 | CNN 특징 벡터 + 표준화된 무게 → 분류층 |

무게 입력은 학습 데이터 기준 평균과 표준편차로 표준화합니다.

```text
무게 표준화 = (현재 무게(g) - weight_mean) / weight_std
```

해당 통계량, 이미지 정규화 통계, 클래스 순서는 모델별 `*_deploy.json` 파일에 저장됩니다.

### 4.2 적용 최적화

- **INT8 정적 양자화:** ONNX Runtime을 이용한 모델 크기 감소 실험
- **구조적 프루닝:** 중요도가 낮은 CNN 채널을 실제로 제거하고 Fine-tuning 수행
- **프루닝 비율 실험:** 10%~80%를 10% 간격으로 비교

## 5. Raspberry Pi 5 성능 평가

### 5.1 최적화 방식 8종 비교

기본 이미지 단독 CNN과 융합 CNN에 각각 FP32, INT8, 프루닝, 프루닝+INT8을 적용한 **총 8개 모델**을 비교했습니다. 이 8개 모델 비교의 프루닝 비율은 **20%**였습니다.

| 모델 | 최적화 | Test 정확도 | 평균 순수 추론시간 | ONNX 크기 |
| --- | --- | ---: | ---: | ---: |
| 이미지 단독 | FP32 | 98.10% | 1.753 ms | 0.3812 MiB |
| 이미지 단독 | INT8 | 99.05% | 1.797 ms | 0.1488 MiB |
| 이미지 단독 | 프루닝 20% | 97.62% | 1.367 ms | 0.2497 MiB |
| 이미지 단독 | 프루닝 20% + INT8 | 97.62% | 1.660 ms | 0.1122 MiB |
| 이미지+무게 | FP32 | 100.00% | 1.763 ms | 0.3818 MiB |
| 이미지+무게 | INT8 | 100.00% | 1.835 ms | 0.1491 MiB |
| 이미지+무게 | 프루닝 20% | 100.00% | 1.379 ms | 0.2502 MiB |
| 이미지+무게 | 프루닝 20% + INT8 | 100.00% | 1.687 ms | 0.1124 MiB |

이 실험에서 **INT8은 모델 크기를 줄였지만 Raspberry Pi 5 CPU의 추론시간은 다소 늘어났습니다.** 구조적 프루닝은 추론시간을 단축했습니다.

### 5.2 프루닝 비율별 비교

| 프루닝 비율 | 이미지 단독 정확도 | 단독 평균 추론(ms) | 융합 정확도 | 융합 평균 추론(ms) |
| --- | ---: | ---: | ---: | ---: |
| 0% (원본) | 98.10% | 1.790 | 100.00% | 1.732 |
| 10% | 98.10% | 1.463 | 100.00% | 1.442 |
| 20% | 97.62% | 1.385 | 100.00% | 1.407 |
| 30% | 98.10% | 1.151 | 100.00% | 1.153 |
| **40%** | 97.62% | 1.123 | **100.00%** | **1.133** |
| **50%** | **98.57%** | **0.895** | 99.52% | 0.888 |
| 60% | 95.71% | 0.738 | 99.05% | 0.678 |
| 70% | 94.29% | 0.679 | 97.62% | 0.667 |
| 80% | 73.33% | 0.529 | 52.38% | 0.501 |

**최종 선정 모델**

| 선정 모델 | 프루닝 | 정확도 | 평균 순수 추론시간 | ONNX 크기 |
| --- | ---: | ---: | ---: | ---: |
| 이미지 단독 CNN | **50%** | **98.57%** | **0.895 ms** | **0.1063 MiB** |
| 이미지+무게 융합 CNN | **40%** | **100.00%** | **1.133 ms** | **0.1484 MiB** |

**평가 조건:** Raspberry Pi 5 CPU, ONNX Runtime, Test 210장, 모델별 워밍업 10회, 210장 × 3회 = 630회 추론, CPU 스레드 4개. **순수 ONNX 추론시간**에는 카메라 촬영·로드셀 측정·이미지 전처리·UI 갱신 시간이 포함되지 않습니다. 모델 크기는 MiB로 표기했습니다. 실행마다 약간의 측정 변동이 있을 수 있습니다.

측정 결과 CSV(저장소에 추가할 파일):

- `results/raspberry_pi/benchmark_8models.csv`: 최적화 방식 8개 비교
- `results/raspberry_pi/pruning_comparison.csv`: 프루닝 0~80% 비교

> **주의:** 저장소 루트의 `model/custom_image_pruned.onnx`, `model/custom_fusion_pruned.onnx`는 반복 실험 중 마지막 생성 파일일 수 있습니다. **프루닝 비율이 확정된 모델은 반드시 `model/pruning_runs/pXX/`에서 선택**하세요. 과거 8개 모델 비교를 다시 실행할 때는 당시 사용한 20% 프루닝 파일을 지정해야 합니다.

## 6. 프로젝트 폴더 구성

아래는 GitHub에 정리할 **목표 구조**입니다. `scripts/realtime/`, `scripts/calibration/`, `results/raspberry_pi/`는 라즈베리파이에서 파일을 복사하여 추가하는 폴더입니다.

```text
raspberrypi-citrus-fusion-ai/
├── dataset/
│   ├── raw/
│   ├── processed/
│   └── metadata/splits.csv
├── loadcell/
│   └── fruit_weights.csv
├── model/
│   ├── custom_image.onnx
│   ├── custom_fusion.onnx
│   └── pruning_runs/
│       ├── p10/ ... p80/
│       ├── p40/custom_fusion_pruned.onnx            # 최종 융합 모델
│       └── p50/custom_image_pruned.onnx             # 최종 단독 모델
├── scripts/
│   ├── citrus.py
│   ├── preprocess_images.py
│   ├── quantize_int8.py
│   ├── quantize_pruned_int8.py
│   ├── prune_models.py
│   ├── benchmark_citrus_8models.py
│   ├── calibration/                                 # 캘리브레이션 Python 코드
│   └── realtime/                                    # 실시간 UI Python 코드
├── results/
│   └── raspberry_pi/
│       ├── benchmark_8models.csv
│       └── pruning_comparison.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## 7. PC에서 개발·학습하기

저장소 루트에서 실행하세요. **Python 3.12** 가상환경을 권장합니다(PC에서 사용한 환경 기준). GPU용 PyTorch는 환경에 따라 설치 방법이 다를 수 있습니다.

```powershell
# Windows PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

데이터셋과 모델을 다시 생성하고자 할 때만 다음 명령을 사용하세요. 이미 존재하는 학습 결과를 사용한다면 재학습할 필요는 없습니다.

```powershell
python scripts/citrus.py prepare
python scripts/citrus.py train --models custom --epochs 60
python scripts/citrus.py evaluate --models custom
python scripts/citrus.py export --models custom --verify
```

양자화와 프루닝 코드:

```powershell
python scripts/quantize_int8.py
python scripts/prune_models.py --ratio 0.40 --epochs 8
python scripts/quantize_pruned_int8.py
```

- `scripts/citrus.py prepare`는 이미 `splits.csv`가 존재하면 기본적으로 기존 분할을 유지합니다.
- 프루닝 스크립트 출력은 기본적으로 `model/`에 생성됩니다. 여러 비율을 비교할 때는 각 결과를 **별도 `model/pruning_runs/pXX/`로 보관**해야 합니다.
- `--overwrite`는 기존 모델 파일을 교체하므로 주의해서 사용하세요.

## 8. Raspberry Pi 5에 배포하고 실행하기

### 8.1 필요한 파일

실시간 추론에 필요한 ONNX/설정 파일 **4개**입니다.

```text
model/pruning_runs/p50/custom_image_pruned.onnx
model/pruning_runs/p50/custom_image_pruned_deploy.json
model/pruning_runs/p40/custom_fusion_pruned.onnx
model/pruning_runs/p40/custom_fusion_pruned_deploy.json
```

PC에서 Pi로 복사하는 예시(실제 접속 가능한 IP를 사용하세요):

```powershell
ssh sunghopark@192.168.0.11 "mkdir -p ~/citrus_final"
scp .\model\pruning_runs\p50\custom_image_pruned.onnx .\model\pruning_runs\p50\custom_image_pruned_deploy.json sunghopark@192.168.0.11:citrus_final/
scp .\model\pruning_runs\p40\custom_fusion_pruned.onnx .\model\pruning_runs\p40\custom_fusion_pruned_deploy.json sunghopark@192.168.0.11:citrus_final/
```

### 8.2 Raspberry Pi 의존성

`requirements.txt`는 **PC 학습·최적화까지 포함한 전체 개발 의존성**입니다. Raspberry Pi에서 실시간 추론만 하려면 PyTorch를 설치할 필요가 없습니다.

```bash
# Raspberry Pi 터미널
source ~/citrus-venv/bin/activate
python -m pip install numpy Pillow onnxruntime opencv-python
```

HX711은 `lgpio`를 사용합니다. `lgpio`가 설치되지 않았거나 빌드 실패가 발생하면:

```bash
sudo apt update
sudo apt install -y swig liblgpio-dev
python -m pip install lgpio
```

OS 저장소에 패키지가 없다면 현재 Raspberry Pi OS에 맞는 제공 패키지를 확인해야 합니다. 카메라 연속 프레임용 Picamera2는 보통 Raspberry Pi OS 패키지(`python3-picamera2`)로 설치합니다. 단, **기존 가상환경에서 `from picamera2 import Picamera2`가 실제로 동작하는지 확인**해야 합니다. 없으면 실행 코드가 `rpicam-still` 대체 모드로 전환될 수 있으며 이 경우 영상이 느리게 갱신될 수 있습니다.

### 8.3 실시간 UI 실행

라즈베리파이의 실제 실행 폴더는 다음과 같이 유지합니다.

```text
/home/sunghopark/citrus_final/
├── custom_image_pruned.onnx
├── custom_image_pruned_deploy.json
├── custom_fusion_pruned.onnx
├── custom_fusion_pruned_deploy.json
└── (실시간 분류 Python 파일)

/home/sunghopark/Loadcell_calibration/calibration.json
```

**최종 실시간(연속 영상 + 주기적 추론 + 최근 5회 확률 평균) 버전의 파일명이 `citrus_realtime_smooth.py`라면:**

```bash
source ~/citrus-venv/bin/activate
python ~/citrus_final/citrus_realtime_smooth.py --preview
```

**라즈베리파이에 복사한 실행 파일명이 `citrus_realtime.py`라면:**

```bash
python ~/citrus_final/citrus_realtime.py --preview
```

실행 옵션은 각 파일에 따라 다를 수 있습니다. `--infer-interval 0.2 --smooth-window 5 --crop-ratio 0.95`는 `citrus_realtime_smooth.py`의 옵션입니다. 프로그램에서 실제 영상과 세 과일별 예측 비율을 함께 볼 수 있습니다.

- `citrus_realtime_smooth.py`는 두 모델을 약 **0.2초 간격**으로 실행하고 **최근 5회 예측 확률의 평균**을 표시합니다.
- **무게가 일정 기준 이상일 때만 분류하는 기능은 사용하지 않습니다.** 빈 화면에서도 계속 3종 분류 예측이 나올 수 있습니다.
- 영상 창에서 `Q`, 터미널에서는 `Ctrl+C`로 종료합니다.

### 8.4 성능 평가 재실행

`benchmark_citrus_8models.py`는 원래 라즈베리파이 홈 디렉터리에 복사된 모델과 `~/pi_test` 데이터셋을 입력으로 사용하도록 작성되었습니다. 조건을 맞춘 후 실행합니다.

```bash
source ~/citrus-venv/bin/activate
python ~/benchmark_citrus_8models.py
```

기본 결과 경로는 `~/citrus_benchmark_8models/results/comparison.csv`입니다. 프루닝 비율별 별도 실험 결과는 `~/citrus_pruning_benchmark/pruning_comparison.csv`입니다.

## 9. 확인된 한계와 개선 방향

1. **실제 촬영 환경 차이:** 조명·배경·카메라 구도가 학습 이미지와 다르면 귤을 오렌지로 오분류하는 경우가 있습니다.
2. **Unknown 미지원:** 3종 외의 물체를 거부하는 별도 분류 클래스나 검출기가 없습니다.
3. **일반화 검증 필요:** Test 210장은 과일 개체 15개의 다중 촬영 이미지이므로 더 많은 미관측 개체로 평가할 필요가 있습니다.
4. **종단 간 지연시간 미측정:** 모델 추론시간과 실제 카메라→센서→UI 전체 처리시간을 구분해야 합니다.
5. **무게 교정 의존성:** 로드셀 재설치·받침대 변경 시 재교정이 필요할 수 있습니다.

## 10. 오픈소스 및 라이선스

주요 사용 소프트웨어는 Python, PyTorch, torchvision, NumPy, pandas, OpenCV, Pillow, ONNX, ONNX Runtime, lgpio, Picamera2(지원 환경) 및 Git입니다. 각 외부 라이브러리에는 고유 라이선스가 적용됩니다. 저장소 자체의 라이선스는 별도 `LICENSE` 파일이 명시되지 않았다면 자동으로 오픈소스 라이선스가 부여되는 것이 아닙니다.
