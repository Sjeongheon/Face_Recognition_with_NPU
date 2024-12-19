# Face_Recognition_with_NPU
이 저장소는 **차량용 도난 감시 플랫폼** 프로젝트의 핵심 모듈 중 하나인 **실시간 얼굴 인식 파이프라인**을 담고 있습니다.
전체 프로젝트는 **두 개의 보드(D3 보드 & NPU 보드)를 연동**하여 동작하는 구조를 가지며, 본 저장소는 그중 NPU 보드에서 동작할 인공지능 모델을 튜닝하고 검증하는 목적의 코드를 포함하고 있는 서브 프로젝트입니다.

## 프로젝트 구조

안전하고 빠른 차량 보안 및 제어를 위해 연산을 분산하는 이원화 아키텍처를 채택했습니다.

* **NPU 보드 (연산 전담):** NPU(Neural Processing Unit)를 활용해 카메라로부터 들어오는 비디오 스트림에서 실시간으로 얼굴을 검출하고 인가된 사용자인지 인식합니다.
* **D3 보드 (제어 전담):** NPU 보드로부터 얼굴 인식 결과를 전달받아, 최종적으로 차량의 시동을 걸거나 제어권을 획득(또는 차단)하는 차량 통신 및 하드웨어 제어를 수행합니다.

> 본 저장소에 포함된 코드는 NPU 타겟 보드에 직접 포팅하기 전, **Host PC 환경**에서 ArcFace(IR-SE50) 모델의 그래프 변환, ONNX 출력 및 특징 추출 성능을 검증한 테스트(PoC) 결과를 담고 있습니다. 실제 NPU SDK 및 보드 실행은 별도의 타겟 환경에서 확인해야 합니다.

---

### 1. NPU 및 양자화를 위한 모델 변환

* **MaxPool2d ➡️ Depthwise Conv2d:** `1x1 MaxPool2d` shortcut을 가중치가 1로 고정된 Depthwise `Conv2d` subsampling으로 변환
* **Linear + BatchNorm1d ➡️ Conv2d:** `Flatten + Linear + BatchNorm1d`를 BatchNorm이 fold된 spatial `Conv2d`로 변환
* **PReLU 등가 분해:** 기존 채널별 slope를 보존하면서 `PReLU`를 `ReLU`, `Sub`, `Mul`, `Add` 연산 조합으로 변환
* **PReLU ➡️ LeakyReLU 근사:** 채널별 slope의 평균을 사용하는 `LeakyReLU`로 대체하며, 다채널 PReLU에서는 비등가 변환
* **BatchNorm graph workaround:** 단독 `BatchNorm2d` 앞에 가중치가 1인 identity Depthwise `Conv2d`를 추가
* **SE module 제거:** `SEModule`을 `Identity`로 교체하는 비등가 ablation

위 변환은 원본 `baseline`, 등가 변환을 적용한 `exact`, PReLU 근사 실험인 `prelu_approx`, `exact` 기반의 `bn_workaround`, SE를 제거한 `no_se` variant로 구분됩니다.

### 2. 특징 추출 및 얼굴 검증 (LFW / FAISS)

* 전처리된 LFW 데이터셋 12,000장을 바탕으로 512차원의 특징(Feature) 벡터 추출
* 각 fold의 학습 쌍에서 threshold를 결정하는 LFW 10-fold same/different verification 수행
* **t-SNE 시각화:** sampled pair embedding 분포를 정성적으로 확인
* **FAISS 검색:** `faiss.IndexFlatIP`를 이용한 cosine-similarity 기반 Top-K 검색
* **ONNX 검증:** `prelu_approx`의 PyTorch/ONNX Runtime 출력 최대 절댓값 차이 약 `1.56e-7`

`ArcFace_model_conversion_and_verification.ipynb`에 기록된 전체 LFW 결과는 다음과 같습니다.

| Variant | LFW 10-fold accuracy |
| --- | ---: |
| `baseline` | `0.9958 ± 0.0030` |
| `exact` | `0.9958 ± 0.0030` |
| `prelu_approx` | `0.9955 ± 0.0026` |

## NPU Deployment Pipeline

이 저장소에서 검증된 ONNX 모델을 실제 **NPU 보드(예: TOPST AI-G)** 에 배포하기 위해 다음과 같은 OPENEDGES SDK(ENLIGHT Toolkit) 파이프라인을 거치게 됩니다.

1. **Export ONNX**

   ```bash
   face-npu-ai export --checkpoint weights/model_ir_se50.pth --variant prelu_approx --output arcface_prelu_approx.onnx
   ```

2. **Convert & Quantize**

   ```bash
   # .enlight 포맷 변환 및 Activation 통계 추출
   python ./EnlightSDK/converter.py arcface_prelu_approx.onnx --dataset Custom --dataset-root <dataset_path> --output arcface.enlight --enable-track

   # INT8 양자화
   python ./EnlightSDK/quantizer.py arcface.enlight --output arcface_quantized.enlight
   ```

3. **Compile & Build Network Object**

   ```bash
   python ./EnlightSDK/compiler.py arcface_quantized.enlight
   ```

4. **Deploy & Run (tc-nn-app)**

   컴파일된 바이너리 파일과 네트워크 오브젝트 파일을 NPU 보드(TOPST AI-G)로 복사 및 실행합니다.

   ```bash
   tcnnapp -n /arcface_network/ -i camera -o display
   ```
