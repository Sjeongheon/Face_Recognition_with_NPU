# Face_Recognition_with_NPU
이 저장소는 **차량용 도난 감시 플랫폼** 프로젝트의 핵심 모듈 중 하나인 **실시간 얼굴 인식 파이프라인**을 담고 있습니다. 
전체 프로젝트는 **두 개의 보드(D3 보드 & NPU 보드)를 연동**하여 동작하는 구조를 가지며, 본 저장소는 그중 NPU 보드에서 동작할 인공지능 모델을 튜닝하고 검증하는 목적의 코드를 포함하고 있는 서브 프로젝트입니다.

## 프로젝트 구조

안전하고 빠른 차량 보안 및 제어를 위해 연산을 분산하는 이원화 아키텍처를 채택했습니다.

* **NPU 보드 (연산 전담):** NPU(Neural Processing Unit)를 활용해 카메라로부터 들어오는 비디오 스트림에서 실시간으로 얼굴을 검출하고 인가된 사용자인지 인식합니다.
* **D3 보드 (제어 전담):** NPU 보드로부터 얼굴 인식 결과를 전달받아, 최종적으로 차량의 시동을 걸거나 제어권을 획득(또는 차단)하는 차량 통신 및 하드웨어 제어를 수행합니다.

> 본 저장소에 포함된 코드는 NPU 타겟 보드에 직접 포팅하기 전, **Host PC 환경**에서 ArcFace(IR-SE50) 모델을 NPU 하드웨어 특성에 맞게 튜닝하고 양자화 시뮬레이션 및 특징 추출 성능을 검증한 테스트(PoC) 결과를 담고 있습니다.

---

### 1. NPU 및 양자화를 위한 모델 튜닝
* **BatchNorm2D 구조 변경:** BatchNorm을 가중치가 1인 `Conv2D`와 결합하거나 고정(Static)하는 형태로 우회
* **PReLU ➡️ LeakyReLU:** NPU 연산에 적합하도록 기존 PReLU 레이어의 가중치 평균을 계산해 `LeakyReLU`로 일괄 대체
* **Linear ➡️ Conv2D:** `Linear` 레이어를 `kernel_size=7`인 `Conv2D`로 변환하여 기존 가중치 복사

### 2. 특징 추출 및 얼굴 유사도 검색 검증 (FAISS)
* 전처리된 LFW 데이터셋 12,000장을 바탕으로 512차원의 특징(Feature) 벡터 추출
* **t-SNE 시각화:** 추출된 특징들이 2차원 공간에서 동일 인물끼리 군집화(Clustering)되는지 시각적으로 검증
* **FAISS 검색:** `faiss.IndexFlatIP` (내적 기반)를 이용해 쿼리 얼굴과 가장 유사한 Top-6 얼굴 이미지를 초고속으로 검색해내는 시스템 테스트 완료

## NPU Deployment Pipeline

이 저장소에서 검증된 ONNX 모델을 실제 **NPU 보드(예: TOPST AI-G)** 에 배포하기 위해 다음과 같은 OPENEDGES SDK(ENLIGHT Toolkit) 파이프라인을 거치게 됩니다.

1. **Export ONNX**
   
   `export_to_onnx.py`를 통해 구조가 변경된 모델 추출
   
2. **Convert & Quantize**
   
   ```bash
   # .enlight 포맷 변환 및 Activation 통계 추출
   python ./EnlightSDK/converter.py arcface.onnx --dataset Custom --dataset-root <dataset_path> --output arcface.enlight --enable-track
   
   # INT8 양자화
   python ./EnlightSDK/quantizer.py arcface.enlight --output arcface_quantized.enlight
   
3. **Compile & Build Network Object**
   
   ```bash
   python ./EnlightSDK/compiler.py arcface_quantized.enlight
   
4. **Deploy & Run (tc-nn-app)**
   
   컴파일된 바이너리 파일과 방금 빌드한 네트워크 오브젝트 파일을 NPU 보드(TOPST AI-G)로 복사 및 실행
   ```bash
   tcnnapp -n /arcface_network/ -i camera -o display
