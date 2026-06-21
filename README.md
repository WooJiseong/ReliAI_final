# Assignment 4
데이터셋 : SpeechCommands (10개 Class)를 ResNet 기반으로 학습 후 alpha-beta-CROWN을 사용해 log-mel feature 공간에서의 robustness 검증

## 1. 설치

```bash
pip install -r requirements.txt
```

alpha-beta-CROWN은 `alpha-beta-CROWN/README.md` 참고

```text
alpha-beta-CROWN/.venv/bin/python
```

## 2. 학습

SpeechCommands 데이터셋으로 `AudioCResNet5`를 학습한다.

```bash
python test.py --data-root data/SpeechCommands --download --epochs 10 --checkpoint model/AudioCResNet5.pt
```

## 3. Accuracy 평가

```bash
python test.py --data-root data/SpeechCommands --checkpoint model/AudioCResNet5.pt --eval-only
```

## 4. alpha-beta-CROWN으로 검증

기본 검증 실행:

```bash
bash run_abcrown.sh
```

validation set에서 class 균형 샘플을 사용:

```bash
ABCROWN_SAMPLE_COUNT=20 ABCROWN_SAMPLE_STRATEGY=balanced bash run_abcrown.sh
```

## 5. 검증 설정 변경


```text
verification/audio_cresnet5.yaml
```
에 있는 파라미터를 변경하여 실험을 진행한다.
주된 변경사항은

```yaml
data:
  start: 0
  end: 20

specification:
  norm: .inf
  epsilon: 0.007

bab:
  timeout: 20
```

`safe-incomplete`는 BaB 단계에 들어가기 전에 incomplete CROWN/alpha-CROWN
bound만으로 property가 증명된 경우를 의미한다. `safe`는 BaB 탐색까지 수행하여
안전성을 증명한 경우이다. `unknown`은 주어진 timeout 안에서 safe를 증명하지도,
unsafe counterexample을 찾지도 못한 경우이다.

`.wav` 파일은 원본 SpeechCommands 음성 파일이다. `.png` 파일은 verifier가
사용한 32x32 log-mel feature 시각화이다.
