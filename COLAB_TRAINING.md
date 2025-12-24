# Colab에서 모델 학습하기

Google Colab에서 무료 GPU를 사용하여 LSTM 모델을 학습하는 가이드입니다.

## 방법 1: Jupyter 노트북 사용 (추천)

1. **Colab 열기**
   - [Google Colab](https://colab.research.google.com/) 접속
   - `File > Upload notebook` 클릭
   - `/workspaces/trading_lstm/notebooks/train_on_colab.ipynb` 업로드

2. **GPU 활성화**
   - `Runtime > Change runtime type` 클릭
   - `Hardware accelerator`를 `GPU` 또는 `T4 GPU`로 선택
   - `Save` 클릭

3. **노트북 실행**
   - 위에서부터 순서대로 셀 실행 (Shift + Enter)
   - 모든 과정이 자동으로 진행됩니다

4. **결과 다운로드**
   - 마지막 셀이 실행되면 `trained_model.zip` 자동 다운로드
   - 또는 파일 브라우저에서 수동 다운로드

## 방법 2: 수동 명령어 실행

Colab에서 새 노트북을 만들고 아래 명령어를 순서대로 실행:

### 1. 저장소 클론
```python
!git clone https://github.com/edenkim00/trading_lstm.git
%cd trading_lstm
```

### 2. 패키지 설치
```python
!pip install -q -r requirements.txt
```

### 3. 데이터 다운로드
```python
!python scripts/download_data.py --symbol BTCUSDT --interval 1h
```

### 4. 모델 학습
```python
!python scripts/train_model.py \
    --symbol BTCUSDT \
    --interval 1h \
    --model-path models/lstm/model.pth \
    --scaler-path models/lstm/scaler.pkl
```

### 5. 결과 압축 및 다운로드
```python
# 압축
!zip -r trained_model.zip models/lstm/

# 다운로드
from google.colab import files
files.download('trained_model.zip')
```

## Codespace에 모델 가져오기

### 옵션 A: 직접 업로드
1. Colab에서 다운로드한 `trained_model.zip` 파일 준비
2. Codespace의 Explorer에서 파일을 드래그 앤 드롭
3. 터미널에서 압축 해제:
```bash
cd /workspaces/trading_lstm
unzip trained_model.zip
```

### 옵션 B: GitHub 사용
Colab에서 GitHub에 직접 푸시:

```python
# Colab에서 실행
!git config user.email "your-email@example.com"
!git config user.name "Your Name"
!git add models/lstm/*.pth models/lstm/*.pkl
!git commit -m "Add trained LSTM model from Colab"

# GitHub Personal Access Token 사용
!git push https://YOUR_TOKEN@github.com/edenkim00/trading_lstm.git main
```

Codespace에서 풀:
```bash
git pull origin main
```

### 옵션 C: Google Drive 사용
Colab에서:
```python
from google.colab import drive
drive.mount('/content/drive')

# 모델을 Google Drive에 복사
!cp -r models/lstm /content/drive/MyDrive/trading_lstm_models/
```

Codespace에서 Google Drive 링크로 다운로드 (공유 설정 후)

## 학습 후 백테스트

모델을 Codespace에 가져온 후:

```bash
python scripts/run_backtest.py \
    --symbol BTCUSDT \
    --strategy-path models/lstm/model.pth \
    --scaler-path models/lstm/scaler.pkl
```

## 팁

### 학습 시간 절약
- Colab Pro 사용 시 더 강력한 GPU (A100, V100) 사용 가능
- 무료 버전은 12시간 제한이 있으니 주의

### 데이터 크기 조정
시간이 오래 걸리면 `configs/data.yaml`에서 데이터 기간 조정:
```yaml
start_date: "2024-01-01"  # 더 짧은 기간으로
end_date: "2024-12-31"
```

### 학습 모니터링
- 학습 중 loss와 metrics 확인
- TensorBoard 사용 가능 (추가 설정 필요)

## 문제 해결

### GPU가 할당되지 않음
- Runtime을 재시작하고 GPU 설정 다시 확인
- 무료 할당량 소진 시 나중에 다시 시도

### 메모리 부족
- `configs/strategy.yaml`에서 배치 사이즈 줄이기:
```yaml
training:
  batch_size: 32  # 64에서 32로 줄이기
```

### 패키지 설치 오류
- requirements.txt에서 버전 호환성 확인
- Colab의 기본 패키지와 충돌 시 버전 명시

## 참고
- [Google Colab 공식 문서](https://colab.research.google.com/)
- [Colab GPU 사용 가이드](https://colab.research.google.com/notebooks/gpu.ipynb)
