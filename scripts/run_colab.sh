#!/bin/bash

# 전체 트레이딩 파이프라인 실행 스크립트
# 사용법: ./scripts/run_all.sh [SYMBOL] [INTERVAL] [START_DATE] [END_DATE]

set -e  # 에러 발생시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 기본값 설정
SYMBOL=${1:-BTCUSDT}
INTERVAL=${2:-1h}
TRAIN_START=${3:-2025-01-01}
TRAIN_END=${4:-2025-10-31}
TEST_START=${5:-2025-11-01}
TEST_END=${6:-2025-12-30}

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  LSTM 트레이딩 자동화 파이프라인${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}심볼:${NC}           $SYMBOL"
echo -e "${GREEN}간격:${NC}           $INTERVAL"
echo -e "${YELLOW}[학습 기간]${NC}"
echo -e "${GREEN}  시작일:${NC}       $TRAIN_START"
echo -e "${GREEN}  종료일:${NC}       $TRAIN_END"
echo -e "${YELLOW}[백테스트 기간]${NC}"
echo -e "${GREEN}  시작일:${NC}       $TEST_START"
echo -e "${GREEN}  종료일:${NC}       $TEST_END"
echo -e "${BLUE}============================================================${NC}\n"

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# 가상환경 활성화
# if [ ! -d "venv" ]; then
#     echo -e "${RED}❌ 가상환경을 찾을 수 없습니다. 먼저 'python -m venv venv'를 실행하세요.${NC}"
#     exit 1
# fi

# echo -e "${YELLOW}🔧 가상환경 활성화...${NC}"
# source venv/bin/activate

# 1. 학습 데이터 다운로드
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${YELLOW}📥 Step 1/4: 학습 데이터 다운로드${NC}"
echo -e "${BLUE}============================================================${NC}"

python scripts/download_data.py \
    --symbol "$SYMBOL" \
    --interval "$INTERVAL" \
    --start "$TRAIN_START" \
    --end "$TRAIN_END" \
    --force

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 학습 데이터 다운로드 실패${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 학습 데이터 다운로드 완료${NC}"

# 2. 백테스트 데이터 다운로드
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${YELLOW}📥 Step 2/4: 백테스트 데이터 다운로드 (Out-of-Sample)${NC}"
echo -e "${BLUE}============================================================${NC}"

python scripts/download_data.py \
    --symbol "$SYMBOL" \
    --interval "$INTERVAL" \
    --start "$TEST_START" \
    --end "$TEST_END"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 백테스트 데이터 다운로드 실패${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 백테스트 데이터 다운로드 완료${NC}"

# 3. 모델 학습
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${YELLOW}🧠 Step 3/4: LSTM 모델 학습 (Train+Val)${NC}"
echo -e "${BLUE}============================================================${NC}"

python scripts/train_model.py \
    --symbol "$SYMBOL" \
    --interval "$INTERVAL" \
    --config configs/strategy.yaml \
    --features-config configs/features.yaml \
    --model-path "models/lstm/model.pth" \
    --scaler-path "models/lstm/scaler.pkl"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 모델 학습 실패${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 모델 학습 완료${NC}"

# 4. 백테스트 실행 (Out-of-Sample)
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${YELLOW}📊 Step 4/4: 백테스트 실행 (Out-of-Sample 데이터)${NC}"
echo -e "${BLUE}============================================================${NC}"

# backtest.yaml을 동적으로 업데이트
BACKTEST_CONFIG="configs/backtest.yaml"
if [ -f "$BACKTEST_CONFIG" ]; then
    # 백테스트 기간을 임시로 변경
    sed -i.bak "s/start_date: .*/start_date: \"$TEST_START\"/" "$BACKTEST_CONFIG"
    sed -i.bak "s/end_date: .*/end_date: \"$TEST_END\"/" "$BACKTEST_CONFIG"
fi

python scripts/run_backtest.py \
    --symbol "$SYMBOL" \
    --interval "$INTERVAL" \
    --strategy-path "models/lstm/model_strategy.pth" \
    --scaler-path "models/lstm/scaler.pkl" \
    --features-config configs/features.yaml \
    --backtest-config configs/backtest.yaml \
    --output-dir results

# 원본 설정 복원
if [ -f "${BACKTEST_CONFIG}.bak" ]; then
    mv "${BACKTEST_CONFIG}.bak" "$BACKTEST_CONFIG"
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 백테스트 실패${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 백테스트 완료${NC}"

# 최종 결과 표시
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${GREEN}✨ 전체 파이프라인 완료!${NC}"
echo -e "${BLUE}============================================================${NC}"

# 최신 결과 디렉토리 찾기
LATEST_RESULT=$(ls -td results/*/ 2>/dev/null | head -n1)

if [ -n "$LATEST_RESULT" ]; then
    echo -e "${GREEN}📁 결과 위치:${NC} $LATEST_RESULT"
    echo -e "\n${YELLOW}생성된 파일:${NC}"
    ls -lh "$LATEST_RESULT"
    
    # metrics.csv가 있으면 간단한 요약 출력
    if [ -f "${LATEST_RESULT}metrics.csv" ]; then
        echo -e "\n${BLUE}📈 간단 요약:${NC}"
        echo -e "${GREEN}-----------------------------------------------------------${NC}"
        head -2 "${LATEST_RESULT}metrics.csv" | column -t -s','
        echo -e "${GREEN}-----------------------------------------------------------${NC}"
    fi
fi

echo -e "\n${BLUE}💡 다음 단계:${NC}"
echo -e "  - 결과 확인: ${YELLOW}cat ${LATEST_RESULT}metrics.csv${NC}"
echo -e "  - 차트 보기: ${YELLOW}ls ${LATEST_RESULT}*.png${NC}"
echo -e "  - 리포트 생성: ${YELLOW}python scripts/generate_report.py --result-dir ${LATEST_RESULT}${NC}"

echo -e "\n${GREEN}🎉 완료!${NC}\n"
