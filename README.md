# Financial Analysis System (FinAnaSys)

금융 데이터 분석을 위한 종합 웹 애플리케이션입니다. 다양한 분석 도구와 AI 기반 예측 모델을 제공합니다.

## 주요 기능

### 1. 머신러닝 분석

- 딥러닝 모델 기반 예측
- 시계열 패턴 분석
- AI 기반 투자 전략

### 2. 배당 분석

- 최근 10년 배당금 이력 분석
- 연간 배당률 계산
- 배당 성장률 추세 확인

### 3. 포트폴리오 최적화

- 최대 샤프 비율 포트폴리오
- 최소 변동성 포트폴리오
- Risk Parity & Black-Litterman

### 4. 퀀텀 분석

- 슈뢰딩거 방정식 기반 분석
- 가격 확률 분포 예측
- 신뢰구간별 가격 범위 제시

### 5. 변동성 분석

- 베이지안과 에르고딕 변동성 측정
- GARCH 모델 기반 분석
- VaR(Value at Risk) 산출

### 6. 재무-기술적 분석

- 전략 지표 분석
- 추세 패턴 식별
- 매매 신호 제공

### 7. 워드 클라우드 분석

- 텍스트 파일 분석
- 주요 키워드 추출
- 단어 빈도 시각화

## 시스템 요구사항

- Node.js 18.0.0 이상
- Python 3.8.0 이상
- pip (Python 패키지 관리자)

## 설치 방법

1. 저장소 클론

```bash
git clone https://github.com/yourusername/FinAnaSys.git
cd FinAnaSys
```

2. Node.js 패키지 설치

```bash
npm install
```

3. Python 패키지 설치

```bash
pip install -r requirements.txt
```

## 실행 방법

1. 개발 모드 실행

```bash
npm run dev
```

2. 프로덕션 모드 실행

```bash
npm run build
npm start
```

기본적으로 웹 애플리케이션은 http://localhost:5050 에서 실행됩니다.

## 분석 모듈 포트

각 분석 모듈은 다음 포트에서 실행됩니다:

- 머신러닝 분석: 8507
- 배당 분석: 8501
- 포트폴리오 최적화: 8502
- 퀀텀 분석: 8503
- 변동성 트레이딩 분석: 8504
- 재무-기술적 분석: 8505
- 워드 클라우드 분석: 8506

## 기술 스택

### Frontend

- Next.js
- React
- CSS Modules

### Backend

- Node.js
- Python
- Streamlit

### 분석 도구

- NumPy
- Pandas
- Scikit-learn
- PyTorch
- Plotly
- Matplotlib
- NLTK

## 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 문의사항

버그 리포트나 기능 요청은 이슈 트래커를 사용해 주세요.
