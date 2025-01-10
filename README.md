# Financial Analysis System (FinAnaSys)

고급 금융 분석 도구를 활용한 데이터 기반 의사결정 시스템

## 프로젝트 개요

이 프로젝트는 Next.js와 Python을 결합하여 다양한 금융 분석 도구를 제공하는 웹 애플리케이션입니다. 사용자는 직관적인 대시보드를 통해 다양한 금융 분석을 수행할 수 있습니다.

## 주요 기능

1. **워드 클라우드 분석**

   - 텍스트 데이터 시각적 분석
   - 주요 키워드 추출
   - 단어 빈도 시각화

2. **배당 분석**

   - 최근 10년 배당금 이력 분석
   - 연간 배당률 계산
   - 배당 성장률 추세 확인

3. **포트폴리오 최적화**

   - 최대 샤프 비율 포트폴리오
   - 최소 변동성 포트폴리오
   - Risk Parity & Black-Litterman

4. **퀀텀 분석**

   - 슈뢰딩거 방정식 기반 분석
   - 가격 확률 분포 예측
   - 신뢰구간별 가격 범위 제시

5. **변동성 분석**

   - 베이지안과 에르고딕 변동성 측정
   - GARCH 모델 기반 분석
   - VaR(Value at Risk) 산출

6. **재무-기술적 분석**
   - AI 기반 전략 지표 분석
   - 추세 패턴 식별
   - 매매 신호 제공

## 기술 스택

### Frontend

- Next.js 13.5.8
- React 18.2.0
- CSS Modules
- Noto Sans KR 폰트

### Backend

- Python 3.9
- Streamlit
- 각종 데이터 분석 라이브러리

## 설치 방법

1. 저장소 클론

```bash
git clone [repository-url]
cd FinAnaSys
```

2. 의존성 설치

```bash
npm install
```

3. Python 패키지 설치

```bash
pip install -r requirements.txt
```

## 실행 방법

개발 모드로 실행:

```bash
npm run dev
```

프로덕션 빌드:

```bash
npm run build
npm start
```

## 시스템 요구사항

- Node.js >= 14.0.0
- Python >= 3.9
- npm 또는 yarn

## 프로젝트 구조

```
FinAnaSys/
├── components/         # React 컴포넌트
├── pages/             # Next.js 페이지
│   ├── api/          # API 라우트
│   └── _document.js  # 커스텀 문서
├── public/           # 정적 파일
├── styles/           # CSS 스타일
└── *.py             # Python 분석 스크립트
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다.

## 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 라의사항

프로젝트에 대한 문의사항이나 버그 리포트는 GitHub Issues를 통해 제출해 주세요.
