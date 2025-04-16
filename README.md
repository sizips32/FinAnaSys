# FinAnaSys (Financial Analysis System)

## 소개

FinAnaSys는 금융 데이터 분석을 위한 종합 플랫폼입니다. 이 시스템은 Next.js와 Python을 결합하여 강력한 금융 데이터 분석 도구를 제공합니다.

## 주요 기능

- 📈 포트폴리오 분석
- 📊 변동성 분석
- 💰 배당 분석
- 🤖 AI 기반 기술적 분석
- 📉 퀀텀 분석
- 🔍 워드클라우드 기반 시장 동향 분석

## 기술 스택

### Frontend

- Next.js
- TypeScript
- Tailwind CSS
- React
- Headless UI

### Backend

- Python
- Flask
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- YFinance
- FinanceDataReader

## 시작하기

### 사전 요구사항

- Node.js 18.0.0 이상
- Python 3.11.6 이상
- npm 또는 yarn
- pip

### 설치 방법

1. 저장소 클론

```bash
git clone [repository-url]
cd FinAnaSys
```

2. Frontend 의존성 설치

```bash
npm install
# 또는
yarn install
```

3. Python 의존성 설치

```bash
pip install -r requirements.txt
```

4. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 필요한 환경 변수를 설정하세요
```

### 실행 방법

1. 개발 서버 실행

```bash
npm run dev
# 또는
yarn dev
```

2. 브라우저에서 확인

```
http://localhost:3000
```

## 프로젝트 구조

```
FinAnaSys/
├── components/          # React 컴포넌트
├── pages/              # Next.js 페이지
│   └── api/           # API 라우트
├── styles/            # 스타일 파일
├── public/            # 정적 파일
└── python/            # Python 백엔드 코드
```

## API 문서

- `/api/analyze/[type]` - 다양한 유형의 금융 분석 API
- `/api/run-streamlit` - Streamlit 대시보드 실행 API

## 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 연락처

- 이메일: [your-email@example.com]
- 프로젝트 링크: [https://github.com/username/FinAnaSys]

## 감사의 글

이 프로젝트는 다음과 같은 오픈 소스 프로젝트들의 도움을 받았습니다:

- [Next.js](https://nextjs.org/)
- [React](https://reactjs.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Python](https://www.python.org/)
- [FinanceDataReader](https://github.com/FinanceData/FinanceDataReader)
- [YFinance](https://github.com/ranaroussi/yfinance)
