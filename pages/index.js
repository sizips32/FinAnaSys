import Head from 'next/head';
import { useState } from 'react';

export default function Home() {
  const [outputs, setOutputs] = useState({});
  const [loading, setLoading] = useState({});

  const startAnalysis = async (type) => {
    setLoading(prev => ({ ...prev, [type]: true }));
    try {
      const response = await fetch(`/api/analyze/${type}`, {
        method: 'POST'
      });
      const data = await response.json();
      setOutputs(prev => ({ ...prev, [type]: data.message }));
    } catch (error) {
      setOutputs(prev => ({ ...prev, [type]: `Error: ${error.message}` }));
    } finally {
      setLoading(prev => ({ ...prev, [type]: false }));
    }
  };

  return (
    <div>
      <Head>
        <title>Financial Analysis Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet" />
      </Head>

      <style jsx global>{`
        :root {
          --gradient-primary: linear-gradient(135deg, #B8860B 0%, #DAA520 50%, #FFD700 100%);
          --gradient-card: linear-gradient(135deg, rgba(135, 206, 235, 0.25) 0%, rgba(135, 206, 235, 0.15) 100%);
          --accent-color: #4FB3E8;
          --text-primary: #ffffff;
          --text-secondary: rgba(255,255,255,0.9);
          --card-shadow: 0 8px 32px 0 rgba(79, 179, 232, 0.3);
        }

        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        body {
          font-family: 'Noto Sans KR', sans-serif;
          background: var(--gradient-primary);
          color: var(--text-primary);
          min-height: 100vh;
          line-height: 1.6;
        }

        .header {
          padding: 2rem;
          text-align: center;
          position: relative;
          overflow: hidden;
          margin-bottom: 1rem;
        }

        .header::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
          transform: skewY(-5deg);
          z-index: 0;
        }

        .header h1 {
          font-size: 2.5rem;
          font-weight: 700;
          margin-bottom: 0.5rem;
          position: relative;
          z-index: 1;
          background: linear-gradient(to right, #fff, #FFD700);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .container {
          max-width: 100%;
          margin: 0 auto;
          padding: 1rem;
        }

        .analysis-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1.5rem;
          margin-top: 1rem;
          padding: 0 1rem;
        }

        .analysis-card {
          background: var(--gradient-card);
          backdrop-filter: blur(16px);
          -webkit-backdrop-filter: blur(16px);
          border-radius: 15px;
          padding: 1.5rem;
          box-shadow: var(--card-shadow);
          border: 1px solid rgba(135, 206, 235, 0.3);
          transition: transform 0.3s ease, box-shadow 0.3s ease;
          height: 100%;
          display: flex;
          flex-direction: column;
        }

        .analysis-card:hover {
          transform: translateY(-5px);
          box-shadow: 0 12px 40px 0 rgba(79, 179, 232, 0.4);
        }

        .analysis-card h2 {
          color: #2E8B57;
          margin-bottom: 1rem;
          font-size: 1.4rem;
          font-weight: 700;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .analysis-description {
          color: var(--text-secondary);
          margin: 0.5rem 0;
          flex-grow: 1;
        }

        .analysis-details {
          margin-top: 0.5rem;
          padding-left: 1rem;
          border-left: 3px solid var(--accent-color);
          color: var(--text-secondary);
          font-size: 0.9rem;
        }

        .run-button {
          width: 100%;
          padding: 0.8rem;
          margin-top: 1rem;
          border: none;
          border-radius: 8px;
          background: linear-gradient(45deg, #4FB3E8, #87CEEB);
          color: white;
          font-weight: 500;
          font-size: 1rem;
          cursor: pointer;
          transition: all 0.3s ease;
        }

        .run-button:hover {
          background: linear-gradient(45deg, #87CEEB, #B0E2FF);
          transform: translateY(-2px);
          box-shadow: 0 5px 15px rgba(79, 179, 232, 0.4);
        }

        .output-area {
          margin-top: 1rem;
          padding: 0.8rem;
          background: rgba(0,0,0,0.15);
          border-radius: 8px;
          color: var(--text-secondary);
          font-family: monospace;
          min-height: 50px;
          white-space: pre-wrap;
        }

        @media (max-width: 1200px) {
          .analysis-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }

        @media (max-width: 768px) {
          .analysis-grid {
            grid-template-columns: 1fr;
          }
          .header h1 {
            font-size: 2rem;
          }
        }
      `}</style>

      <header className="header">
        <h1>Financial Analysis Dashboard</h1>
        <p>고급 금융 분석 도구를 활용한 데이터 기반 의사결정 시스템</p>
      </header>

      <div className="container">
        <div className="analysis-grid">
          <div className="analysis-card">
            <h2>워드 클라우드 분석</h2>
            <div className="analysis-description">
              텍스트 데이터를 시각적으로 분석합니다.
              <div className="analysis-details">
                • 텍스트 파일 분석<br />
                • 주요 키워드 추출<br />
                • 단어 빈도 시각화
              </div>
            </div>
            <button 
              className="run-button" 
              onClick={() => startAnalysis('wordcloud')}
              disabled={loading.wordcloud}
            >
              {loading.wordcloud ? '분석 중...' : '분석 시작'}
            </button>
            <div className="output-area">{outputs.wordcloud}</div>
          </div>

          <div className="analysis-card">
            <h2>배당 분석</h2>
            <div className="analysis-description">
              기업의 배당 정책과 수익률을 분석합니다.
              <div className="analysis-details">
                • 최근 10년 배당금 이력 분석<br />
                • 연간 배당률 계산<br />
                • 배당 성장률 추세 확인
              </div>
            </div>
            <button 
              className="run-button" 
              onClick={() => startAnalysis('dividend')}
              disabled={loading.dividend}
            >
              {loading.dividend ? '분석 중...' : '분석 시작'}
            </button>
            <div className="output-area">{outputs.dividend}</div>
          </div>

          <div className="analysis-card">
            <h2>포트폴리오 최적화</h2>
            <div className="analysis-description">
              최적의 자산 배분 분석을 수행합니다.
              <div className="analysis-details">
                • 최대 샤프 비율 포트폴리오<br />
                • 최소 변동성 포트폴리오<br />
                • Risk Parity & Black-Litterman
              </div>
            </div>
            <button 
              className="run-button" 
              onClick={() => startAnalysis('portfolio')}
              disabled={loading.portfolio}
            >
              {loading.portfolio ? '분석 중...' : '분석 시작'}
            </button>
            <div className="output-area">{outputs.portfolio}</div>
          </div>

          <div className="analysis-card">
            <h2>퀀텀 분석</h2>
            <div className="analysis-description">
              양자역학 기반의 가격 예측 모델을 제공합니다.
              <div className="analysis-details">
                • 슈뢰딩거 방정식 기반 분석<br />
                • 가격 확률 분포 예측<br />
                • 신뢰구간별 가격 범위 제시
              </div>
            </div>
            <button 
              className="run-button" 
              onClick={() => startAnalysis('quantum')}
              disabled={loading.quantum}
            >
              {loading.quantum ? '분석 중...' : '분석 시작'}
            </button>
            <div className="output-area">{outputs.quantum}</div>
          </div>

          <div className="analysis-card">
            <h2>변동성 분석</h2>
            <div className="analysis-description">
              주가의 변동성과 리스크를 분석합니다.
              <div className="analysis-details">
                • 베이지안과 에르고딕 변동성 측정<br />
                • GARCH 모델 기반 분석<br />
                • VaR(Value at Risk) 산출
              </div>
            </div>
            <button 
              className="run-button" 
              onClick={() => startAnalysis('volatility')}
              disabled={loading.volatility}
            >
              {loading.volatility ? '분석 중...' : '분석 시작'}
            </button>
            <div className="output-area">{outputs.volatility}</div>
          </div>

          <div className="analysis-card">
            <h2>재무-기술적 분석</h2>
            <div className="analysis-description">
              AI 기반의 재무-기술적 분석을 제공합니다
              <div className="analysis-details">
                • 전략 지표 분석<br />
                • 추세 패턴 식별<br />
                • 매매 신호 제공
              </div>
            </div>
            <button 
              className="run-button" 
              onClick={() => startAnalysis('technical')}
              disabled={loading.technical}
            >
              {loading.technical ? '분석 중...' : '분석 시작'}
            </button>
            <div className="output-area">{outputs.technical}</div>
          </div>
        </div>
      </div>
    </div>
  );
} 
