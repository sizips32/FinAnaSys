import Head from 'next/head';
import { useState, useEffect } from 'react';

export default function Home() {
  const [loading, setLoading] = useState({
    machine: false,
    dividend: false,
    portfolio: false,
    quantum: false,
    volatility: false,
    technical: false,
    wordcloud: false
  });

  // 전체 화면 모드로 새 탭을 여는 함수
  const openInFullscreen = (url) => {
    const newWindow = window.open(url, '_blank');
    if (newWindow) {
      newWindow.addEventListener('load', () => {
        try {
          if (newWindow.document.documentElement.requestFullscreen) {
            newWindow.document.documentElement.requestFullscreen();
          } else if (newWindow.document.documentElement.webkitRequestFullscreen) {
            newWindow.document.documentElement.webkitRequestFullscreen();
          } else if (newWindow.document.documentElement.msRequestFullscreen) {
            newWindow.document.documentElement.msRequestFullscreen();
          }
        } catch (error) {
          console.error('Fullscreen error:', error);
        }
      });
    }
  };

  const startAnalysis = async (type) => {
    try {
      setLoading(prev => ({ ...prev, [type]: true }));
      const response = await fetch(`/api/analyze/${type}`, {
        method: 'POST'
      });
      const data = await response.json();
      
      if (data.success) {
        openInFullscreen(data.url);
      } else {
        console.error('Error:', data.error);
        alert(data.error);
      }
    } catch (error) {
      console.error('Error:', error);
      alert('분석 시작 중 오류가 발생했습니다.');
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
          max-width: 1400px;
          margin: 0 auto;
          padding: 2rem;
        }

        // 워드 클라우드 배너 스타일
        .featured-banner {
          background: linear-gradient(135deg, rgba(135, 206, 235, 0.15) 0%, rgba(135, 206, 235, 0.05) 100%);
          border-radius: 12px;
          margin: 0 1rem 2rem 1rem;
          padding: 1rem 1.5rem;
          display: flex;
          align-items: center;
          gap: 2rem;
          backdrop-filter: blur(8px);
          border: 1px solid rgba(135, 206, 235, 0.2);
          box-shadow: 0 4px 24px 0 rgba(79, 179, 232, 0.15);
        }

        .banner-title-group {
          display: flex;
          align-items: center;
          gap: 1.5rem;
          min-width: fit-content;
        }

        .banner-title {
          font-size: 1.4rem;
          font-weight: 700;
          margin: 0;
          white-space: nowrap;
          background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .banner-details {
          display: flex;
          align-items: center;
          gap: 2rem;
          flex: 1;
          margin: 0;
          overflow-x: auto;
          padding: 0.5rem 0;
        }

        .banner-detail-item {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          color: var(--text-secondary);
          font-size: 0.9rem;
          white-space: nowrap;
        }

        .banner-button {
          background: linear-gradient(135deg, #141E30 0%, #243B55 100%);
          color: white;
          border: none;
          padding: 0.6rem 1.2rem;
          border-radius: 8px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.3s ease;
          white-space: nowrap;
          min-width: fit-content;
          box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .banner-button:hover {
          background: linear-gradient(135deg, #243B55 0%, #141E30 100%);
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        }

        .banner-button:disabled {
          background: linear-gradient(135deg, #808080 0%, #666666 100%);
          cursor: not-allowed;
          transform: none;
        }

        .banner-output {
          background: rgba(0,0,0,0.1);
          padding: 0.8rem;
          border-radius: 8px;
          font-family: monospace;
          font-size: 0.9rem;
          color: var(--text-secondary);
          margin-top: 1rem;
          max-width: 600px;
        }

        .analysis-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1.5rem;
          padding: 0 1rem;
        }

        .analysis-card {
          background: var(--gradient-card);
          backdrop-filter: blur(16px);
          -webkit-backdrop-filter: blur(16px);
          border-radius: 15px;
          padding: 1.8rem;
          box-shadow: var(--card-shadow);
          border: 1px solid rgba(135, 206, 235, 0.3);
          transition: transform 0.3s ease, box-shadow 0.3s ease;
          height: 100%;
          display: flex;
          flex-direction: column;
        }

        // 워드 클라우드 카드 특별 스타일
        .featured-analysis .analysis-card {
          background: linear-gradient(135deg, rgba(135, 206, 235, 0.35) 0%, rgba(135, 206, 235, 0.25) 100%);
          border: 1px solid rgba(135, 206, 235, 0.4);
          box-shadow: 0 12px 40px 0 rgba(79, 179, 232, 0.4);
        }

        .analysis-card:hover {
          transform: translateY(-5px);
          box-shadow: 0 12px 40px 0 rgba(79, 179, 232, 0.4);
        }

        .analysis-card h2 {
          margin-bottom: 1rem;
          font-size: 1.4rem;
          font-weight: 700;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          background: linear-gradient(135deg, #24243e 0%, #0f0c29 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          text-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
          background: linear-gradient(135deg, #000428 0%, #004e92 100%);
          color: white;
          font-weight: 500;
          font-size: 1rem;
          cursor: pointer;
          transition: all 0.3s ease;
          box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .run-button:hover {
          background: linear-gradient(135deg, #004e92 0%, #000428 100%);
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        }

        .run-button:disabled {
          background: linear-gradient(135deg, #808080 0%, #666666 100%);
          cursor: not-allowed;
          transform: none;
          box-shadow: none;
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

        @media (max-width: 1024px) {
          .featured-banner {
            flex-wrap: wrap;
            padding: 1rem;
            gap: 1rem;
          }

          .banner-details {
            order: 3;
            justify-content: flex-start;
            width: 100%;
          }
        }

        @media (max-width: 768px) {
          .analysis-grid {
            grid-template-columns: 1fr;
          }
          .container {
            padding: 1rem;
          }
          .featured-banner {
            margin: 0 0.5rem 1.5rem 0.5rem;
          }
          
          .banner-title-group {
            width: 100%;
            justify-content: space-between;
          }
        }
      `}</style>

      <header className="header">
        <h1>Financial Analysis Dashboard</h1>
        <p>고급 금융 분석 도구를 활용한 데이터 기반 의사결정 시스템</p>
      </header>

      <div className="container">
        <div className="featured-banner">
          <div className="banner-title-group">
            <h2 className="banner-title">워드 클라우드 분석</h2>
            <button 
              className="banner-button" 
              onClick={async () => {
                try {
                  setLoading(prev => ({ ...prev, wordcloud: true }));
                  const response = await fetch('/api/analyze/wordcloud', {
                    method: 'POST'
                  });
                  const data = await response.json();
                  if (data.success) {
                    window.open(data.url, '_blank');
                  } else {
                    console.error('Error:', data.error);
                    alert(data.error);
                  }
                } catch (error) {
                  console.error('Error:', error);
                  alert('분석 시작 중 오류가 발생했습니다.');
                } finally {
                  setLoading(prev => ({ ...prev, wordcloud: false }));
                }
              }}
              disabled={loading.wordcloud}
            >
              {loading.wordcloud ? '분석 중...' : '분석 시작'}
            </button>
          </div>
          <div className="banner-details">
            <div className="banner-detail-item">
              <span>•</span>
              <span>텍스트 파일 분석</span>
            </div>
            <div className="banner-detail-item">
              <span>•</span>
              <span>주요 키워드 추출</span>
            </div>
            <div className="banner-detail-item">
              <span>•</span>
              <span>단어 빈도 시각화</span>
            </div>
          </div>
        </div>

        <div className="analysis-grid">
          <div className="analysis-card">
            <h2>머신러닝 분석</h2>
            <div className="analysis-description">
              머신러닝 기반의 고급 분석을 제공합니다.
              <div className="analysis-details">
                • 딥러닝 모델 기반 예측<br />
                • 시계열 패턴 분석<br />
                • AI 기반 투자 전략
              </div>
            </div>
            <button 
              className="run-button" 
              onClick={() => startAnalysis('machine')}
              disabled={loading.machine}
            >
              {loading.machine ? '분석 중...' : '분석 시작'}
            </button>
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
          </div>
        </div>
      </div>
    </div>
  );
} 
