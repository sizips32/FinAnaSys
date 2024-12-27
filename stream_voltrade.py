import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import datetime as dt
from arch import arch_model
from scipy import stats
from sklearn.mixture import GaussianMixture

# 한글 폰트 설정 (필요시 주석 처리 가능)
# font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # 시스템에 맞는 한글 폰트 경로
# font_prop = fm.FontProperties(fname=font_path, size=12)
# plt.rc('font', family=font_prop.get_name())

# Title
st.title('Volatility-Based Investment Strategy Simulation')

# Sidebar Inputs
st.sidebar.header('Settings')

# 설명 추가
st.sidebar.markdown("### Parameter Descriptions")
st.sidebar.markdown("- **Stock Ticker**: The code of the stock or ETF to analyze (SPY, QQQ, etc).")
st.sidebar.markdown("- **Start/End Date**: Set the analysis period.")
st.sidebar.markdown("- **Hedge Ratio**: The proportion of the portfolio invested in the hedge asset.")
st.sidebar.markdown("- **Volatility Threshold**: The volatility threshold based on VIX to trigger additional hedge.")

# 입력 파라미터
ticker = st.sidebar.text_input('Stock Ticker Input', value='SPY')
start_date = st.sidebar.date_input('Start Date', dt.date(2015, 1, 1))
end_date = st.sidebar.date_input('End Date', dt.date.today())
hedge_ratio = st.sidebar.slider('Hedge Ratio (%)', 5, 20, 10)
volatility_threshold = st.sidebar.slider('Volatility Threshold (VIX Basis)', 10, 50, 20)
initial_cash = st.sidebar.number_input('Initial Cash Input', value=10000)

# 실행 버튼
execute = st.sidebar.button("Run Simulation")

if execute:
    # 데이터 가져오기 및 수익률 계산
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Adj Close'].pct_change().dropna()

    # NaN 값 제거
    data = data.dropna(subset=['Returns'])

    # 변동성 계산 (수익률의 20일 이동 평균 표준편차)
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data = data.dropna(subset=['Volatility'])

    # 변동성 그래프 시각화
    st.subheader('Volatility Graph')
    fig_volatility, ax_volatility = plt.subplots(figsize=(10, 6))
    ax_volatility.plot(data['Volatility'], label='20-Day Moving Average Volatility', color='orange')
    ax_volatility.axhline(y=volatility_threshold/100, color='red', linestyle='--', label='Volatility Threshold')
    ax_volatility.set_title('Volatility Graph')
    ax_volatility.set_xlabel('Date')
    ax_volatility.set_ylabel('Volatility')
    ax_volatility.legend()
    st.pyplot(fig_volatility)

    # 일간 누적 수익률 계산
    data['Cumulative Returns'] = (1 + data['Returns']).cumprod() - 1

    # 일간 누적 수익률 그래프 시각화
    st.subheader('Cumulative Returns Graph')
    fig_cumulative_returns, ax_cumulative_returns = plt.subplots(figsize=(10, 6))
    ax_cumulative_returns.plot(data['Cumulative Returns'], label='Cumulative Returns', color='blue')
    ax_cumulative_returns.set_title('Cumulative Returns Graph')
    ax_cumulative_returns.set_xlabel('Date')
    ax_cumulative_returns.set_ylabel('Cumulative Returns')
    ax_cumulative_returns.legend()
    st.pyplot(fig_cumulative_returns)

    # 베이지안 변동성 추정
    def estimate_volatility_distribution(returns, n_components=2):
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(returns.reshape(-1, 1))
        return gmm
    
    # 몬테카를로 시뮬레이션
    def monte_carlo_simulation(returns, vol_model, n_sims=1000, horizon=30):
        simulations = []
        for _ in range(n_sims):
            sample = np.random.choice(returns, size=horizon)
            vol_pred = vol_model.predict(sample.reshape(-1, 1))
            sim_returns = sample * np.sqrt(vol_pred)
            simulations.append(np.cumprod(1 + sim_returns))
        return np.array(simulations)
    
    # 변동성 분포 추정
    if len(data['Returns'].dropna()) > 1:
        vol_dist = estimate_volatility_distribution(data['Returns'].values)
    else:
        st.error("Not enough data to estimate volatility distribution.")
    
    # 신뢰구간 계산
    def calculate_confidence_intervals(simulations, confidence_levels=[0.05, 0.95]):
        return np.percentile(simulations, [level * 100 for level in confidence_levels], axis=0)
    
    # 시뮬레이션 실행
    mc_sims = monte_carlo_simulation(data['Returns'].values, vol_dist)
    confidence_intervals = calculate_confidence_intervals(mc_sims)
    
    # 베이지안 포트폴리오 최적화
    def optimize_portfolio_weights(returns, volatility, confidence_level=0.95):
        posterior_vol = stats.norm.ppf(confidence_level) * volatility
        optimal_hedge = np.minimum(hedge_ratio/100 * (posterior_vol/(volatility_threshold/100)), 0.4)
        return optimal_hedge
    
    # 동적 헤지 전략 시뮬레이션 업데이트
    portfolio_values = []
    current_hedge_ratio = hedge_ratio/100
    
    for i in range(1, len(data)):
        # 베이지안 최적화된 헤지 비율 계산
        optimal_hedge = optimize_portfolio_weights(
            data['Returns'].iloc[i],
            data['Volatility'].iloc[i]
        )
        
        # 포트폴리오 재조정
        if data['Volatility'].iloc[i] > volatility_threshold/100:
            current_hedge_ratio = min(current_hedge_ratio * 1.05, optimal_hedge)
        else:
            current_hedge_ratio = max(current_hedge_ratio * 0.95, hedge_ratio/100)
            
        portfolio_value = initial_cash * (1 + data['Returns'].iloc[i] * (1 - current_hedge_ratio))
        portfolio_values.append(portfolio_value)
    
    # 결과 시각화
    st.subheader('Portfolio Simulation Results')
    fig, ax = plt.subplots(figsize=(12, 8))  # 그래프 크기 확대
    ax.plot(portfolio_values, label='Portfolio Value')
    ax.fill_between(range(len(confidence_intervals[0])),
                    confidence_intervals[0],
                    confidence_intervals[1],
                    alpha=0.2,
                    label='95% Confidence Interval')
    ax.legend()
    st.pyplot(fig)
    
    # 전략 성과 분석
    st.subheader('Bayesian Strategy Analysis')
    sharpe_ratio = np.mean(np.diff(portfolio_values)) / np.std(np.diff(portfolio_values)) * np.sqrt(252)
    var_95 = np.percentile(np.diff(portfolio_values), 5)
    
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    st.write(f"Value at Risk (95%): ${var_95:,.2f}")
    
    # 베이지안 전략 분석 결과 설명
    st.write("""
    **Sharpe Ratio** indicates the risk-adjusted return of the portfolio. 
    A higher value means that the portfolio has achieved higher returns for the level of risk taken. 
    Generally, a Sharpe Ratio above 1 is considered good performance.

    **Value at Risk (VaR)** represents the maximum amount of loss that the portfolio could incur over a specified period. 
    If the 95% VaR is $X, it means that there is a 95% probability that the loss will not exceed $X during that period.
    """)

    # 에르고딕 가설 적용
    expected_return = np.mean(data['Returns']) * 252  # 연간 기대 수익률
    st.write(f"Expected Annual Return: {expected_return:.2%}")

    # 투자 전략 제안
    st.subheader('Investment Strategy Recommendation')
    current_volatility = data['Volatility'].iloc[-1]
    current_return = data['Cumulative Returns'].iloc[-1]

    # 베이지안 전략 분석 결과에 따른 포트폴리오 비중 및 매수/매도/홀드 추천
    if sharpe_ratio > 1 and current_volatility <= volatility_threshold / 100:
        recommendation = "The portfolio is performing well. Consider increasing your position in this stock."
        portfolio_weight = "20-30%"
    elif sharpe_ratio > 1 and current_volatility > volatility_threshold / 100:
        recommendation = "The portfolio is performing well, but volatility is high. Consider maintaining your position."
        portfolio_weight = "15-20%"
    elif sharpe_ratio <= 1 and current_volatility <= volatility_threshold / 100:
        recommendation = "The portfolio is underperforming. Consider reducing your position in this stock."
        portfolio_weight = "10-15%"
    else:
        recommendation = "The portfolio is underperforming and volatility is high. Consider selling this stock."
        portfolio_weight = "5-10%"

    st.write(f"Current Volatility: {current_volatility:.2%}")
    st.write(f"Current Cumulative Return: {current_return:.2%}")
    st.write(f"Recommended Portfolio Weight: {portfolio_weight}")
    st.write(recommendation)

# Footer
st.sidebar.markdown("---")
st.sidebar.text("Created by Sean J. Kim")
