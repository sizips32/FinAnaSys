import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
from arch import arch_model

# Title
st.title('변동성 기반 투자 전략 시뮬레이션')

# Sidebar Inputs
st.sidebar.header('설정 옵션')

# 설명 추가
st.sidebar.markdown("### 파라미터 설명")
st.sidebar.markdown("- **주식 티커**: 분석할 주식 또는 ETF 코드(SPY, QQQ 등).")
st.sidebar.markdown("- **시작/종료 날짜**: 분석 기간 설정.")
st.sidebar.markdown("- **헤지 비율**: 포트폴리오에서 변동성 헤지 자산의 비율.")
st.sidebar.markdown("- **변동성 임계값**: VIX 기반으로 추가 헤지를 실행할 변동성 기준치.")

# 입력 파라미터
ticker = st.sidebar.text_input('주식 티커 입력', value='SPY')
start_date = st.sidebar.date_input('시작 날짜', dt.date(2015, 1, 1))
end_date = st.sidebar.date_input('종료 날짜', dt.date.today())
hedge_ratio = st.sidebar.slider('헤지 비율 (%)', 5, 20, 10)
volatility_threshold = st.sidebar.slider('변동성 임계값 (VIX 기준)', 10, 50, 20)

# 실행 버튼
execute = st.sidebar.button("시뮬레이션 실행")

if execute:
    # 데이터 가져오기 및 수익률 계산
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Adj Close'].pct_change().dropna()

    # 변동성 계산 (GARCH 모델 사용)
    st.header('변동성 분석')
    model = arch_model(data['Returns'].dropna(), vol='Garch', p=1, q=1)
    results = model.fit(disp='off')
    data['Volatility'] = np.sqrt(results.conditional_volatility)

    # 변동성 시각화
    st.subheader('변동성 시각화')
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Volatility'], label='Volatility')
    ax.axhline(volatility_threshold / 100, color='r', linestyle='--', label='Threshold')
    ax.legend()
    st.pyplot(fig)

    # 동적 헤지 전략 시뮬레이션
    st.header('헤지 전략 시뮬레이션')
    initial_cash = 100000  # 초기 자본 설정
    portfolio_value = initial_cash  # 포트폴리오 초기 가치
    cash = initial_cash * (1 - hedge_ratio / 100)  # 현금 비중
    hedge = initial_cash * (hedge_ratio / 100)  # 헤지 자산 비중
    portfolio = []  # 포트폴리오 기록 저장 리스트

    for i in range(1, len(data)):
        # 변동성이 임계값 초과 시 헤지 강화
        if data['Volatility'].iloc[i] > volatility_threshold / 100:
            hedge *= 1.05
            cash -= (hedge * 0.05)
        # 변동성이 임계값 이하 시 헤지 축소
        else:
            hedge *= 0.95
            cash += (hedge * 0.05)

        # 포트폴리오 가치 업데이트
        portfolio_value = cash + hedge
        portfolio.append(portfolio_value)

    data['Portfolio'] = [initial_cash] + portfolio

    # 포트폴리오 성과 시각화
    st.subheader('포트폴리오 성과 시각화')
    fig2, ax2 = plt.subplots()
    ax2.plot(data.index, data['Portfolio'], label='Portfolio Value')
    ax2.legend()
    st.pyplot(fig2)

    # 성과 요약
    st.subheader('성과 요약')
    st.write(f"최종 포트폴리오 가치: ${data['Portfolio'].iloc[-1]:,.2f}")
    st.write(f"수익률: {((data['Portfolio'].iloc[-1] / initial_cash - 1) * 100):.2f}%")

    # 에르고딕 가설 분석 추가
    st.header('에르고딕 가설 분석')
    # 시간 평균 계산
    time_avg = np.mean(data['Returns'].cumsum())
    # 집합 평균 계산
    ensemble_avg = data['Returns'].mean() * len(data)

    st.write(f"시간 평균 (Time Average): {time_avg:.4f}")
    st.write(f"집합 평균 (Ensemble Average): {ensemble_avg:.4f}")
    if abs(time_avg - ensemble_avg) < 0.01:
        st.write("에르고딕 성질이 성립합니다. 장기적으로 전략이 안정적일 가능성이 높습니다.")
    else:
        st.write("에르고딕 성질이 성립하지 않습니다. 전략의 장기 안정성을 재검토해야 합니다.")

    # 전략 추천 섹션
    st.write("### 전략 추천")
    if data['Volatility'].iloc[-1] > volatility_threshold / 100:
        st.write("📊 **변동성이 높은 상황입니다. 헤지 비중을 확대하고 단기 옵션을 고려하세요.**")
    else:
        st.write("📈 **변동성이 안정적입니다. 핵심 자산 비중을 유지하며 장기 성장 전략을 고려하세요.**")

    # 결과 분석 설명
    st.write("### 결과 분석")
    st.write("이 전략은 변동성 임계값을 기반으로 동적 헤지를 수행하여 시장 급변 상황에 대비합니다.")

# Footer
st.sidebar.markdown("---")
st.sidebar.text("Created by Sean J. Kim")
