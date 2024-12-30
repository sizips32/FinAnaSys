import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
from arch import arch_model
import matplotlib.font_manager as fm
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# 한글 폰트 설정
font_path = './NanumGothic.ttf'  # 시스템에 설치된 한글 폰트 경로
font_prop = fm.FontProperties(fname=font_path, size=12)
plt.rc('font', family=font_prop.get_name())  # 기본 폰트 설정
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 표시

# Title
st.title('주가 변동성 분석 시스템')
st.markdown("""
### 분석 개요
이 시스템은 주식의 변동성과 리스크를 다각도로 분석합니다.

#### 주요 지표:
1. **베타 계수 분석**
   - 시장 대비 변동성 측정
   - 상관관계 분석

2. **변동성 지표**
   - 역사적 변동성 계산
   - 내재 변동성 추정

3. **리스크 평가**
   - VaR (Value at Risk) 계산
   - 최대 손실 예상치 제공
""")

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

# 사용자 입력을 통한 변동성 임계값 설정
threshold = st.sidebar.number_input('변동성 임계값 입력', value=0.025, format="%.4f")  # 기본값 0.025

# 사용자 입력을 통한 무위험 수익률 설정
risk_free_rate = st.sidebar.number_input('무위험 수익률 입력 (예: 0.025)', value=0.025, format="%.4f")  # 기본값 0.025

# 실행 버튼
execute = st.sidebar.button("시뮬레이션 실행")

initial_cash = 100000  # 초기 현금 정의

def fetch_data(ticker, start_date, end_date):
    """주식 데이터를 다운로드하고 수익률을 계산합니다."""
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Adj Close'].pct_change().dropna()
    return data

def calculate_volatility(data):
    """GARCH 모델을 사용하여 변동성을 계산합니다."""
    model = arch_model(data['Returns'].dropna(), vol='Garch', p=1, q=1)
    results = model.fit(disp='off')
    data['Volatility'] = np.sqrt(results.conditional_volatility)
    return data

def bayesian_analysis(data, prior_mean=0.02, prior_variance=0.01):
    """베이지안 변동성 분석을 수행하고 사후 평균과 분산을 계산합니다."""
    likelihood_variance = np.var(data['Returns'].dropna())
    posterior_mean = (prior_mean / prior_variance + np.mean(data['Returns']) / likelihood_variance) / (1 / prior_variance + 1 / likelihood_variance)
    posterior_variance = 1 / (1 / prior_variance + 1 / likelihood_variance)
    return posterior_mean, posterior_variance

def generate_investment_signal(posterior_mean, threshold=0.025):
    """투자 신호를 생성합니다."""
    if posterior_mean > threshold:
        return "💡 **투자 신호: 사후 평균이 임계값을 초과했습니다. 투자 고려하세요!**"
    else:
        return "🔍 **관망 신호: 사후 평균이 임계값 이하입니다. 신중하게 접근하세요.**"

def calculate_var(data, confidence_level=0.95):
    """VaR (Value at Risk)를 계산합니다."""
    var_value = np.percentile(data['Returns'].dropna(), (1 - confidence_level) * 100)
    return var_value

def simulate_hedging(data, hedge_ratio, initial_cash=100000):
    """동적 헤지 전략을 시뮬레이션합니다."""
    portfolio_value = initial_cash  # 초기 포트폴리오 가치 설정
    cash = initial_cash * (1 - hedge_ratio / 100)  # 현금 설정
    hedge = initial_cash * (hedge_ratio / 100)  # 헤지 자산 설정
    portfolio = []

    for i in range(1, len(data)):
        if data['Volatility'].iloc[i] > volatility_threshold / 100:
            hedge *= 1.05
            cash -= (hedge * 0.05)
        else:
            hedge *= 0.95
            cash += (hedge * 0.05)

        portfolio_value = cash + hedge
        portfolio.append(portfolio_value)

    data['Portfolio'] = [initial_cash] + portfolio
    return data

if execute:
    # 사용자 입력을 통한 사전 확률 설정
    prior_mean = st.sidebar.number_input('사전 기대 수익률 입력', value=0.02, format="%.4f")  # 초기 기대 수익률
    prior_variance = st.sidebar.number_input('사전 불확실성 입력', value=0.01, format="%.4f")  # 초기 불확실성

    # 데이터 수집
    data = fetch_data(ticker, start_date, end_date)

    # 수익률 데이터가 비어 있는지 확인
    if data['Returns'].isnull().all():
        st.error("수익률 데이터가 없습니다. 다른 주식 티커를 입력해 주세요.")
    else:
        # 변동성 계산
        st.header('변동성 분석')
        data = calculate_volatility(data)

        # 베이지안 변동성 분석
        st.header('베이지안 변동성 분석')
        posterior_mean, posterior_variance = bayesian_analysis(data, prior_mean, prior_variance)

        st.write(f"사후 평균 (Posterior Mean): {posterior_mean:.4f}")
        st.write(f"사후 분산 (Posterior Variance): {posterior_variance:.4f}")

        # 투자 신호 생성
        investment_signal = generate_investment_signal(posterior_mean, risk_free_rate)  # 사용자 입력으로 무위험 수익률 전달
        st.write(investment_signal)

        # 투자 설명 추가
        if posterior_mean > risk_free_rate:
            st.write("### 투자 설명")
            st.write("사후 평균이 무위험 수익률을 초과했습니다. 이는 해당 주식이 앞으로 긍정적인 수익을 낼 가능성이 높다는 것을 의미합니다. "
                      "따라서, 이 주식에 투자하는 것이 좋습니다.")
        else:
            st.write("### 투자 설명")
            st.write("사후 평균이 무위험 수익률 이하입니다. 이는 해당 주식의 수익률이 기대에 미치지 못할 가능성이 높다는 것을 의미합니다. "
                      "따라서, 신중하게 접근하고 다른 투자 기회를 고려하는 것이 좋습니다.")

        # VaR 계산
        var_value = calculate_var(data)
        st.write(f"VaR (신뢰 수준 95%): {var_value:.4f}")

        # 변동성 시각화
        st.subheader('변동성 시각화')
        fig, ax = plt.subplots()
        ax.plot(data.index, data['Volatility'], label='Volatility')
        ax.axhline(volatility_threshold / 100, color='r', linestyle='--', label='Threshold')
        ax.legend()
        st.pyplot(fig)

        # 동적 헤지 전략 시뮬레이션
        st.header('헤지 전략 시뮬레이션')
        data = simulate_hedging(data, hedge_ratio, initial_cash)

        # 포트폴리오 성과 시각화
        st.subheader('포트폴리오 가치 변화')
        fig2, ax2 = plt.subplots()
        ax2.plot(data.index, data['Portfolio'], label='포트폴리오 가치', color='blue')
        ax2.set_title('포트폴리오 가치 변화', fontproperties=font_prop)
        ax2.set_xlabel('날짜', fontproperties=font_prop)
        ax2.set_ylabel('포트폴리오 가치 ($)', fontproperties=font_prop)
        ax2.legend(prop=font_prop)
        st.pyplot(fig2)

        # 누적 수익률 계산
        data['Cumulative Returns'] = (data['Returns'] + 1).cumprod() - 1

        # 누적 수익률 시각화
        st.subheader('누적 수익률')
        fig3, ax3 = plt.subplots()
        ax3.plot(data.index, data['Cumulative Returns'], label='누적 수익률', color='green')
        ax3.set_title('누적 수익률', fontproperties=font_prop)
        ax3.set_xlabel('날짜', fontproperties=font_prop)
        ax3.set_ylabel('누적 수익률 (%)', fontproperties=font_prop)
        ax3.legend(prop=font_prop)
        st.pyplot(fig3)

        # 성과 요약
        st.subheader('성과 요약')
        st.write(f"최종 포트폴리오 가치: ${data['Portfolio'].iloc[-1]:,.2f}")
        st.write(f"수익률: {((data['Portfolio'].iloc[-1] / initial_cash - 1) * 100):.2f}%")

        # 에르고딕 가설 분석 추가
        st.header('에르고딕 가설 분석')

        # 1. 누적 수익률 계산
        cumulative_returns = (data['Returns'] + 1).cumprod() - 1  # 누적 수익률 계산
        st.line_chart(cumulative_returns)  # 누적 수익률 시각화

        # 2. 시간 평균 (Time Average) 계산
        time_avg = np.mean(cumulative_returns)  # 누적 수익률의 평균
        st.write(f"시간 평균 (Time Average): {time_avg:.4f}")

        # 3. 집합 평균 (Ensemble Average) 계산
        ensemble_avg = data['Returns'].mean() * len(data)  # 수익률의 평균에 데이터 포인트 수를 곱함
        st.write(f"집합 평균 (Ensemble Average): {ensemble_avg:.4f}")

        # 4. 에르고딕 성질 검증
        difference = abs(time_avg - ensemble_avg)  # 시간 평균과 집합 평균의 차이 계산
        st.write(f"시간 평균과 집합 평균의 차이: {difference:.4f}")

        # 5. 에르고딕 성질의 성립 여부 판단
        if difference < 0.01:  # 차이가 0.01 이하일 경우
            st.write("✅ 에르고딕 성질이 성립합니다. 장기적으로 전략이 안정적일 가능성이 높습니다.")
        else:
            st.write("❌ 에르고딕 성질이 성립하지 않습니다. 전략의 기 안정성을 재검토해야 합니다.")

        # 전략 추천 섹션
        st.write("### 전략 추천")
        if data['Volatility'].iloc[-1] > volatility_threshold / 100:
            st.write("🔥 **변동성이 높은 상황입니다. 헤지 비중을 확대하고 단기 옵션을 고려하세요.**")
        else:
            st.write("📈 **변동성이 안정적입니다. 핵심 자산 비중을 유지하며 장기 성장 전략을 고려하세요.**")

        # 결과 분석 설명
        st.write("### 결과 분석")
        st.write("이 전략은 변동성 임계값을 기반으로 동적 헤지를 수행하여 시장 급변 상황에 대비합니다.")

        # 종합 결과 분석
        st.header('종합 결과 분석')
        final_portfolio_value = data['Portfolio'].iloc[-1] if 'Portfolio' in data.columns else initial_cash
        st.write(f"최종 포트폴리오 가치: ${final_portfolio_value:,.2f}")
        st.write(f"수익률: {((final_portfolio_value / initial_cash - 1) * 100):.2f}%")

        # 투자 추천
        st.subheader('투자 추천')

        # 1. 사후 평균과 임계값 비교
        if posterior_mean > risk_free_rate:  # 무위험 수익률과 비교
            st.write("🔍 **사후 평균이 무위험 수익률을 초과했습니다.**")
            
            # 2. 포트폴리오 가치 평가
            if final_portfolio_value > initial_cash:
                st.write("📈 **추천: 매수!**")
                st.write("변동성이 높고, 사후 평균이 무위험 수익률을 초과했습니다. "
                         "이는 해당 주식이 긍정적인 수익을 낼 가능성이 높다는 것을 의미합니다.")
            else:
                st.write("🔄 **추천: 홀딩!**")
                st.write("포트폴리오 가치가 초기 투자금보다 낮지만, 사후 평균이 무위험 수익률을 초과합니다. "
                         "따라서, 추가적인 관찰이 필요합니다.")

        # 3. 사후 평균과 임계값 비교 (하위 조건)
        else:
            st.write("🔍 **사후 평균이 무위험 수익률 이하입니다.**")
            
            # 4. 포트폴리오 가치 평가
            if final_portfolio_value < initial_cash:
                st.write("🔻 **추천: 매도!**")
                st.write("변동성이 낮고, 사후 평균이 무위험 수익률 이하입니다. "
                         "이는 해당 주식의 수익률이 기대에 미치지 못할 가능성이 높다는 것을 의미합니다.")
            else:
                st.write("🔄 **추천: 홀딩!**")
                st.write("포트폴리오 가치가 초기 투자금보다 높지만, 사후 평균이 무위험 수익률 이하입니다. "
                         "따라서, 신중하게 접근하고 다른 투자 기회를 고려하는 것이 좋습니다.")

# Footer
st.sidebar.markdown("---")
st.sidebar.text("Created by Sean J. Kim")
