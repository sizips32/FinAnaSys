import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="포트폴리오 분석", layout="wide")

st.title("포트폴리오 분석")

# 사이드바에서 포트폴리오 구성 입력
st.sidebar.header("포트폴리오 구성")

# 기본 포트폴리오 구성
default_portfolio = {
    "005930.KS": 30,  # 삼성전자
    "035420.KS": 20,  # NAVER
    "035720.KS": 20,  # Kakao
    "051910.KS": 15,  # LG화학
    "000660.KS": 15   # SK하이닉스
}

# 포트폴리오 입력
portfolio = {}
for stock, weight in default_portfolio.items():
    input_weight = st.sidebar.number_input(f"{stock} 비중 (%)", 
                                         min_value=0, 
                                         max_value=100, 
                                         value=weight)
    if input_weight > 0:
        portfolio[stock] = input_weight

# 총 비중이 100%가 되도록 조정
total_weight = sum(portfolio.values())
if total_weight != 100:
    st.warning(f"포트폴리오 비중의 합이 100%가 아닙니다. (현재: {total_weight}%)")
    portfolio = {k: (v/total_weight)*100 for k, v in portfolio.items()}

# 기간 선택
period = st.sidebar.selectbox("분석 기간", ["1y", "2y", "5y"])

if portfolio:
    try:
        # 포트폴리오 데이터 수집
        portfolio_data = pd.DataFrame()
        for symbol in portfolio.keys():
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)['Close']
            portfolio_data[symbol] = data

        # 수익률 계산
        returns = portfolio_data.pct_change()
        weighted_returns = returns.mul(np.array(list(portfolio.values()))/100, axis=1)
        portfolio_return = weighted_returns.sum(axis=1)

        # 누적 수익률 계산
        cumulative_return = (1 + portfolio_return).cumprod()

        # 포트폴리오 가치 변화 차트
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cumulative_return.index, 
                               y=cumulative_return, 
                               name="포트폴리오"))
        
        fig.update_layout(title="포트폴리오 가치 변화",
                         yaxis_title="누적 수익률",
                         xaxis_title="날짜")
        
        st.plotly_chart(fig, use_container_width=True)

        # 포트폴리오 구성 파이 차트
        fig_pie = px.pie(values=list(portfolio.values()),
                        names=list(portfolio.keys()),
                        title="포트폴리오 구성")
        
        st.plotly_chart(fig_pie, use_container_width=True)

        # 기초 통계량
        st.subheader("포트폴리오 통계")
        annual_return = portfolio_return.mean() * 252 * 100
        annual_volatility = portfolio_return.std() * np.sqrt(252) * 100
        sharpe_ratio = annual_return / annual_volatility

        col1, col2, col3 = st.columns(3)
        col1.metric("연간 기대 수익률", f"{annual_return:.2f}%")
        col2.metric("연간 변동성", f"{annual_volatility:.2f}%")
        col3.metric("샤프 비율", f"{sharpe_ratio:.2f}")

    except Exception as e:
        st.error(f"데이터를 분석하는 중 오류가 발생했습니다: {str(e)}") 
