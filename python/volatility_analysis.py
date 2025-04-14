import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd

st.set_page_config(page_title="변동성 분석", layout="wide")

st.title("주식 변동성 분석")

# 사이드바에 주식 심볼 입력
symbol = st.sidebar.text_input("주식 심볼을 입력하세요 (예: 005930.KS)", "005930.KS")
period = st.sidebar.selectbox("기간 선택", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
window = st.sidebar.slider("변동성 계산 기간 (일)", 5, 30, 20)

if symbol:
    try:
        # 주식 데이터 가져오기
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)

        # 일일 수익률 계산
        df['Returns'] = df['Close'].pct_change()
        
        # 변동성 계산 (표준편차)
        df['Volatility'] = df['Returns'].rolling(window=window).std() * np.sqrt(252) * 100
        
        # 변동성 차트
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index,
                               y=df['Volatility'],
                               name='연간화 변동성 (%)',
                               line=dict(color='red')))
        
        fig.update_layout(title=f"{symbol} 변동성 추이",
                         yaxis_title="연간화 변동성 (%)",
                         xaxis_title="날짜")
        
        st.plotly_chart(fig, use_container_width=True)

        # 주가와 변동성 관계
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index,
                                y=df['Close'],
                                name='주가',
                                yaxis='y'))
        fig2.add_trace(go.Scatter(x=df.index,
                                y=df['Volatility'],
                                name='변동성',
                                yaxis='y2'))
        
        fig2.update_layout(title="주가와 변동성 관계",
                          yaxis=dict(title="주가"),
                          yaxis2=dict(title="변동성 (%)",
                                    overlaying='y',
                                    side='right'),
                          xaxis_title="날짜")
        
        st.plotly_chart(fig2, use_container_width=True)

        # 변동성 통계
        st.subheader("변동성 통계")
        col1, col2, col3 = st.columns(3)
        
        current_vol = df['Volatility'].iloc[-1]
        avg_vol = df['Volatility'].mean()
        max_vol = df['Volatility'].max()
        
        col1.metric("현재 변동성", f"{current_vol:.2f}%")
        col2.metric("평균 변동성", f"{avg_vol:.2f}%")
        col3.metric("최대 변동성", f"{max_vol:.2f}%")

        # 변동성 분포
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=df['Volatility'],
                                  nbinsx=30,
                                  name='변동성 분포'))
        
        fig3.update_layout(title="변동성 분포",
                          xaxis_title="변동성 (%)",
                          yaxis_title="빈도")
        
        st.plotly_chart(fig3, use_container_width=True)

    except Exception as e:
        st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {str(e)}") 
