import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="기술적 분석", layout="wide")

st.title("주식 기술적 분석")

# 사이드바에 주식 심볼 입력
symbol = st.sidebar.text_input("주식 심볼을 입력하세요 (예: 005930.KS)", "005930.KS")
period = st.sidebar.selectbox("기간 선택", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])

if symbol:
    try:
        # 주식 데이터 가져오기
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)

        # 캔들스틱 차트
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                           open=df['Open'],
                                           high=df['High'],
                                           low=df['Low'],
                                           close=df['Close'])])
        
        fig.update_layout(title=f"{symbol} 주가 차트",
                        yaxis_title="주가",
                        xaxis_title="날짜")

        st.plotly_chart(fig, use_container_width=True)

        # 기술적 지표 계산
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()

        # 이동평균선 차트
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df.index, y=df['Close'], name="주가"))
        fig_ma.add_trace(go.Scatter(x=df.index, y=df['MA5'], name="5일 이동평균"))
        fig_ma.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="20일 이동평균"))
        fig_ma.add_trace(go.Scatter(x=df.index, y=df['MA60'], name="60일 이동평균"))

        fig_ma.update_layout(title="이동평균선",
                           yaxis_title="주가",
                           xaxis_title="날짜")

        st.plotly_chart(fig_ma, use_container_width=True)

        # 거래량 차트
        fig_volume = go.Figure(data=[go.Bar(x=df.index, y=df['Volume'])])
        fig_volume.update_layout(title="거래량",
                               yaxis_title="거래량",
                               xaxis_title="날짜")

        st.plotly_chart(fig_volume, use_container_width=True)

    except Exception as e:
        st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {str(e)}") 
