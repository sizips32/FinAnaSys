import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="배당 분석", layout="wide")

st.title("배당금 분석")

# 사이드바에 주식 심볼 입력
symbol = st.sidebar.text_input("주식 심볼을 입력하세요 (예: 005930.KS)", "005930.KS")

if symbol:
    try:
        # 주식 데이터 가져오기
        stock = yf.Ticker(symbol)
        
        # 기본 정보
        info = stock.info
        
        # 기업 정보 표시
        st.subheader("기업 정보")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("현재 주가", f"₩{info.get('currentPrice', 'N/A'):,.0f}")
        with col2:
            st.metric("배당 수익률", f"{info.get('dividendYield', 0)*100:.2f}%")
        with col3:
            st.metric("시가 배당률", f"{info.get('trailingAnnualDividendYield', 0)*100:.2f}%")

        # 배당금 히스토리
        dividends = stock.dividends
        if not dividends.empty:
            # 배당금 추이 차트
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dividends.index, 
                                   y=dividends.values,
                                   mode='lines+markers',
                                   name='배당금'))
            
            fig.update_layout(title="배당금 추이",
                            yaxis_title="배당금 (원)",
                            xaxis_title="날짜")
            
            st.plotly_chart(fig, use_container_width=True)

            # 연간 배당금 합계
            yearly_dividends = dividends.groupby(dividends.index.year).sum()
            
            # 연간 배당금 차트
            fig_yearly = go.Figure()
            fig_yearly.add_trace(go.Bar(x=yearly_dividends.index,
                                      y=yearly_dividends.values,
                                      name='연간 배당금'))
            
            fig_yearly.update_layout(title="연간 배당금",
                                   yaxis_title="배당금 (원)",
                                   xaxis_title="연도")
            
            st.plotly_chart(fig_yearly, use_container_width=True)

            # 배당금 통계
            st.subheader("배당금 통계")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("최근 배당금", f"₩{dividends.iloc[-1]:.2f}")
            with col2:
                st.metric("평균 배당금", f"₩{dividends.mean():.2f}")
            with col3:
                st.metric("최대 배당금", f"₩{dividends.max():.2f}")

        else:
            st.warning("배당금 데이터가 없습니다.")

    except Exception as e:
        st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {str(e)}") 
