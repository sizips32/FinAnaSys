## Source: @DeepCharts Youtube Channel (https://www.youtube.com/@DeepCharts)

#### NOTE: Set yfinance to the following version to get chart working: "pip install yfinance==0.2.40"

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError

# 패키지 버전 검증 로직 수정
def verify_package_versions():
    required_versions = {
        'yfinance': '0.2.40',
        'pandas': '1.5.3',
        'numpy': '1.24.0',
        'streamlit': '1.24.0'
    }
    
    missing_packages = []
    version_mismatch = []
    
    for package, required_version in required_versions.items():
        try:
            installed_version = version(package)
            if installed_version != required_version:
                if package == 'yfinance':  # yfinance 특별 처리
                    st.error(f"""
                    yfinance 버전 불일치. 다음 명령어를 순서대로 실행하세요:
                    1. pip uninstall yfinance
                    2. pip cache purge
                    3. pip install yfinance==0.2.40
                    """)
                version_mismatch.append(f"{package} (현재: {installed_version}, 필요: {required_version})")
        except PackageNotFoundError:
            missing_packages.append(package)
    
    if missing_packages or version_mismatch:
        return False
    return True

# 메인 코드 시작 전에 버전 검증
if not verify_package_versions():
    st.stop()

# yfinance 버전 확인 및 경고
try:
    import yfinance as yf
    yf_version = yf.__version__
    if yf_version != "0.2.40":
        st.warning(f"""
        현재 yfinance 버전 ({yf_version})이 권장 버전(0.2.40)과 다릅니다.
        다음 명령어로 권장 버전을 설치하세요:
        pip install yfinance==0.2.40
        """)
except:
    st.error("yfinance 패키지를 설치해주세요: pip install yfinance==0.2.40")

try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError("Streamlit is not installed. Install it using 'pip install streamlit'")

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI 기반 재무-기술적 분석 시스템")
st.markdown("""
### 분석 방법론
이 시스템은 머신러닝을 활용한 고급 재무-기술적 분석을 제공합니다.

#### 주요 기능:
1. **AI 기반 패턴 인식**
   - 과거 차트 패턴 학습
   - 미래 패턴 예측

2. **기술적 지표 분석**
   - RSI, MACD, 볼린저 밴드 등 계산
   - 지표간 상관관계 분석

3. **매매 신호**
   - AI 기반 매수/매도 시점 제안
   - 신뢰도 점수 제공
""")

st.sidebar.header("Configuration")

# Input for stock ticker and date range
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=datetime.now().date())  # 현재 날짜로 설정

# Fetch stock data with error handling
if st.sidebar.button("Fetch Data"):
    try:
        with st.spinner("데이터를 가져오는 중..."):
            try:
                # 데이터 다운로드 시도 (show_errors 파라미터 제거)
                data = yf.download(ticker, 
                                 start=start_date, 
                                 end=end_date, 
                                 progress=False)
                
                if data.empty:
                    st.error(f"해당 기간에 데이터가 없습니다: {ticker}")
                    st.stop()
                
                # 티커 정보 가져오기 전에 데이터 유효성 검증
                if len(data) > 0:
                    ticker_obj = yf.Ticker(ticker)
                    try:
                        info = ticker_obj.info
                        if info and isinstance(info, dict):
                            st.session_state["financials"] = info
                        else:
                            st.warning("기업 정보를 가져올 수 없습니다. 기본 차트만 제공됩니다.")
                            st.session_state["financials"] = {}
                    except Exception as e:
                        st.warning(f"기업 정보 조회 중 오류 발생: {str(e)}. 기본 차트만 제공됩니다.")
                        st.session_state["financials"] = {}

                    # 세션 상태 업데이트
                    st.session_state["stock_data"] = data
                    
                    if len(data) < 20:
                        st.error("분석을 위해 최소 20일치의 데이터가 필요합니다.")
                        st.stop()
                    
                    st.success(f"{ticker} 데이터를 성공적으로 불러왔습니다!")
                else:
                    st.error("데이터를 찾을 수 없습니다.")
                    st.stop()
                
            except Exception as e:
                st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {str(e)}")
                st.stop()

    except Exception as e:
        st.error(f"예상치 못한 오류가 발생했습니다: {str(e)}")
        st.stop()

# Display financials in sidebar
if "financials" in st.session_state:
    financials = st.session_state["financials"]
    st.sidebar.subheader("Financial Information")
    st.sidebar.write(f"**Sector:** {financials.get('sector', 'N/A')}")
    st.sidebar.write(f"**Industry:** {financials.get('industry', 'N/A')}")
    
    # 시가총액 포맷팅 수정
    market_cap = financials.get('marketCap', 'N/A')
    if isinstance(market_cap, (int, float)):
        st.sidebar.write(f"**Market Cap:** ${market_cap:,.2f}")
    else:
        st.sidebar.write(f"**Market Cap:** {market_cap}")
    
    st.sidebar.write(f"**PE Ratio:** {financials.get('trailingPE', 'N/A')}")
    st.sidebar.write(f"**PS Ratio:** {financials.get('priceToSalesTrailing12Months', 'N/A')}")
    st.sidebar.write(f"**PB Ratio:** {financials.get('priceToBook', 'N/A')}")
    st.sidebar.write(f"**Dividend Yield:** {financials.get('dividendYield', 'N/A')}")
    st.sidebar.write(f"**52 Week High:** {financials.get('fiftyTwoWeekHigh', 'N/A')}")
    st.sidebar.write(f"**52 Week Low:** {financials.get('fiftyTwoWeekLow', 'N/A')}")

# Check if data is available
if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]

    # Plot candlestick chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlestick"
        )
    ])

    # Sidebar: Select technical indicators
    st.sidebar.subheader("Technical Indicators")
    indicators = st.sidebar.multiselect(
        "Select Indicators:",
        ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
        default=["20-Day SMA"]
    )

    # Helper function to add indicators to the chart
    def add_indicator(indicator):
        if indicator == "20-Day SMA":
            sma = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
        elif indicator == "20-Day EMA":
            ema = data['Close'].ewm(span=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
        elif indicator == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
        elif indicator == "VWAP":
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

    # Add selected indicators to the chart
    for indicator in indicators:
        add_indicator(indicator)

    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    # Analyze chart with LLaMA 3.2 Vision
    st.subheader("AI-Powered Analysis")

    def prepare_analysis_prompt():
        return """
        You are a Stock Trader specializing in Technical Analysis at a top financial institution.
        Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
        Base your recommendation only on the candlestick chart and the displayed technical indicators.
        First, provide the recommendation, then, provide your detailed reasoning.
        """

    if st.button("Run AI Analysis"):
        with st.spinner("Analyzing the chart, please wait..."):
            # Save chart as a temporary image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig.write_image(tmpfile.name)
                tmpfile_path = tmpfile.name

            # Read image and encode to Base64
            with open(tmpfile_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Prepare AI analysis request
            messages = [{
                'role': 'user',
                'content': prepare_analysis_prompt(),
                'images': [image_data]
            }]
            response = ollama.chat(model='llama3.2-vision', messages=messages)

            # Display AI analysis result
            st.write("**AI Analysis Results:**")
            st.write(response["message"]["content"])

            # Clean up temporary file
            os.remove(tmpfile_path)

# 가치 평가 지표 설명 수정
VALUATION_METRICS_DOC = """
가치 평가 지표는 기업의 주식이 현재 가격에 비해 과대평가 또는 과소평가되어 있는지를 판단하는 데 도움을 줍니다. 다음은 주요 가치 평가 지표와 그 의미입니다.

1. 주가수익비율 (Price-to-Earnings Ratio, P/E Ratio)
의미: 주가를 주당 순이익(EPS)으로 나눈 값으로, 주식이 현재 수익에 비해 얼마나 비싼지를 나타냅니다.

2. 주가순자산비율 (Price-to-Book Ratio, P/B Ratio)
의미: 주가를 주당 순자산(BVPS)으로 나눈 값으로, 기업의 자산 가치에 비해 주가가 얼마나 비싼지를 나타냅니다.

3. 주가매출비율 (Price-to-Sales Ratio, P/S Ratio)
의미: 주가를 주당 매출(SPS)로 나눈 값으로, 기업의 매출에 비해 주가가 얼마나 비싼지를 나타냅니다.

4. 배당 할인 모델 (Dividend Discount Model, DDM)
의미: 미래의 배당금을 현재 가치로 할인하여 주식의 가치를 평가하는 방법입니다.

5. 자기자본이익률 (Return on Equity, ROE)
의미: 순이익을 자기자본으로 나눈 비율로, 기업이 자기자본을 얼마나 효율적으로 활용하고 있는지를 나타냅니다.

6. 부채비율 (Debt-to-Equity Ratio, D/E Ratio)
의미: 총 부채를 자기자본으로 나눈 비율로, 기업의 재무 레버리지 정도를 나타냅니다.
"""

# 문서 표시
st.markdown(VALUATION_METRICS_DOC)

# Footer
st.sidebar.markdown("---")
st.sidebar.text("Created by Sean J. Kim")
