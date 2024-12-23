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

try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError("Streamlit is not installed. Install it using 'pip install streamlit'")

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# Input for stock ticker and date range
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-21"))

# Fetch stock data
if st.sidebar.button("Fetch Data"):
    st.session_state["stock_data"] = yf.download(ticker, start=start_date, end=end_date)
    ticker_info = yf.Ticker(ticker)
    st.session_state["financials"] = ticker_info.info
    st.success("Stock data loaded successfully!")

# Display financials in sidebar
if "financials" in st.session_state:
    financials = st.session_state["financials"]
    st.sidebar.subheader("Financial Information")
    st.sidebar.write(f"**Sector:** {financials.get('sector', 'N/A')}")
    st.sidebar.write(f"**Industry:** {financials.get('industry', 'N/A')}")
    st.sidebar.write(f"**Market Cap:** {financials.get('marketCap', 'N/A'):,}")
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
                'content': """
                    You are a Stock Trader specializing in Technical Analysis at a top financial institution.
                    Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
                    Base your recommendation only on the candlestick chart and the displayed technical indicators.
                    First, provide the recommendation, then, provide your detailed reasoning.
                """,
                'images': [image_data]
            }]
            response = ollama.chat(model='llama3.2-vision', messages=messages)

            # Display AI analysis result
            st.write("**AI Analysis Results:**")
            st.write(response["message"]["content"])

            # Clean up temporary file
            os.remove(tmpfile_path)

""" 
가치 평가 지표는 기업의 주식이 현재 가격에 비해 과대평가 또는 과소평가되어 있는지를 판단하는 데 도움을 줍니다. 다음은 주요 가치 평가 지표와 그 의미입니다.

1. 주가수익비율 (Price-to-Earnings Ratio, P/E Ratio)
의미: 주가를 주당 순이익(EPS)으로 나눈 값으로, 주식이 현재 수익에 비해 얼마나 비싼지를 나타냅니다. 일반적으로 P/E 비율이 낮을수록 주식이 저평가되어 있다고 볼 수 있습니다.
투자 적용: P/E 비율이 업종 평균보다 낮다면, 해당 주식이 저평가되어 있을 가능성이 있습니다. 반대로, P/E 비율이 높다면 과대평가된 것으로 볼 수 있습니다.

2. 주가순자산비율 (Price-to-Book Ratio, P/B Ratio)
의미: 주가를 주당 순자산(BVPS)으로 나눈 값으로, 기업의 자산 가치에 비해 주가가 얼마나 비싼지를 나타냅니다. P/B 비율이 1보다 낮으면 자산 가치에 비해 주가가 저평가되어 있다고 볼 수 있습니다.
투자 적용: P/B 비율이 낮은 기업은 자산 기반의 가치가 높아, 투자자에게 매력적일 수 있습니다. 반면, P/B 비율이 높으면 자산 가치에 비해 주가가 비쌀 수 있습니다.

3. 주가매출비율 (Price-to-Sales Ratio, P/S Ratio)
의미: 주가를 주당 매출(SPS)로 나눈 값으로, 기업의 매출에 비해 주가가 얼마나 비싼지를 나타냅니다. P/S 비율이 낮을수록 매출 대비 주가가 저평가되어 있다고 볼 수 있습니다.
투자 적용: P/S 비율이 낮은 기업은 매출 기반의 가치가 높아, 성장 가능성이 있는 기업으로 평가될 수 있습니다.

4. 배당 할인 모델 (Dividend Discount Model, DDM)
의미: 미래의 배당금을 현재 가치로 할인하여 주식의 가치를 평가하는 방법입니다. 이 모델은 배당금이 안정적으로 지급되는 기업에 적합합니다.
투자 적용: DDM을 통해 계산된 주가가 현재 주가보다 높다면, 해당 주식이 저평가되어 있다고 판단할 수 있습니다.

5. 자기자본이익률 (Return on Equity, ROE)
의미: 순이익을 자기자본으로 나눈 비율로, 기업이 자기자본을 얼마나 효율적으로 활용하고 있는지를 나타냅니다. ROE가 높을수록 기업의 수익성이 좋다고 볼 수 있습니다.
투자 적용: ROE가 업종 평균보다 높다면, 해당 기업이 자본을 효율적으로 운영하고 있다는 신호로, 투자 매력도가 높아질 수 있습니다.

6. 부채비율 (Debt-to-Equity Ratio, D/E Ratio)
의미: 총 부채를 자기자본으로 나눈 비율로, 기업의 재무 레버리지 정도를 나타냅니다. D/E 비율이 높을수록 기업이 부채에 의존하고 있다는 의미입니다.
투자 적용: D/E 비율이 낮은 기업은 재무적으로 안정적일 가능성이 높아, 투자자에게 더 안전한 선택이 될 수 있습니다.

예시
예를 들어, B기업의 P/E 비율이 12, P/B 비율이 0.8, ROE가 15%라고 가정해 보겠습니다. 이 경우, B기업은 상대적으로 저평가된 주식일 가능성이 있으며, 자본을 효율적으로 활용하고 있다는 신호를 보입니다. 따라서, 장기 투자자로서 B기업에 투자하는 것이 좋은 선택이 될 수 있습니다.
이러한 가치 평가 지표들을 종합적으로 분석하여 투자 결정을 내리는 것이 중요합니다. 각 지표는 서로 보완적인 역할을 하며, 이를 통해 보다 신뢰할 수 있는 투자 결정을 내릴 수 있습니다.
"""
