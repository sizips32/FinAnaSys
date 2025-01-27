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

#### 기술적 지표 설명
1. **단순이동평균선 (Simple Moving Average, SMA)**
   - 20일 동안의 종가 평균
   - 추세 방향과 지지/저항 수준 파악
   - 가격이 SMA 위 = 상승추세, 아래 = 하락추세

2. **볼린저 밴드 (Bollinger Bands)**
   - 20일 이동평균선을 중심으로 ±2 표준편차
   - 변동성과 추세 강도 측정
   - 밴드 수축 = 변동성 감소, 확장 = 변동성 증가
   - 가격이 상단/하단 밴드 접근 시 과매수/과매도 가능성

3. **VWAP (Volume Weighted Average Price)**
   - 거래량 가중 평균 가격
   - 기관투자자들의 매매 기준선으로 활용
   - VWAP 위 = 매수 우위, 아래 = 매도 우위

4. **MACD (Moving Average Convergence Divergence)**
   - 12일 EMA와 26일 EMA의 차이
   - 9일 시그널선과의 교차로 매매 시그널 생성
   - MACD > 시그널선 = 매수, MACD < 시그널선 = 매도
   - 다이버전스 발생 시 추세 전환 가능성

5. **RSI (Relative Strength Index)**
   - 14일 기준 상승/하락 비율
   - 0-100 사이 값, 70이상 과매수, 30이하 과매도
   - 중심선(50) 돌파 = 추세 전환 신호
   - 다이버전스 형성 시 강력한 매매 신호

6. **스퀴즈 모멘텀 (TTM Squeeze)**
   - 볼린저 밴드와 켈트너 채널 결합
   - 빨간점 = 변동성 수축(스퀴즈)
   - 녹색 막대 = 상승 모멘텀, 빨간 막대 = 하락 모멘텀
   - 스퀴즈 해제 시 강한 추세 시작 가능성

7. **MFI (Money Flow Index)**
   - 가격과 거래량 결합한 모멘텀 지표
   - 0-100 사이 값, 80이상 과매수, 20이하 과매도
   - RSI와 유사하나 거래량 반영으로 더 정확
   - 다이버전스 발생 시 추세 전환 신호

### 분석 전략
1. **추세 분석**
   - SMA, MACD로 주요 추세 파악
   - 볼린저 밴드로 변동성 범위 확인

2. **모멘텀 분석**
   - RSI, MFI로 과매수/과매도 판단
   - 스퀴즈 모멘텀으로 강한 추세 시작점 포착

3. **거래량 분석**
   - VWAP으로 기관 매매 동향 파악
   - MFI로 자금 흐름 강도 확인
""")

class AnalysisError(Exception):
    """분석 과정에서 발생하는 사용자 정의 예외"""
    pass

def handle_error(error, context=""):
    """에러 처리 통합 함수"""
    if isinstance(error, AnalysisError):
        st.error(f"분석 오류: {str(error)}")
    elif isinstance(error, ValueError):
        st.error(f"입력값 오류: {str(error)}")
    else:
        st.error(f"{context} 중 오류 발생: {str(error)}")
    return None

def fetch_stock_data(symbol, period):
    """주식 데이터를 가져오는 함수"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            st.error(f"{symbol}에 대한 데이터를 찾을 수 없습니다.")
            return None
            
        # 거래량이 0인 행 제거 전에 데이터 존재 여부 확인
        if 'Volume' not in data.columns:
            st.error("거래량 데이터가 존재하지 않습니다.")
            return None
            
        data = data[data['Volume'] > 0]
        
        # VWAP 계산 전에 필요한 컬럼 존재 여부 확인
        required_columns = ['High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            st.error("필요한 가격 데이터가 누락되었습니다.")
            return None
            
        data['VWAP'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['VWAP'] = (data['VWAP'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        
        return data
        
    except Exception as e:
        st.error(f"데이터 가져오기 실패: {str(e)}")
        return None

def calculate_technical_indicators(data, indicator):
    """기술적 지표 계산 함수 최적화"""
    if not isinstance(data, pd.DataFrame) or data.empty:
        return None
        
    # 계산 결과를 캐시하기 위한 딕셔너리
    if not hasattr(calculate_technical_indicators, 'cache'):
        calculate_technical_indicators.cache = {}
    
    # 캐시 키 생성
    cache_key = f"{indicator}_{data.index[-1]}"
    
    # 캐시된 결과가 있으면 반환
    if cache_key in calculate_technical_indicators.cache:
        return calculate_technical_indicators.cache[cache_key]
    
    result = None
    
    try:
        if indicator == "20-Day SMA":
            result = data['Close'].rolling(window=20).mean()
        elif indicator == "60-Day SMA":
            result = data['Close'].rolling(window=60).mean()
        elif indicator == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            result = sma, sma + 2 * std, sma - 2 * std
        elif indicator == "VWAP":
            result = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        elif indicator == "MACD":
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            result = macd, signal
        elif indicator == "RSI":
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result = 100 - (100 / (1 + rs))
        elif indicator == "Squeeze Momentum":
            # 볼린저 밴드 계산 (20일, 2표준편차)
            bb_mean = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            bb_upper = bb_mean + (2 * bb_std)
            bb_lower = bb_mean - (2 * bb_std)

            # 켈트너 채널 계산 (20일, 1.5배 ATR)
            tr = pd.DataFrame()
            tr['h-l'] = data['High'] - data['Low']
            tr['h-pc'] = abs(data['High'] - data['Close'].shift())
            tr['l-pc'] = abs(data['Low'] - data['Close'].shift())
            tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
            atr = tr['tr'].rolling(window=20).mean()

            kc_mean = data['Close'].rolling(window=20).mean()
            kc_upper = kc_mean + (1.5 * atr)
            kc_lower = kc_mean - (1.5 * atr)

            # 스퀴즈 상태 확인 (1: 스퀴즈 ON, 0: 스퀴즈 OFF)
            squeeze = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(int)

            # 모멘텀 계산
            highest = data['High'].rolling(window=20).max()
            lowest = data['Low'].rolling(window=20).min()
            mm = data['Close'] - (highest + lowest) / 2
            momentum = mm.rolling(window=20).mean()

            result = squeeze, momentum
        elif indicator == "MFI":
            # Typical Price
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            # Raw Money Flow
            raw_money_flow = typical_price * data['Volume']
            
            # Money Flow Direction
            money_flow_direction = np.where(typical_price > typical_price.shift(1), 1, -1)
            
            # Positive and Negative Money Flow
            positive_flow = pd.Series(np.where(money_flow_direction > 0, raw_money_flow, 0))
            negative_flow = pd.Series(np.where(money_flow_direction < 0, raw_money_flow, 0))
            
            # 14-period Money Flow Ratio
            positive_mf = positive_flow.rolling(window=14).sum()
            negative_mf = negative_flow.rolling(window=14).sum()
            
            # Money Flow Index
            money_flow_ratio = positive_mf / negative_mf
            result = 100 - (100 / (1 + money_flow_ratio))
        
        # 결과 캐시 저장
        calculate_technical_indicators.cache[cache_key] = result
        return result
        
    except Exception as e:
        st.error(f"지표 계산 중 오류 발생: {str(e)}")
        return None

class TechnicalAnalysis:
    def __init__(self):
        self.cache = {}
        
    def calculate_indicators(self, data, indicators):
        results = {}
        for indicator in indicators:
            results[indicator] = self.calculate_single_indicator(data, indicator)
        return results
        
    def calculate_single_indicator(self, data, indicator):
        # 기존의 calculate_technical_indicators 함수 로직
        pass
        
    def analyze_signals(self, data, symbol):
        # 기존의 calculate_signal_probabilities 함수 로직
        pass

# 메인 코드에서 사용
technical_analyzer = TechnicalAnalysis()

def get_financial_metrics(ticker):
    """기업 재무 지표 수집 함수"""
    try:
        info = ticker.info
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow  # 현금흐름표 데이터 추가
        
        # 최신 재무제표 날짜 확인
        latest_financial_date = financials.columns[0].strftime('%Y-%m-%d') if not financials.empty else '날짜 없음'
        latest_balance_date = balance_sheet.columns[0].strftime('%Y-%m-%d') if not balance_sheet.empty else '날짜 없음'
        latest_cashflow_date = cashflow.columns[0].strftime('%Y-%m-%d') if not cashflow.empty else '날짜 없음'
        
        # 현금흐름 관련 지표 추가
        operating_cashflow = cashflow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cashflow.index else None
        free_cashflow = cashflow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cashflow.index else None
        capital_expenditure = cashflow.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cashflow.index else None
        
        metrics = {
            'sector': info.get('sector', 'N/A'),  # 업종
            'industry': info.get('industry', 'N/A'),  # 세부 업종
            'marketCap': info.get('marketCap', None),  # 시가총액
            'priceToSalesTrailing12Months': info.get('priceToSalesTrailing12Months', None),  # PSR
            'dividendYield': info.get('dividendYield', None),  # 배당수익률
            'totalDebt': balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else None,
            'totalAssets': balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else None,
            'currentAssets': balance_sheet.loc['Total Current Assets'].iloc[0] if 'Total Current Assets' in balance_sheet.index else None,
            'currentLiabilities': balance_sheet.loc['Total Current Liabilities'].iloc[0] if 'Total Current Liabilities' in balance_sheet.index else None,
            'cashflow': {  # 현금흐름 정보 추가
                'operating': operating_cashflow,
                'free': free_cashflow,
                'capex': capital_expenditure
            },
            'dates': {
                'financial': latest_financial_date,
                'balance': latest_balance_date,
                'cashflow': latest_cashflow_date
            }
        }
        
        # 부채비율 계산
        if metrics['totalDebt'] is not None and metrics['totalAssets'] is not None:
            metrics['debtRatio'] = (metrics['totalDebt'] / metrics['totalAssets']) * 100
        else:
            metrics['debtRatio'] = None
            
        # 유동비율 계산
        if metrics['currentAssets'] is not None and metrics['currentLiabilities'] is not None:
            metrics['currentRatio'] = (metrics['currentAssets'] / metrics['currentLiabilities']) * 100
        else:
            metrics['currentRatio'] = None
            
        return metrics
    except Exception as e:
        st.warning(f"재무 지표 수집 중 오류 발생: {str(e)}")
        return None

def format_number(number):
    """숫자 포맷팅 함수"""
    if number is None:
        return "N/A"
    if number >= 1_000_000_000_000:
        return f"{number/1_000_000_000_000:.2f}조"
    elif number >= 100_000_000:
        return f"{number/100_000_000:.2f}억"
    elif number >= 10000:
        return f"{number/10000:.2f}만"
    return f"{number:.2f}"

def main():
    # 세션 상태 초기화 및 메모리 관리
    if 'stock_data' not in st.session_state:
        st.session_state['stock_data'] = None
    if 'last_symbol' not in st.session_state:
        st.session_state['last_symbol'] = None
    
    # 캐시 크기 제한
    MAX_CACHE_SIZE = 1000
    if hasattr(calculate_technical_indicators, 'cache'):
        if len(calculate_technical_indicators.cache) > MAX_CACHE_SIZE:
            calculate_technical_indicators.cache.clear()
    
    # 임시 파일 자동 정리
    @st.cache(allow_output_mutation=True)
    def cleanup_temp_files():
        temp_dir = tempfile.gettempdir()
        for file in os.listdir(temp_dir):
            if file.endswith('.png'):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except Exception:
                    pass
    
    st.title("AI Technical Analysis")
    
    # 사이드바 구성
    st.sidebar.header("Settings")
    
    # 티커 심볼 입력
    symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")
    
    # 기간 선택
    period = st.sidebar.selectbox(
        "Select Time Period",
        ("1mo", "3mo", "6mo", "1y", "2y", "5y", "max")
    )
    
    # Fetch Data 버튼
    if st.sidebar.button("Fetch Data", key="main_fetch_data_button"):
        try:
            data = fetch_stock_data(symbol, period)
            if data is not None:
                st.session_state['stock_data'] = data
                st.success(f"{symbol} 데이터를 성공적으로 불러왔습니다.")
            else:
                st.error("데이터를 가져오는데 실패했습니다.")
                return
        except Exception as e:
            st.error(f"데이터 로딩 중 오류 발생: {str(e)}")
            return

    # Check if data is available
    if "stock_data" in st.session_state and st.session_state["stock_data"] is not None:
        data = st.session_state["stock_data"]

        # Plot candlestick chart
        try:
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

            # Helper function to add indicators to the chart
            def add_indicator(indicator):
                if indicator == "20-Day SMA":
                    sma = calculate_technical_indicators(data, "20-Day SMA")
                    fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
                elif indicator == "60-Day SMA":
                    sma60 = calculate_technical_indicators(data, "60-Day SMA")
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=sma60,
                        name='60-Day SMA',
                        line=dict(color='orange', width=1)
                    ))
                elif indicator == "20-Day Bollinger Bands":
                    sma, bb_upper, bb_lower = calculate_technical_indicators(data, "20-Day Bollinger Bands")
                    fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
                    fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
                elif indicator == "VWAP":
                    vwap = calculate_technical_indicators(data, "VWAP")
                    fig.add_trace(go.Scatter(x=data.index, y=vwap, mode='lines', name='VWAP'))
                elif indicator == "MACD":
                    macd, signal = calculate_technical_indicators(data, "MACD")
                    # MACD를 하단에 별도의 subplot으로 표시
                    fig.add_trace(go.Scatter(x=data.index, y=macd, name='MACD',
                                           yaxis="y2"))
                    fig.add_trace(go.Scatter(x=data.index, y=signal, name='Signal',
                                           yaxis="y2"))
                    # MACD Histogram
                    fig.add_trace(go.Bar(x=data.index, y=macd-signal, name='MACD Histogram',
                                       yaxis="y2"))
                elif indicator == "RSI":
                    rsi = calculate_technical_indicators(data, "RSI")
                    # RSI를 하단에 별도의 subplot으로 표시
                    fig.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI',
                                           yaxis="y3"))
                    # 과매수/과매도 기준선
                    fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1],
                                 y0=70, y1=70, line=dict(dash="dash", color="red"),
                                 opacity=0.5, yref="y3")
                    fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1],
                                 y0=30, y1=30, line=dict(dash="dash", color="green"),
                                 opacity=0.5, yref="y3")
                elif indicator == "Squeeze Momentum":
                    squeeze, momentum = calculate_technical_indicators(data, "Squeeze Momentum")
                    
                    # 스퀴즈 상태를 나타내는 막대 그래프
                    colors = ['red' if x == 1 else 'gray' for x in squeeze]
                    
                    # 모멘텀 값을 기준으로 색상 설정
                    momentum_colors = ['green' if x >= 0 else 'red' for x in momentum]
                    
                    # 스퀴즈 모멘텀을 하단에 표시
                    fig.add_trace(go.Bar(
                        x=data.index,
                        y=momentum,
                        name='Squeeze Momentum',
                        marker_color=momentum_colors,
                        yaxis="y4"
                    ))
                    
                    # 스퀴즈 상태 표시 (점으로)
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=[min(momentum) * 1.1 if x == 1 else None for x in squeeze],
                        mode='markers',
                        marker=dict(color='red', size=8),
                        name='Squeeze',
                        yaxis="y4"
                    ))
                elif indicator == "MFI":
                    mfi = calculate_technical_indicators(data, "MFI")
                    # MFI를 하단에 별도의 subplot으로 표시
                    fig.add_trace(go.Scatter(x=data.index, y=mfi, name='MFI',
                                           yaxis="y5"))
                    
                    # 과매수/과매도 기준선
                    fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1],
                                 y0=80, y1=80, line=dict(dash="dash", color="red"),
                                 opacity=0.5, yref="y5")
                    fig.add_shape(type="line", x0=data.index[0], x1=data.index[-1],
                                 y0=20, y1=20, line=dict(dash="dash", color="green"),
                                 opacity=0.5, yref="y5")

            # Add selected indicators to the chart
            st.sidebar.subheader("Technical Indicators")
            indicators = st.sidebar.multiselect(
                "Select Indicators",
                [
                    "20-Day SMA",
                    "60-Day SMA",
                    "20-Day Bollinger Bands",
                    "VWAP",
                    "MACD",
                    "RSI",
                    "Squeeze Momentum",
                    "MFI"
                ],
                default=["20-Day SMA", "60-Day SMA", "20-Day Bollinger Bands", "VWAP"]
            )

            for indicator in indicators:
                add_indicator(indicator)

            fig.update_layout(xaxis_rangeslider_visible=False)

            # 레이아웃 업데이트
            if "Squeeze Momentum" in indicators:
                if "MACD" in indicators and "RSI" in indicators and "MFI" in indicators:
                    # 모든 지표가 있는 경우
                    fig.update_layout(
                        height=1300,
                        yaxis=dict(domain=[0.7, 1]),      # 메인 차트
                        yaxis2=dict(domain=[0.5, 0.65], title="MACD"),  # MACD
                        yaxis3=dict(domain=[0.35, 0.45], title="RSI"),   # RSI
                        yaxis5=dict(domain=[0.2, 0.3], title="MFI"),    # MFI
                        yaxis4=dict(domain=[0, 0.15], title="Squeeze Momentum")  # Squeeze
                    )
                elif len([x for x in ["MACD", "RSI", "MFI"] if x in indicators]) == 2:
                    # 세 개의 지표가 있는 경우
                    fig.update_layout(
                        height=1100,
                        yaxis=dict(domain=[0.6, 1]),
                        yaxis2=dict(domain=[0.4, 0.55], title="First Indicator"),
                        yaxis3=dict(domain=[0.2, 0.35], title="Second Indicator"),
                        yaxis4=dict(domain=[0, 0.15], title="Squeeze Momentum")
                    )
                else:
                    # 두 개의 지표가 있는 경우
                    fig.update_layout(
                        height=900,
                        yaxis=dict(domain=[0.5, 1]),
                        yaxis2=dict(domain=[0.25, 0.45], title="Indicator"),
                        yaxis4=dict(domain=[0, 0.2], title="Squeeze Momentum")
                    )
            else:
                if "MFI" in indicators:
                    if "MACD" in indicators and "RSI" in indicators:
                        # MFI, MACD, RSI가 있는 경우
                        fig.update_layout(
                            height=1100,
                            yaxis=dict(domain=[0.7, 1]),
                            yaxis2=dict(domain=[0.5, 0.65], title="MACD"),
                            yaxis3=dict(domain=[0.25, 0.45], title="RSI"),
                            yaxis5=dict(domain=[0.1, 0.3], title="MFI")
                        )
                    elif "MACD" in indicators or "RSI" in indicators:
                        # MFI와 다른 하나의 지표가 있는 경우
                        fig.update_layout(
                            height=900,
                            yaxis=dict(domain=[0.6, 1]),
                            yaxis2=dict(domain=[0.35, 0.55], title="MACD" if "MACD" in indicators else "RSI"),
                            yaxis5=dict(domain=[0.1, 0.3], title="MFI")
                        )
                    else:
                        # MFI만 있는 경우
                        fig.update_layout(
                            height=700,
                            yaxis=dict(domain=[0.35, 1]),
                            yaxis5=dict(domain=[0, 0.25], title="MFI")
                        )
                else:
                    # 기존 레이아웃 유지
                    fig.update_layout(
                        height=500,
                        yaxis=dict(domain=[0.5, 1]),
                        yaxis2=dict(domain=[0.25, 0.75], title="Technical Indicators")
                    )

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

            if st.button("Run AI Analysis", key="main_ai_analysis_button"):
                with st.spinner("Analyzing the chart, please wait..."):
                    try:
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
                    except Exception as e:
                        st.error(f"AI 분석 중 오류 발생: {str(e)}")

            def calculate_signal_probabilities(data, symbol):
                """각 지표별 시그널을 분석하여 매수/매도/관망 확률 계산"""
                
                signals = {
                    'trend': 0,
                    'momentum': 0,
                    'volatility': 0,
                    'volume': 0,
                    'fundamental': 0  # 초기값 설정
                }
                
                weights = {
                    'trend': 0.25,      # 추세 지표 (SMA, MACD)
                    'momentum': 0.25,   # 모멘텀 지표 (RSI, MFI)
                    'volatility': 0.2,  # 변동성 지표 (볼린저 밴드, 스퀴즈)
                    'volume': 0.15,     # 거래량 지표 (VWAP)
                    'fundamental': 0.15 # 재무 지표 (ROE, PER, PBR)
                }
                
                try:
                    # 1. 추세 분석
                    sma = calculate_technical_indicators(data, "20-Day SMA")
                    macd, signal = calculate_technical_indicators(data, "MACD")
                    
                    # SMA 시그널 (-1 ~ 1)
                    current_price = data['Close'].iloc[-1]
                    sma_signal = (current_price - sma.iloc[-1]) / sma.iloc[-1]
                    sma_signal = max(min(sma_signal, 1), -1)
                    
                    # MACD 시그널 (-1 ~ 1)
                    macd_signal = (macd.iloc[-1] - signal.iloc[-1]) / abs(signal.iloc[-1])
                    macd_signal = max(min(macd_signal, 1), -1)
                    
                    signals['trend'] = (sma_signal + macd_signal) / 2
                    
                    # 2. 모멘텀 분석
                    rsi = calculate_technical_indicators(data, "RSI")
                    mfi = calculate_technical_indicators(data, "MFI")
                    
                    # RSI 시그널 (-1 ~ 1)
                    rsi_value = rsi.iloc[-1]
                    rsi_signal = 0
                    if rsi_value > 70: rsi_signal = -1
                    elif rsi_value < 30: rsi_signal = 1
                    else: rsi_signal = (rsi_value - 50) / 20
                    
                    # MFI 시그널 (-1 ~ 1)
                    mfi_value = mfi.iloc[-1]
                    mfi_signal = 0
                    if mfi_value > 80: mfi_signal = -1
                    elif mfi_value < 20: mfi_signal = 1
                    else: mfi_signal = (mfi_value - 50) / 30
                    
                    signals['momentum'] = (rsi_signal + mfi_signal) / 2
                    
                    # 3. 변동성 분석
                    sma, bb_upper, bb_lower = calculate_technical_indicators(data, "20-Day Bollinger Bands")
                    squeeze, momentum = calculate_technical_indicators(data, "Squeeze Momentum")
                    
                    # 볼린저 밴드 시그널 (-1 ~ 1)
                    bb_middle = sma.iloc[-1]
                    bb_signal = 0
                    if current_price > bb_upper.iloc[-1]: bb_signal = -1
                    elif current_price < bb_lower.iloc[-1]: bb_signal = 1
                    else: bb_signal = (current_price - bb_middle) / (bb_upper.iloc[-1] - bb_middle)
                    
                    # 스퀴즈 모멘텀 시그널 (-1 ~ 1)
                    squeeze_signal = 1 if momentum.iloc[-1] > 0 else -1
                    
                    signals['volatility'] = (bb_signal + squeeze_signal) / 2
                    
                    # 4. 거래량 분석
                    vwap = calculate_technical_indicators(data, "VWAP")
                    volume_ma = data['Volume'].rolling(window=20).mean()
                    
                    # VWAP 시그널 (-1 ~ 1)
                    vwap_signal = (current_price - vwap.iloc[-1]) / vwap.iloc[-1]
                    vwap_signal = max(min(vwap_signal, 1), -1)
                    
                    # 거래량 증감 시그널 (-1 ~ 1)
                    volume_signal = (data['Volume'].iloc[-1] - volume_ma.iloc[-1]) / volume_ma.iloc[-1]
                    volume_signal = max(min(volume_signal, 1), -1)
                    
                    signals['volume'] = (vwap_signal + volume_signal) / 2
                    
                    # 5. 재무 분석
                    try:
                        ticker = yf.Ticker(symbol)
                        
                        # 재무제표 데이터 가져오기
                        financials = ticker.financials
                        balance_sheet = ticker.balance_sheet
                        
                        # ROE 계산
                        if not financials.empty and not balance_sheet.empty:
                            try:
                                # 당기순이익 가져오기
                                if 'Net Income' in financials.index:
                                    net_income = financials.loc['Net Income'].iloc[0]
                                else:
                                    net_income = None
                                
                                # 자기자본 가져오기 (여러 가능한 키 시도)
                                equity_keys = ['Stockholders Equity', 'Total Stockholder Equity', 
                                             'Total Equity', 'Shareholders Equity']
                                total_equity = None
                                
                                for key in equity_keys:
                                    if key in balance_sheet.index:
                                        total_equity = balance_sheet.loc[key].iloc[0]
                                        break
                                
                                # ROE 계산 및 시그널 생성
                                if net_income is not None and total_equity is not None and total_equity != 0:
                                    roe = (net_income / total_equity) * 100
                                    
                                    if roe > 15: roe_signal = 1
                                    elif roe > 10: roe_signal = 0.5
                                    elif roe > 5: roe_signal = 0
                                    else: roe_signal = -1
                                else:
                                    roe = None
                                    roe_signal = 0
                                    
                            except Exception as e:
                                st.warning(f"ROE 계산 중 오류 발생: {str(e)}")
                                roe = None
                                roe_signal = 0
                        else:
                            roe = None
                            roe_signal = 0
                        
                        # 기타 재무 정보
                        info = ticker.info
                        
                        # PER 분석
                        per = info.get('forwardPE')
                        if per and per > 0:
                            if per < 10: per_signal = 1
                            elif per < 20: per_signal = 0.5
                            elif per < 30: per_signal = -0.5
                            else: per_signal = -1
                        else:
                            per = None
                            per_signal = 0
                        
                        # PBR 분석
                        pbr = info.get('priceToBook')
                        if pbr and pbr > 0:
                            if pbr < 1: pbr_signal = 1
                            elif pbr < 3: pbr_signal = 0.5
                            elif pbr < 5: pbr_signal = -0.5
                            else: pbr_signal = -1
                        else:
                            pbr = None
                            pbr_signal = 0
                        
                        # 재무 종합 점수 계산
                        signals['fundamental'] = (roe_signal + per_signal + pbr_signal) / 3
                        
                    except Exception as e:
                        st.warning(f"재무 데이터 분석 중 오류 발생: {str(e)}")
                        roe, per, pbr = None, None, None
                    
                    # 최종 확률 계산
                    final_score = sum(signals[k] * weights[k] for k in weights)
                    
                    # 확률 변환 (시그모이드 함수 사용)
                    def sigmoid(x): return 1 / (1 + np.exp(-5 * x))
                    
                    base_prob = sigmoid(final_score)
                    buy_prob = round(base_prob * 100, 1)
                    sell_prob = round((1 - base_prob) * 100, 1)
                    hold_prob = round((1 - abs(2 * base_prob - 1)) * 100, 1)
                    
                    return {
                        'buy': buy_prob,
                        'sell': sell_prob,
                        'hold': hold_prob,
                        'signals': signals,
                        'details': {
                            'roe': roe,
                            'per': per,
                            'pbr': pbr,
                            'rsi': rsi_value if 'rsi_value' in locals() else None,
                            'mfi': mfi_value if 'mfi_value' in locals() else None
                        }
                    }
                    
                except Exception as e:
                    st.error(f"확률 계산 중 오류 발생: {str(e)}")
                    return None

            # 확률 계산
            probabilities = calculate_signal_probabilities(data, symbol)
            
            if probabilities:
                # 확률 표시
                st.subheader("투자 의사결정 확률")
                
                # 확률 게이지 표시
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("매수 확률", f"{probabilities['buy']}%")
                    if probabilities['buy'] > 60:
                        st.success("강력 매수 시그널")
                    elif probabilities['buy'] > 40:
                        st.info("매수 고려")
                
                with col2:
                    st.metric("관망 확률", f"{probabilities['hold']}%")
                    if probabilities['hold'] > 60:
                        st.warning("관망 권장")
                
                with col3:
                    st.metric("매도 확률", f"{probabilities['sell']}%")
                    if probabilities['sell'] > 60:
                        st.error("강력 매도 시그널")
                    elif probabilities['sell'] > 40:
                        st.warning("매도 고려")
                
                # 상세 분석 결과
                st.subheader("상세 분석")
                st.write("각 지표별 시그널 강도 (-1: 매도, 0: 중립, 1: 매수)")
                signals = probabilities['signals']
                
                signal_desc = {
                    'trend': '추세',
                    'momentum': '모멘텀',
                    'volatility': '변동성',
                    'volume': '거래량',
                    'fundamental': '재무'  # fundamental 키 추가
                }
                
                for key, value in signals.items():
                    st.write(f"**{signal_desc[key]}**: {value:.2f}")
                    
                # 투자 제안
                st.subheader("투자 제안")
                max_prob = max(probabilities['buy'], probabilities['sell'], probabilities['hold'])
                
                if max_prob == probabilities['buy']:
                    st.success("""
                    **매수 포지션 추천**
                    - 추세와 모멘텀이 상승을 지지
                    - 리스크 관리를 위해 분할 매수 고려
                    """)
                elif max_prob == probabilities['sell']:
                    st.error("""
                    **매도 포지션 추천**
                    - 하락 추세와 부정적 모멘텀 감지
                    - 보유 중인 경우 손절 고려
                    """)
                else:
                    st.info("""
                    **관망 추천**
                    - 명확한 방향성 부재
                    - 추가 시그널 확인 후 포지션 진입 고려
                    """)

                # 재무 지표 정보 표시
                st.subheader("🏢 기업 정보 및 재무 지표")
                
                # 기업 기본 정보
                metrics = get_financial_metrics(yf.Ticker(symbol))
                if metrics:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 기업 기본 정보")
                        st.markdown(f"""
                        - 🏭 업종: {metrics['sector']}
                        - 🔍 세부업종: {metrics['industry']}
                        - 💰 시가총액: {format_number(metrics['marketCap'])}
                        """)
                    
                    with col2:
                        st.markdown("#### 📅 재무제표 기준일")
                        st.markdown(f"""
                        - 📊 재무상태표: {metrics['dates']['balance']}
                        - 💵 손익계산서: {metrics['dates']['financial']}
                        """)
                
                # 재무 비율 표시
                st.markdown("#### 📈 주요 재무 비율")
                col1, col2, col3, col4 = st.columns(4)
                col5, col6, col7 = st.columns(3)  # 새로운 행 추가
                
                with col1:
                    if metrics['priceToSalesTrailing12Months'] is not None:
                        psr = metrics['priceToSalesTrailing12Months']
                        st.metric("💹 PSR", f"{psr:.2f}")
                        if psr < 1:
                            st.success("매우 저평가")
                        elif psr < 3:
                            st.info("적정 수준")
                        else:
                            st.warning("고평가")
                
                with col2:
                    if metrics['dividendYield'] is not None:
                        div_yield = metrics['dividendYield'] * 100
                        st.metric("💰 배당수익률", f"{div_yield:.2f}%")
                        if div_yield > 5:
                            st.success("높은 배당")
                        elif div_yield > 2:
                            st.info("적정 배당")
                        else:
                            st.warning("낮은 배당")
                
                with col3:
                    if metrics['debtRatio'] is not None:
                        debt_ratio = metrics['debtRatio']
                        st.metric("🏦 부채비율", f"{debt_ratio:.2f}%")
                        if debt_ratio < 100:
                            st.success("안정적")
                        elif debt_ratio < 200:
                            st.info("보통")
                        else:
                            st.warning("위험")
                
                with col4:
                    if metrics['currentRatio'] is not None:
                        current_ratio = metrics['currentRatio']
                        st.metric("💵 유동비율", f"{current_ratio:.2f}%")
                        if current_ratio > 200:
                            st.success("매우 안정적")
                        elif current_ratio > 100:
                            st.info("안정적")
                        else:
                            st.warning("주의 필요")

                # 새로운 재무 비율 추가
                with col5:
                    if probabilities['details']['per'] is not None:
                        per = probabilities['details']['per']
                        st.metric("📊 PER", f"{per:.2f}")
                        if per < 10:
                            st.success("저평가")
                        elif per < 20:
                            st.info("적정 수준")
                        else:
                            st.warning("고평가")

                with col6:
                    if probabilities['details']['pbr'] is not None:
                        pbr = probabilities['details']['pbr']
                        st.metric("📚 PBR", f"{pbr:.2f}")
                        if pbr < 1:
                            st.success("청산가치 이하")
                        elif pbr < 3:
                            st.info("적정 수준")
                        else:
                            st.warning("고평가")

                with col7:
                    if probabilities['details']['roe'] is not None:
                        roe = probabilities['details']['roe']
                        st.metric("💫 ROE", f"{roe:.2f}%")
                        if roe > 15:
                            st.success("수익성 우수")
                        elif roe > 10:
                            st.info("적정 수준")
                        else:
                            st.warning("수익성 저조")

                # 주요 재무비율 설명 추가
                with st.expander("💡 주요 재무비율 설명"):
                    st.markdown("""
                    ### 주요 재무비율 해석 가이드
                    
                    #### 1️⃣ 수익성 지표
                    - **ROE (자기자본이익률)**
                        - 15% 이상: 우수한 수익성
                        - 10~15%: 양호한 수익성
                        - 10% 미만: 개선 필요
                    
                    #### 2️⃣ 밸류에이션 지표
                    - **PSR (주가매출비율)**
                        - 1 미만: 매우 저평가
                        - 1~3: 적정 수준
                        - 3 초과: 고평가 가능성
                    
                    - **PER (주가수익비율)**
                        - 10 미만: 저평가 가능성
                        - 10~20: 적정 수준
                        - 20 초과: 고평가 가능성
                    
                    - **PBR (주가순자산비율)**
                        - 1 미만: 청산가치 이하
                        - 1~3: 적정 수준
                        - 3 초과: 고평가 가능성
                    
                    #### 3️⃣ 안정성 지표
                    - **부채비율**
                        - 100% 미만: 매우 안정적
                        - 100~200%: 보통
                        - 200% 초과: 재무건전성 주의
                    
                    - **유동비율**
                        - 200% 초과: 매우 안정적
                        - 100~200%: 양호
                        - 100% 미만: 단기지급능력 주의
                    
                    #### 4️⃣ 수익 환원 지표
                    - **배당수익률**
                        - 5% 초과: 고배당
                        - 2~5%: 적정 배당
                        - 2% 미만: 저배당
                    
                    ### 💡 투자 시 고려사항
                    1. 단일 지표가 아닌 여러 지표를 종합적으로 분석
                    2. 동일 업종 내 다른 기업들과 비교 분석 필요
                    3. 과거 추세와 현재 지표 변화 방향성 고려
                    4. 기업의 성장단계와 산업 특성 반영
                    """)

                # 현금흐름 정보 표시 (재무 비율 표시 다음에 추가)
                if metrics and metrics.get('cashflow'):
                    st.markdown("#### 💰 현금흐름 분석")
                    cf_col1, cf_col2, cf_col3 = st.columns(3)
                    
                    with cf_col1:
                        if metrics['cashflow']['operating'] is not None:
                            operating_cf = metrics['cashflow']['operating']
                            st.metric("영업현금흐름", format_number(operating_cf))
                            if operating_cf > 0:
                                st.success("양호한 영업활동")
                            else:
                                st.error("영업활동 주의")
                    
                    with cf_col2:
                        if metrics['cashflow']['free'] is not None:
                            free_cf = metrics['cashflow']['free']
                            st.metric("잉여현금흐름", format_number(free_cf))
                            if free_cf > 0:
                                st.success("투자여력 있음")
                            else:
                                st.warning("투자여력 부족")
                    
                    with cf_col3:
                        if metrics['cashflow']['capex'] is not None:
                            capex = abs(metrics['cashflow']['capex'])  # CAPEX는 보통 음수로 표시되므로 절대값 처리
                            st.metric("설비투자", format_number(capex))
                            
                    # 현금흐름 설명 추가
                    with st.expander("💡 현금흐름 지표 설명"):
                        st.markdown("""
                        ### 현금흐름 지표 해석 가이드
                        
                        #### 1️⃣ 영업현금흐름
                        - 기업의 주된 영업활동에서 발생한 현금흐름
                        - 양(+)의 값: 건전한 영업활동
                        - 음(-)의 값: 영업활동에서 현금 유출 발생
                        
                        #### 2️⃣ 잉여현금흐름
                        - 영업현금흐름에서 필수 투자비용을 제외한 금액
                        - 양(+)의 값: 추가 투자/배당 여력 존재
                        - 음(-)의 값: 외부 자금조달 필요 가능성
                        
                        #### 3️⃣ 설비투자(CAPEX)
                        - 기업의 생산설비, 부동산 등 고정자산 투자금액
                        - 높은 투자: 미래 성장을 위한 투자 진행
                        - 낮은 투자: 보수적 재무운영 또는 성장 둔화
                        """)

            # Footer
            st.sidebar.markdown("---")
            
            st.sidebar.text("Created by Sean J. Kim")

        except Exception as e:
            st.error(f"차트 생성 중 오류 발생: {str(e)}")
            return
    else:
        st.info("👆 왼쪽 사이드바에서 종목 심볼을 입력하고 'Fetch Data' 버튼을 클릭하세요.")
        return

if __name__ == "__main__":
    main()

