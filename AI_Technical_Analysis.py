# Source: @DeepCharts Youtube Channel (https://www.youtube.com/@DeepCharts)

# NOTE: Set yfinance to the following version to get chart working: 
# "pip install yfinance==0.2.40"

import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import random

st.set_page_config(layout="wide")
st.title("AI 기반 재무-기술적 분석 시스템")

with st.expander("📚 시스템 소개"):
    st.markdown("""
    이 시스템은 주식 시장 데이터를 분석하여 투자 의사결정을 지원하는 AI 기반 분석 도구입니다.

    #### 주요 기능
    1. 실시간 주가 데이터 분석
    2. 기술적 지표 기반 매매 시그널
    3. AI 기반 투자 추천
    4. 재무제표 분석
    """)

with st.expander("📊 분석 방법론"):
    st.markdown("""
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
    """)

with st.expander("📈 분석 전략"):
    st.markdown("""
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

# 패키지 버전 검증 로직 수정
def verify_package_versions():
    """필수 패키지 설치 여부 확인"""
    required_packages = ['streamlit', 'pandas', 'numpy', 'plotly', 'sklearn']
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.error(f"""
        다음 패키지들이 설치되지 않았습니다: {', '.join(missing_packages)}
        터미널에서 다음 명령어를 실행하세요:
        pip install {' '.join(missing_packages)}
        """)
        return False
    return True

# 메인 코드 시작 전에 패키지 검증
if not verify_package_versions():
    st.stop()

try:
    yf_version = yf.__version__
    st.info(f"현재 yfinance 버전: {yf_version}")
except Exception as e:
    st.error(f"yfinance 패키지 버전 확인 중 오류 발생: {str(e)}")

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
        # 기간에 따른 시작일 계산
        end_date = datetime.now()
        if period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
        else:  # max
            start_date = end_date - timedelta(days=3650)  # 약 10년

        # 한국 주식 여부 확인
        krx = fdr.StockListing('KRX')
        is_korean = False
        
        # KRX 데이터프레임의 컬럼명 확인 및 처리
        symbol_column = None
        name_column = None
        
        if 'Symbol' in krx.columns:
            symbol_column = 'Symbol'
            name_column = 'Name'
        elif 'Code' in krx.columns:
            symbol_column = 'Code'
            name_column = 'Name'
        elif '종목코드' in krx.columns:
            symbol_column = '종목코드'
            name_column = '종목명'
            
        if symbol_column is None:
            st.error("주식 시장 데이터 형식이 올바르지 않습니다.")
            return None
            
        is_korean = symbol in krx[symbol_column].values

        # 미국 주식 여부 확인
        nasdaq = fdr.StockListing('NASDAQ')
        nyse = fdr.StockListing('NYSE')
        
        # NASDAQ/NYSE 데이터프레임의 컬럼명 확인 및 처리
        us_symbol_column = 'Symbol' if 'Symbol' in nasdaq.columns else 'Code'
        us_name_column = 'Name' if 'Name' in nasdaq.columns else 'Symbol'
        
        is_us = (symbol in nasdaq[us_symbol_column].values or 
                symbol in nyse[us_symbol_column].values)

        if not (is_korean or is_us):
            st.error(f"{symbol}은(는) 유효한 종목코드가 아닙니다.")
            return None

        # 주식 데이터 가져오기
        if is_korean:
            market_type = "KRX"
            data = fdr.DataReader(symbol, start_date, end_date)
            company_name = krx[krx[symbol_column] == symbol][name_column].iloc[0]
        else:
            market_type = "US"
            data = fdr.DataReader(symbol, start_date, end_date)
            if symbol in nasdaq[us_symbol_column].values:
                company_name = nasdaq[nasdaq[us_symbol_column] == symbol][us_name_column].iloc[0]
            else:
                company_name = (nyse[nyse[us_symbol_column] == symbol]
                              [us_name_column].iloc[0])

        if data.empty:
            st.error(f"{symbol}에 대한 데이터를 찾을 수 없습니다.")
            return None

        # 거래량이 0인 행 제거
        if 'Volume' not in data.columns:
            st.error("거래량 데이터가 존재하지 않습니다.")
            return None

        data = data[data['Volume'] > 0]

        # VWAP 계산
        data['VWAP'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['VWAP'] = ((data['VWAP'] * data['Volume']).cumsum() / 
                        data['Volume'].cumsum())

        # 시장 정보와 회사명 추가
        data.attrs['market_type'] = market_type
        data.attrs['company_name'] = company_name

        return data

    except Exception as e:
        st.error(f"데이터 가져오기 실패: {str(e)}\n상세 에러: {type(e).__name__}")
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

def get_yahoo_symbol(symbol, market_type):
    """Yahoo Finance 심볼로 변환"""
    if market_type == "KRX":
        try:
            # 종목 코드를 6자리로 맞추기
            symbol = symbol.zfill(6)
            
            krx = fdr.StockListing('KRX')
            market_column = None
            symbol_column = None
            
            # KRX 데이터프레임의 컬럼명 확인
            if 'Market' in krx.columns:
                market_column = 'Market'
            elif 'SecuGroup' in krx.columns:
                market_column = 'SecuGroup'
            elif '시장구분' in krx.columns:
                market_column = '시장구분'
            elif 'MarketId' in krx.columns:
                market_column = 'MarketId'
                
            if 'Symbol' in krx.columns:
                symbol_column = 'Symbol'
            elif 'Code' in krx.columns:
                symbol_column = 'Code'
            elif '종목코드' in krx.columns:
                symbol_column = '종목코드'
                
            if not market_column or not symbol_column:
                st.warning("시장 구분 정보를 찾을 수 없습니다. KOSPI로 가정합니다.")
                return f"{symbol}.KS"
                
            stock_info = krx[krx[symbol_column] == symbol]
            if stock_info.empty:
                st.warning(f"{symbol} 종목을 KRX 상장 종목 목록에서 찾을 수 없습니다.")
                return f"{symbol}.KS"
                
            market = str(stock_info[market_column].iloc[0]).upper()
            
            # KOSDAQ 여부 확인
            kosdaq_keywords = ['KOSDAQ', 'KOSDAQ', '코스닥', 'KQ', 'KSQ']
            if any(keyword in market for keyword in kosdaq_keywords):
                return f"{symbol}.KQ"
                
            # KOSPI 여부 확인
            kospi_keywords = ['KOSPI', 'KOSPI', '코스피', 'KS', 'KSE']
            if any(keyword in market for keyword in kospi_keywords):
                return f"{symbol}.KS"
                
            # 기본값으로 KOSPI 처리
            st.warning(f"시장 구분이 명확하지 않습니다: {market}. KOSPI로 가정합니다.")
            return f"{symbol}.KS"
            
        except Exception as e:
            st.warning(f"시장 구분 확인 중 오류 발생: {str(e)}")
            return f"{symbol}.KS"  # 에러 발생 시 기본값으로 KOSPI 처리
            
    return symbol  # US stocks

def get_financial_metrics(symbol):
    """기업 재무 지표 수집 함수"""
    MAX_RETRIES = 3  # 최대 재시도 횟수
    BASE_DELAY = 2   # 기본 대기 시간 (초)
    
    def fetch_with_retry(ticker, operation_name, operation_func):
        """재시도 로직이 포함된 데이터 가져오기 함수"""
        for attempt in range(MAX_RETRIES):
            try:
                return operation_func()
            except Exception as e:
                if "429" in str(e):  # Too Many Requests 에러
                    if attempt < MAX_RETRIES - 1:  # 마지막 시도가 아닌 경우
                        delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)  # 지수 백오프 + 무작위성
                        st.warning(f"{operation_name} 데이터 가져오기 재시도 중... ({attempt + 1}/{MAX_RETRIES})")
                        time.sleep(delay)
                        continue
                st.error(f"{operation_name} 데이터 가져오기 실패: {str(e)}")
                return None
        return None

    try:
        # 기본 정보 가져오기
        krx = fdr.StockListing('KRX')
        is_korean = False
        
        # KRX 데이터프레임의 컬럼명 확인 및 처리
        symbol_column = None
        name_column = None
        
        if 'Symbol' in krx.columns:
            symbol_column = 'Symbol'
            name_column = 'Name'
        elif 'Code' in krx.columns:
            symbol_column = 'Code'
            name_column = 'Name'
        elif '종목코드' in krx.columns:
            symbol_column = '종목코드'
            name_column = '종목명'
            
        if symbol_column is None:
            st.error("주식 시장 데이터 형식이 올바르지 않습니다.")
            return None
            
        is_korean = symbol in krx[symbol_column].values
        
        # 한국 주식인 경우
        if is_korean:
            company_info = krx[krx[symbol_column] == symbol].iloc[0]
            sector = company_info.get('Sector', 'N/A')
            industry = company_info.get('Industry', 'N/A')
            market_cap = company_info.get('MarketCap', None)
            market_type = "KRX"
            
        # 미국 주식인 경우
        else:
            nasdaq = fdr.StockListing('NASDAQ')
            nyse = fdr.StockListing('NYSE')
            
            if symbol in nasdaq['Symbol'].values:
                company_info = nasdaq[nasdaq['Symbol'] == symbol].iloc[0]
                market_type = "NASDAQ"
            elif symbol in nyse['Symbol'].values:
                company_info = nyse[nyse['Symbol'] == symbol].iloc[0]
                market_type = "NYSE"
            else:
                st.error(f"{symbol}은(는) 지원하지 않는 종목입니다.")
                return None
                
            sector = company_info.get('Sector', 'N/A')
            industry = company_info.get('Industry', 'N/A')
            market_cap = company_info.get('MarketCap', None)

        # Yahoo Finance에서 재무제표 데이터 가져오기
        try:
            ticker = yf.Ticker(symbol)
            
            # 재무제표 데이터 가져오기 (재시도 로직 적용)
            financials = fetch_with_retry(ticker, "손익계산서", lambda: ticker.get_financials())
            balance_sheet = fetch_with_retry(ticker, "재무상태표", lambda: ticker.get_balance_sheet())
            info = fetch_with_retry(ticker, "기업 정보", lambda: ticker.info)

            if financials is None and balance_sheet is None and info is None:
                st.error("Yahoo Finance API에서 데이터를 가져올 수 없습니다. 잠시 후 다시 시도해주세요.")
                return None

            # 당기순이익 가져오기
            net_income = None
            try:
                if financials is not None and not financials.empty:
                    if 'Net Income' in financials.index:
                        net_income = financials.loc['Net Income'].iloc[0]
                    elif 'NetIncome' in financials.index:
                        net_income = financials.loc['NetIncome'].iloc[0]
            except (KeyError, AttributeError) as e:
                st.warning(f"당기순이익 데이터 추출 실패: {str(e)}")

            # 자기자본 가져오기
            total_equity = None
            try:
                if balance_sheet is not None and not balance_sheet.empty:
                    if 'Total Stockholder Equity' in balance_sheet.index:
                        total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                    elif 'StockholderEquity' in balance_sheet.index:
                        total_equity = balance_sheet.loc['StockholderEquity'].iloc[0]
            except (KeyError, AttributeError) as e:
                st.warning(f"자기자본 데이터 추출 실패: {str(e)}")

            # ROE 계산 및 시그널 생성
            if net_income is not None and total_equity is not None and total_equity != 0:
                roe = (net_income / total_equity) * 100
                
                if roe > 15:
                    roe_signal = 1
                elif roe > 10:
                    roe_signal = 0.5
                elif roe > 5:
                    roe_signal = 0
                else:
                    roe_signal = -1
            else:
                roe = None
                roe_signal = 0
                
            # 기타 재무 정보
            info = ticker.info
            
            # PER 분석
            per = info.get('forwardPE')
            per_signal = 0
            if per and per > 0:
                if per < 10:
                    per_signal = 1
                elif per < 20:
                    per_signal = 0.5
                elif per < 30:
                    per_signal = -0.5
                else:
                    per_signal = -1
            
            # PBR 분석
            pbr = info.get('priceToBook')
            pbr_signal = 0
            if pbr and pbr > 0:
                if pbr < 1:
                    pbr_signal = 1
                elif pbr < 3:
                    pbr_signal = 0.5
                elif pbr < 5:
                    pbr_signal = -0.5
                else:
                    pbr_signal = -1
            
            # 재무 종합 점수 계산
            signals = {
                'per': per,
                'pbr': pbr,
                'eps': info.get('last_year_eps', None),
                'bps': None,  # 직접 계산 필요
                'dividend_yield': info.get('dividend_yield', 0) * 100 if info.get('dividend_yield') else None,
                'market_cap': info.get('market_cap', market_cap),
                'current_price': info.get('last_price', None),
                'avg_volume': info.get('three_month_avg_volume', None)
            }
            
            # BPS 계산 (자기자본/발행주식수)
            try:
                if not balance_sheet.empty:
                    total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[-1]
                    shares_outstanding = info.get('shares_outstanding', None)
                    if total_equity and shares_outstanding:
                        signals['bps'] = total_equity / shares_outstanding
            except Exception as bps_error:
                st.warning(f"BPS 계산 실패: {str(bps_error)}")
            
            # 최근 1년간의 주가 데이터
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                stock_data = ticker.history(period='1y')
                
                if not stock_data.empty:
                    signals['current_price'] = stock_data['Close'].iloc[-1]
                    signals['avg_volume'] = stock_data['Volume'].mean()
            except Exception as price_error:
                st.warning(f"주가 데이터 조회 실패: {str(price_error)}")
            
            return {
                'market_type': market_type,
                'sector': sector,
                'industry': industry,
                'marketCap': signals['market_cap'],
                'currentPrice': signals['current_price'],
                'avgVolume': signals['avg_volume'],
                'per': signals['per'],
                'pbr': signals['pbr'],
                'eps': signals['eps'],
                'bps': signals['bps'],
                'dividendYield': signals['dividend_yield'],
                'dates': {
                    'financial': start_date.strftime('%Y-%m-%d') if 'start_date' in locals() else None,
                    'balance': start_date.strftime('%Y-%m-%d') if 'start_date' in locals() else None,
                    'cashflow': start_date.strftime('%Y-%m-%d') if 'start_date' in locals() else None
                }
            }
                
        except Exception as e:
            st.warning(f"Yahoo Finance 데이터 가져오기 실패: {str(e)}")
            return None
        
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
    @st.cache_resource
    def cleanup_temp_files():
        temp_dir = tempfile.gettempdir()
        for file in os.listdir(temp_dir):
            if file.endswith('.png'):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except Exception:
                    pass
        return None  # 명시적 반환값 추가
    
    st.title("AI Technical Analysis")
    
    # 사이드바 구성
    st.sidebar.header("Settings")
    
    # 티커 심볼 입력 안내문구 수정
    st.sidebar.markdown("""
    ### 종목 코드 입력 가이드
    - 국내 주식: 종목코드 (예: '005930' for 삼성전자)
    - 미국 주식: 심볼 (예: 'AAPL' for Apple)
    """)
    
    # 티커 심볼 입력
    symbol = st.sidebar.text_input("Enter Stock Symbol:", "005930")
    
    # 기간 선택
    period = st.sidebar.selectbox(
        "Select Time Period",
        ("1mo", "3mo", "6mo", "1y", "2y", "5y", "max")
    )
    
    # Fetch Data 버튼
    if st.sidebar.button("Fetch Data"):
        try:
            data = fetch_stock_data(symbol, period)
            if data is not None:
                st.session_state['stock_data'] = data
                market_type = data.attrs.get('market_type', 'Unknown')
                company_name = data.attrs.get('company_name', symbol)
                
                # 시장 정보와 회사명 표시
                st.success(f"""
                데이터를 성공적으로 불러왔습니다.
                - 시장: {market_type}
                - 종목명: {company_name} ({symbol})
                """)
            else:
                st.error("데이터를 가져오는데 실패했습니다.")
                return
        except Exception as e:
            st.error(f"데이터 로딩 중 오류 발생: {str(e)}")
            return

    # Check if data is available
    if "stock_data" in st.session_state and st.session_state["stock_data"] is not None:
        data = st.session_state["stock_data"]
        market_type = data.attrs.get('market_type', 'Unknown')
        company_name = data.attrs.get('company_name', symbol)

        # 차트 제목에 시장 정보와 회사명 추가
        st.subheader(f"{market_type} - {company_name} ({symbol}) 차트 분석")

        # Plot candlestick chart
        try:
            fig = go.Figure(data=[
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=f"{company_name} ({symbol})"
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
                try:
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
                        try:
                            # 최근 재무제표 데이터 가져오기
                            financials = ticker.get_financials()
                            if financials.empty:
                                st.warning("손익계산서 데이터를 가져올 수 없습니다.")
                            
                            balance_sheet = ticker.get_balance_sheet()
                            if balance_sheet.empty:
                                st.warning("재무상태표 데이터를 가져올 수 없습니다.")

                            # 당기순이익 가져오기
                            net_income = None
                            try:
                                if not financials.empty:
                                    # 최근 연간 당기순이익 데이터 찾기
                                    if 'Net Income' in financials.index:
                                        net_income = financials.loc['Net Income'].iloc[0]
                                    elif 'NetIncome' in financials.index:
                                        net_income = financials.loc['NetIncome'].iloc[0]
                            except (KeyError, AttributeError) as e:
                                st.warning(f"당기순이익 데이터 추출 실패: {str(e)}")

                            # 자기자본 가져오기
                            total_equity = None
                            try:
                                if not balance_sheet.empty:
                                    # 최근 자기자본 데이터 찾기
                                    if 'Total Stockholder Equity' in balance_sheet.index:
                                        total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                                    elif 'StockholderEquity' in balance_sheet.index:
                                        total_equity = balance_sheet.loc['StockholderEquity'].iloc[0]
                            except (KeyError, AttributeError) as e:
                                st.warning(f"자기자본 데이터 추출 실패: {str(e)}")

                            # ROE 계산 및 시그널 생성
                            if net_income is not None and total_equity is not None and total_equity != 0:
                                roe = (net_income / total_equity) * 100
                                
                                if roe > 15:
                                    roe_signal = 1
                                elif roe > 10:
                                    roe_signal = 0.5
                                elif roe > 5:
                                    roe_signal = 0
                                else:
                                    roe_signal = -1
                            else:
                                roe = None
                                roe_signal = 0
                                
                        except Exception as fin_error:
                            st.warning(f"재무제표 데이터 분석 중 오류 발생: {str(fin_error)}")
                            roe = None
                            roe_signal = 0
                        
                        # 기타 재무 정보
                        info = ticker.info
                        
                        # PER 분석
                        per = info.get('forwardPE')
                        per_signal = 0
                        if per and per > 0:
                            if per < 10:
                                per_signal = 1
                            elif per < 20:
                                per_signal = 0.5
                            elif per < 30:
                                per_signal = -0.5
                            else:
                                per_signal = -1
                        
                        # PBR 분석
                        pbr = info.get('priceToBook')
                        pbr_signal = 0
                        if pbr and pbr > 0:
                            if pbr < 1:
                                pbr_signal = 1
                            elif pbr < 3:
                                pbr_signal = 0.5
                            elif pbr < 5:
                                pbr_signal = -0.5
                            else:
                                pbr_signal = -1
                        
                        # 재무 종합 점수 계산
                        signals['fundamental'] = (roe_signal + per_signal + pbr_signal) / 3
                        
                    except Exception as e:
                        st.error(f"재무 데이터 분석 중 오류 발생: {str(e)}")
                        signals['fundamental'] = 0
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
                st.subheader("🏢 기업 정보")
                
                # 기업 기본 정보
                metrics = get_financial_metrics(symbol)
                if metrics:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📊 기업 기본 정보")
                        st.markdown(f"""
                        - 🏛️ 시장: {metrics['market_type']}
                        - 🏭 업종: {metrics['sector']}
                        - 🔍 세부업종: {metrics['industry']}
                        - 💰 시가총액: {format_number(metrics['marketCap'])}
                        - 📈 현재가: {format_number(metrics['currentPrice'])}
                        - 📊 평균거래량: {format_number(metrics['avgVolume'])}
                        """)
                    
                    with col2:
                        st.markdown("#### 📅 데이터 기준일")
                        st.markdown(f"""
                        - 📊 기준일: {metrics['dates']['financial']}
                        """)
                
                    # 주요 지표 표시
                    st.markdown("#### 📈 주요 투자 지표")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if metrics['per'] is not None:
                            st.metric("📊 PER", f"{metrics['per']:.2f}")
                            if metrics['per'] < 10:
                                st.success("저평가")
                            elif metrics['per'] < 20:
                                st.info("적정")
                            else:
                                st.warning("고평가")
                    
                    with col2:
                        if metrics['pbr'] is not None:
                            st.metric("📚 PBR", f"{metrics['pbr']:.2f}")
                            if metrics['pbr'] < 1:
                                st.success("저평가")
                            elif metrics['pbr'] < 3:
                                st.info("적정")
                            else:
                                st.warning("고평가")
                    
                    with col3:
                        if metrics['dividendYield'] is not None:
                            div_yield = metrics['dividendYield']
                            st.metric("💰 배당수익률", f"{div_yield:.2f}%")
                            if div_yield > 5:
                                st.success("높은 배당")
                            elif div_yield > 2:
                                st.info("적정 배당")
                            else:
                                st.warning("낮은 배당")

                    # 추가 지표 표시
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if metrics['eps'] is not None:
                            st.metric("💵 EPS", format_number(metrics['eps']))
                    
                    with col2:
                        if metrics['bps'] is not None:
                            st.metric("📖 BPS", format_number(metrics['bps']))

                    # 투자 유의사항
                    with st.expander("💡 투자 지표 설명"):
                        st.markdown("""
                        ### 주요 투자 지표 해석 가이드
                        
                        #### 1️⃣ PER (주가수익비율)
                        - 10배 미만: 저평가 가능성
                        - 10~20배: 적정 수준
                        - 20배 초과: 고평가 가능성
                        
                        #### 2️⃣ PBR (주가순자산비율)
                        - 1배 미만: 청산가치 이하
                        - 1~3배: 적정 수준
                        - 3배 초과: 고평가 가능성
                        
                        #### 3️⃣ 배당수익률
                        - 5% 초과: 고배당 주식
                        - 2~5%: 적정 배당
                        - 2% 미만: 저배당
                        
                        #### 4️⃣ EPS (주당순이익)
                        - 기업이 1주당 창출하는 순이익
                        - 높을수록 수익성이 좋음
                        
                        #### 5️⃣ BPS (주당순자산가치)
                        - 기업이 1주당 보유한 순자산
                        - PBR 계산의 기준이 됨
                        
                        ### 💡 투자 시 고려사항
                        1. 단일 지표가 아닌 여러 지표를 종합적으로 분석
                        2. 동일 업종 내 다른 기업들과 비교 분석 필요
                        3. 과거 추세와 현재 지표 변화 방향성 고려
                        4. 기업의 성장단계와 산업 특성 반영
                        5. 시장별 특성과 기업 규모 고려
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

