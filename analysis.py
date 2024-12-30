import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import wraps
import requests
from requests.exceptions import RequestException

def handle_cors_headers(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return {
                'data': result,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                }
            }
        except Exception as e:
            return {
                'error': str(e),
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                }
            }
    return wrapper

@handle_cors_headers
def perform_dividend_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        if dividends.empty:
            return "배당금 정보가 없습니다."
        
        recent_dividends = dividends.tail(4)
        dividend_info = (f"최근 배당금 내역:\n"
                        f"{recent_dividends.to_string()}\n"
                        f"연간 배당률: {(dividends[-4:].sum() / stock.info['regularMarketPrice'] * 100):.2f}%")
        return dividend_info
    except Exception as e:
        return f"분석 중 오류 발생: {str(e)}"

@handle_cors_headers
def perform_portfolio_optimization(tickers):
    try:
        tickers = [ticker.strip() for ticker in tickers.split(',')]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        portfolio_data = pd.DataFrame()
        for ticker in tickers:
            stock = yf.download(ticker, start=start_date, end=end_date)
            portfolio_data[ticker] = stock['Adj Close'].pct_change()
        
        returns = portfolio_data.mean() * 252
        risk = portfolio_data.cov() * 252
        
        result = (f"포트폴리오 분석 결과:\n"
                 f"연간 수익률:\n{returns.to_string()}\n"
                 f"위험도(변동성):\n{np.sqrt(np.diag(risk)).to_string()}")
        return result
    except Exception as e:
        return f"분석 중 오류 발생: {str(e)}"

@handle_cors_headers
def perform_quantum_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1y')
        
        # 간단한 퀀텀 지표 계산 (예시)
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        hist['MA60'] = hist['Close'].rolling(window=60).mean()
        
        last_price = hist['Close'][-1]
        ma20 = hist['MA20'][-1]
        ma60 = hist['MA60'][-1]
        
        signal = "매수" if ma20 > ma60 else "매도"
        
        analysis = (f"퀀텀 분석 결과:\n"
                   f"현재가격: {last_price:.2f}\n"
                   f"20일 이동평균: {ma20:.2f}\n"
                   f"60일 이동평균: {ma60:.2f}\n"
                   f"신호: {signal}")
        return analysis
    except Exception as e:
        return f"분석 중 오류 발생: {str(e)}"

@handle_cors_headers
def perform_volatility_analysis(ticker):
    """변동성 분석으로 수정된 함수"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1mo')
        
        # 변동성 지표 계산
        returns = hist['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)  # 연간 변동성
        beta = calculate_beta(returns, '^GSPC')  # 시장 대비 베타
        
        analysis = (f"변동성 분석 결과:\n"
                   f"연간 변동성: {volatility:.2%}\n"
                   f"베타 계수: {beta:.2f}\n"
                   f"리스크 수준: {'높음' if volatility > 0.3 else '보통' if volatility > 0.15 else '낮음'}")
        return analysis
    except Exception as e:
        return f"분석 중 오류 발생: {str(e)}"

@handle_cors_headers
def perform_technical_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='6mo')
        
        # 기술적 지표 계산
        hist['RSI'] = calculate_rsi(hist['Close'])
        hist['MACD'], hist['Signal'] = calculate_macd(hist['Close'])
        
        last_close = hist['Close'][-1]
        last_rsi = hist['RSI'][-1]
        last_macd = hist['MACD'][-1]
        last_signal = hist['Signal'][-1]
        
        analysis = (f"기술적 분석 결과:\n"
                   f"종가: {last_close:.2f}\n"
                   f"RSI: {last_rsi:.2f}\n"
                   f"MACD: {last_macd:.2f}\n"
                   f"Signal: {last_signal:.2f}\n"
                   f"매매신호: {'매수' if last_macd > last_signal else '매도'}")
        return analysis
    except Exception as e:
        return f"분석 중 오류 발생: {str(e)}"

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def safe_request(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response
    except RequestException as e:
        raise Exception(f"요청 실패: {str(e)}")
