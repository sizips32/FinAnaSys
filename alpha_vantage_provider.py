"""Alpha Vantage 데이터 제공자 모듈입니다."""

import os
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
import logging
from dotenv import load_dotenv

# 로깅 설정
logger = logging.getLogger(__name__)

class AlphaVantageProvider:
    """Alpha Vantage API를 통해 주식 데이터를 제공하는 클래스"""
    
    def __init__(self):
        """Alpha Vantage Provider 초기화"""
        load_dotenv()  # .env 파일 로드
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY가 설정되지 않았습니다.")
            
        self.base_url = "https://www.alphavantage.co/query"
        self.cache_dir = "data_cache"
        self.calls_per_min = 5
        self.last_call_time = datetime.now()
        self.call_count = 0
        
        # 캐시 디렉토리 생성
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _rate_limit(self):
        """API 호출 속도 제한"""
        current_time = datetime.now()
        if (current_time - self.last_call_time).seconds < 60:
            if self.call_count >= self.calls_per_min:
                sleep_time = 60 - (current_time - self.last_call_time).seconds
                time.sleep(sleep_time)
                self.call_count = 0
                self.last_call_time = datetime.now()
        else:
            self.call_count = 0
            self.last_call_time = current_time
            
        self.call_count += 1
    
    def _get_cache_path(self, ticker, data_type):
        """캐시 파일 경로 반환"""
        return os.path.join(self.cache_dir, f"{ticker}_{data_type}.json")
    
    def _cache_data(self, ticker, data_type, data):
        """데이터 캐싱"""
        cache_path = self._get_cache_path(ticker, data_type)
        cache_data = {
            'cached_at': datetime.now().isoformat(),
            'data': data
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
    
    def _get_cached_data(self, ticker, data_type):
        """캐시된 데이터 조회"""
        cache_path = self._get_cache_path(ticker, data_type)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
                cached_at = datetime.fromisoformat(cached_data['cached_at'])
                # 캐시가 24시간 이내인 경우
                if (datetime.now() - cached_at).days < 1:
                    return cached_data['data']
        return None
    
    def get_stock_price(self, ticker, force_refresh=False):
        """주가 데이터 조회"""
        if not force_refresh:
            cached_data = self._get_cached_data(ticker, 'price')
            if cached_data:
                return pd.DataFrame(cached_data)
        
        self._rate_limit()
        try:
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': ticker,
                'outputsize': 'full',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                logger.error(f"주가 데이터를 가져올 수 없습니다: {data.get('Note', '알 수 없는 오류')}")
                return None
            
            daily_data = data["Time Series (Daily)"]
            price_data = []
            
            for date, values in daily_data.items():
                price_data.append({
                    'Date': date,
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': float(values['6. volume'])
                })
            
            df = pd.DataFrame(price_data)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            self._cache_data(ticker, 'price', price_data)
            return df
            
        except Exception as e:
            logger.error(f"주가 데이터 조회 실패: {str(e)}")
            return None
    
    def get_dividend_data(self, ticker, force_refresh=False):
        """배당 데이터 조회"""
        if not force_refresh:
            cached_data = self._get_cached_data(ticker, 'dividend')
            if cached_data:
                return pd.DataFrame(cached_data)
        
        self._rate_limit()
        try:
            params = {
                'function': 'TIME_SERIES_MONTHLY_ADJUSTED',
                'symbol': ticker,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if "Monthly Adjusted Time Series" not in data:
                logger.error(f"배당 데이터를 가져올 수 없습니다: {data.get('Note', '알 수 없는 오류')}")
                return None
            
            monthly_data = data["Monthly Adjusted Time Series"]
            dividend_data = []
            
            for date, values in monthly_data.items():
                dividend = float(values['7. dividend amount'])
                if dividend > 0:
                    dividend_data.append({
                        'Date': date,
                        'Dividends': dividend
                    })
            
            df = pd.DataFrame(dividend_data)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            self._cache_data(ticker, 'dividend', dividend_data)
            return df
            
        except Exception as e:
            logger.error(f"배당 데이터 조회 실패: {str(e)}")
            return None
    
    def get_company_info(self, ticker):
        """기업 정보 조회"""
        self._rate_limit()
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': ticker,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if not data or 'Symbol' not in data:
                logger.error(f"기업 정보를 가져올 수 없습니다: {data.get('Note', '알 수 없는 오류')}")
                return None
            
            return {
                'name': data.get('Name', ticker),
                'currency': 'USD',  # Alpha Vantage는 기본적으로 USD 사용
                'sector': data.get('Sector', 'N/A'),
                'industry': data.get('Industry', 'N/A'),
                'country': data.get('Country', 'N/A'),
                'market_cap': float(data.get('MarketCapitalization', 0)),
                'dividend_yield': float(data.get('DividendYield', 0)) * 100,
                'dividend_per_share': float(data.get('DividendPerShare', 0))
            }
            
        except Exception as e:
            logger.error(f"기업 정보 조회 실패: {str(e)}")
            return None 
