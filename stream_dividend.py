# 필요한 패키지 임포트 
import pandas as pd
import yfinance as yf  # FinanceDataReader 대신 yfinance 사용
import numpy_financial as num_finance
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate
import os
import streamlit as st
import numpy as np

class DivAnalysis():
    """배당 분석을 위한 클래스"""

    def __init__(self, ticker: str, data_feed: bool = False, d_return: float = 0.05, 
                 years: int = 10, growth_year_pick: int = 7, plot: bool = True, 
                 save: bool = True):
        """
        배당 분석 클래스 초기화
        
        Args:
            ticker (str): 종목 코드
            data_feed (bool): 데이터 피드 여부
            d_return (float): 요구 수익률(할인율)
            years (int): 예측 기간
            growth_year_pick (int): 성장률 계산을 위한 과거 기간
            plot (bool): 차트 표시 여부
            save (bool): 저장 여부
        """
        if not isinstance(ticker, str):
            raise TypeError("ticker는 문자열이어야 합니다.")
        if d_return <= 0:
            raise ValueError("d_return은 0보다 커야 합니다.")
        if years <= 0:
            raise ValueError("years는 0보다 커야 합니다.")
            
        self.ticker = ticker.strip()  # 종목 코드에서 공백 제거
        self.data_feed = data_feed
        self.d_return = d_return  # 요구 수익률(할인율)
        self.years = years  # Number of forecast years
        self.growth_year_choose = growth_year_pick  # Past years to calculate average growth rate
        self.plot = plot  # Flag to show plots 
        self.save = save
        
    def get_data(self):
        """Yahoo Finance에서 주식 데이터와 배당 정보를 가져옵니다."""
        try:
            # yfinance를 사용하여 주식 정보 가져오기
            stock = yf.Ticker(self.ticker)
            
            # 기본 정보 확인 및 검증
            try:
                info = stock.info
                if not info or len(info) == 0:
                    st.error(f"{self.ticker}에 대한 기본 정보를 가져올 수 없습니다.")
                    return False
                
                # 필수 필드 검증
                required_fields = ['longName', 'sector', 'industry']
                missing_fields = [field for field in required_fields if field not in info]
                if missing_fields:
                    st.warning(f"일부 기본 정보가 누락되었습니다: {', '.join(missing_fields)}")
                
            except Exception as e:
                st.error(f"기본 정보 조회 실패: {str(e)}")
                st.info("잠시 후 다시 시도해주세요.")
                return False
            
            # 주가 데이터 가져오기 (최적화된 방식)
            try:
                # 먼저 최근 2년 데이터로 시도
                price_data = stock.history(period="2y", interval="1d")
                if price_data.empty or 'Close' not in price_data.columns:
                    # 1년 데이터로 다시 시도
                    price_data = stock.history(period="1y", interval="1d")
                    if price_data.empty or 'Close' not in price_data.columns:
                        st.error(f"{self.ticker}에 대한 주가 데이터가 없습니다.")
                        return False
                    st.warning("2년치 데이터를 가져올 수 없어 1년 데이터를 사용합니다.")
                
                # 데이터 품질 검사 및 전처리
                if price_data['Close'].isnull().sum() > 0:
                    st.warning("일부 종가 데이터가 누락되었습니다. 보간법을 적용합니다.")
                    price_data['Close'] = price_data['Close'].interpolate(method='linear')
                
            except Exception as e:
                st.error(f"주가 데이터 조회 실패: {str(e)}")
                st.info("잠시 후 다시 시도해주세요.")
                return False
            
            # 배당 데이터 가져오기 (개선된 방식)
            try:
                # 먼저 actions로 시도 (더 포괄적인 데이터)
                actions = stock.actions
                if not actions.empty and 'Dividends' in actions.columns:
                    dividends_data = actions[['Dividends']].copy()
                else:
                    # 기존 방식으로 fallback
                    dividends = stock.dividends
                    if len(dividends) > 0:
                        dividends_data = dividends.to_frame(name='Dividends')
                    else:
                        st.warning(f"{self.ticker}의 배당 데이터가 없습니다.")
                        # 빈 배당 데이터 생성 (날짜 인덱스 사용)
                        date_range = pd.date_range(
                            start=price_data.index.min(),
                            end=price_data.index.max(),
                            freq='D'
                        )
                        dividends_data = pd.DataFrame(
                            index=date_range,
                            columns=['Dividends'],
                            data=0
                        )

                # 인덱스가 DatetimeIndex가 아닌 경우 변환
                if not isinstance(dividends_data.index, pd.DatetimeIndex):
                    if isinstance(dividends_data.index[0], (str, pd.Timestamp)):
                        dividends_data.index = pd.to_datetime(dividends_data.index)
                    else:
                        dividends_data = dividends_data.reset_index()
                        date_col = 'Date' if 'Date' in dividends_data.columns else 'index'
                        dividends_data[date_col] = pd.to_datetime(dividends_data[date_col])
                        dividends_data.set_index(date_col, inplace=True)

                # 배당 데이터 리샘플링 전 인덱스 타입 확인
                if not isinstance(dividends_data.index, pd.DatetimeIndex):
                    raise ValueError("배당 데이터의 인덱스가 DatetimeIndex가 아닙니다.")

                # 배당 데이터 리샘플링 및 정리
                dividends_data = dividends_data.resample('D').ffill()

                # 가격 데이터의 인덱스를 datetime으로 변환
                price_data.index = pd.to_datetime(price_data.index)
                
                # 배당 데이터를 가격 데이터의 인덱스에 맞춰 조정
                dividends_data = dividends_data.reindex(
                    price_data.index,
                    method='ffill'
                ).fillna(0)

                # 최종적으로 date 형식으로 변환 (계산 완료 후)
                price_data.index = price_data.index.date
                dividends_data.index = dividends_data.index.date

            except Exception as e:
                st.error(f"배당 데이터 처리 중 오류 발생: {str(e)}")
                # 오류 발생 시 빈 배당 데이터 생성
                date_range = pd.date_range(
                    start=price_data.index.min(),
                    end=price_data.index.max(),
                    freq='D'
                )
                dividends_data = pd.DataFrame(
                    index=date_range,
                    columns=['Dividends'],
                    data=0
                )
            
            # 최종 데이터 검증
            if price_data.empty or dividends_data.empty:
                st.error("데이터 검증 실패: 가격 또는 배당 데이터가 비어있습니다.")
                return False
            
            if 'Close' not in price_data.columns:
                st.error("필수 데이터(종가) 누락")
                return False
            
            if 'Dividends' not in dividends_data.columns:
                st.error("필수 데이터(배당금) 누락")
                return False
            
            # 데이터 저장
            self.price_data = price_data
            self.dividends_data = dividends_data
            
            # 추가 정보 저장
            self.company_info = {
                'name': info.get('longName', self.ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'country': info.get('country', 'N/A'),
                'website': info.get('website', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD')
            }
            
            return True
            
        except Exception as e:
            st.error(f"데이터 가져오기 실패: {str(e)}")
            st.info("잠시 후 다시 시도해주세요.")
            return False
    
    def statics_analysis(self):
            pass
        
    def cal_sum_of_fcf(self, recent_dividend, expected_growth_rate, cost, cost_discount_rate=0):
        # 리스트 컴프리헨션 사용으로 성능 개선
        future_dividend_list = []
        present_value_list = []
        
        future_dividend = recent_dividend * (1 + expected_growth_rate)
        for year in range(1, self.years + 1):
            if year == self.years:
                future_dividend += (cost * (1 - cost_discount_rate))
            
            future_dividend_list.append(future_dividend)
            present_value_list.append(future_dividend / (1 + self.d_return) ** year)
            future_dividend *= (1 + expected_growth_rate)
        
        return_dict = {
                    'sum_fcf' : sum(present_value_list),
                    'fcf' : future_dividend_list}
            
        return return_dict
    
    def cal_metrics(self, use_all=False, show_metrics=True):
        # 데이터가 없는 경우 조기 반환
        if not self.get_data():
            return None
        
        # 계산 결과를 캐시하여 재사용
        if not hasattr(self, '_metrics_cache'):
            self._metrics_cache = {}
        
        cache_key = f"{use_all}_{show_metrics}"
        if cache_key in self._metrics_cache:
            return self._metrics_cache[cache_key]
        
        """Get Data"""
        # 데이터 복사 및 인덱스를 datetime으로 변환
        dividend_data = self.dividends_data.copy()
        price_data = self.price_data.copy()
        
        # 인덱스를 datetime으로 변환
        dividend_data.index = pd.to_datetime(dividend_data.index)
        price_data.index = pd.to_datetime(price_data.index)
        
        # 연간 배당 데이터 계산 (일별 데이터를 연간으로 변환)
        yearly_data = dividend_data.resample('Y').sum()
        dividend_price_data = price_data.copy()
        
        # 배당 수익률 계산 (인덱스 맞추기)
        dividend_data_aligned = dividend_data.reindex(price_data.index, method='ffill').fillna(0)
        dividend_price_data['Dividend Yield'] = (
            dividend_data_aligned['Dividends'] / dividend_price_data['Close']
        ) * 100
        
        """Calculate Growth Rate"""
        # 연간 배당 성장률 계산
        yearly_data['growth rate'] = yearly_data['Dividends'].pct_change()
        
        # 성장률이 무한대인 경우 제외 (첫 배당 시작 시점)
        yearly_data.loc[yearly_data['growth rate'] == np.inf, 'growth rate'] = np.nan
        
        # 평균 성장률 계산 (중앙값 사용)
        valid_growth_rates = yearly_data['growth rate'].dropna()
        average_growth_rate = valid_growth_rates.median()
        
        # 최근 N년 평균 성장률
        recent_growth_rates = valid_growth_rates.tail(self.growth_year_choose)
        chosen_year_average_growth_rate = recent_growth_rates.median()
        
        # 성장률 설정
        expected_growth_rate = (
            average_growth_rate if use_all else chosen_year_average_growth_rate
        )
        
        """Calculate FCF and NPV & IRR"""
        # 최근 연간 배당금
        recent_dividend = yearly_data['Dividends'].iloc[-1]
        # 현재 주가
        cost = dividend_price_data['Close'].iloc[-1]
        
        # FCF 계산
        result = self.cal_sum_of_fcf(
            recent_dividend=recent_dividend,
            cost=cost,
            expected_growth_rate=expected_growth_rate
        )
        sum_of_discounted_future_cashflow = result['sum_fcf']
        future_dividend_list = result['fcf']
        
        # NPV & IRR 계산
        cashflow_for_npv = [-cost] + future_dividend_list
        npv = num_finance.npv(self.d_return, cashflow_for_npv)
        irr = num_finance.irr(cashflow_for_npv)
        
        """Calculate Safety Margin"""
        # 보수적 시나리오: 성장률 0%
        safety_margin_lv1 = self.cal_sum_of_fcf(
            recent_dividend=recent_dividend,
            cost=cost,
            expected_growth_rate=0
        )
        sum_of_lv1 = safety_margin_lv1['sum_fcf']
        
        # 현실적 시나리오: 최저 배당 & 성장률 0%
        min_dividend = yearly_data['Dividends'].min()
        safety_margin_lv2 = self.cal_sum_of_fcf(
            recent_dividend=min_dividend,
            cost=cost,
            expected_growth_rate=0
        )
        sum_of_lv2 = safety_margin_lv2['sum_fcf']
        
        # 비관적 시나리오: 최저 배당 & 성장률 0% & 주가 할인 20%
        safety_margin_lv3 = self.cal_sum_of_fcf(
            recent_dividend=min_dividend,
            cost=cost,
            expected_growth_rate=0,
            cost_discount_rate=0.2
        )
        sum_of_lv3 = safety_margin_lv3['sum_fcf']
        
        """Calculate Dividend Metrics"""
        # 배당 수익률 계산
        max_yield = dividend_price_data['Dividend Yield'].max()
        min_yield = dividend_price_data['Dividend Yield'].min()
        avg_yield = dividend_price_data['Dividend Yield'].mean()
        cur_yield = dividend_price_data['Dividend Yield'].iloc[-1]
        
        # 연속 배당 지급 계산
        yearly_data['dividend paid'] = yearly_data['Dividends'] > 0
        yearly_data['consecutive dividend count'] = (
            yearly_data['dividend paid']
            .groupby((yearly_data['dividend paid'] != yearly_data['dividend paid'].shift())
            .cumsum())
            .cumsum()
        )
        consecutive_dividend_count = (
            yearly_data.loc[yearly_data['dividend paid'], 'consecutive dividend count']
            .iloc[-1] if not yearly_data.empty else 0
        )
        
        # 연속 배당 성장 계산
        yearly_data['dividend growth'] = yearly_data['growth rate'] > 0
        yearly_data['consecutive growth'] = (
            yearly_data['dividend growth']
            .groupby((yearly_data['dividend growth'] != yearly_data['dividend growth'].shift())
            .cumsum())
            .cumsum()
        )
        consecutive_dividend_growth = (
            yearly_data.loc[yearly_data['dividend growth'], 'consecutive growth']
            .iloc[-1] if not yearly_data.empty else 0
        )
        
        """Create Metrics Dictionary"""
        metrics_dict = {
            'Average Growth Rate': f"{round(average_growth_rate * 100, 2)}%",
            f'Recent {self.growth_year_choose} Years AVG Growth Rate':
                f"{round(chosen_year_average_growth_rate * 100, 2)}%",
            'Average Dividend Yield': f"{round(avg_yield, 2)}%",
            'Max Dividend Yield': f"{round(max_yield, 2)}%",
            'Min Dividend Yield': f"{round(min_yield, 2)}%",
            'Current Dividend Yield': f"{round(cur_yield, 2)}%",
            'Consecutive Dividend Paid': f"{int(consecutive_dividend_count)} years",
            'Consecutive Dividend Growth': f"{int(consecutive_dividend_growth)} years",
            'Sum of FCF': round(sum_of_discounted_future_cashflow, 2),
            'Safety Margin Lv1': round(sum_of_lv1, 2),
            'Safety Margin Lv2': round(sum_of_lv2, 2),
            'Safety Margin Lv3': round(sum_of_lv3, 2),
            'NPV': round(npv, 2),
            'IRR': f"{round(irr * 100, 2)}%"
        }
        
        """Plotting"""
        # 차트 데이터 준비
        yield_price_x = dividend_price_data.index
        yield_price_y = dividend_price_data['Close']
        yield_price_y2 = dividend_price_data['Dividends']
        
        yearly_dividend_history_x = yearly_data.index
        yearly_growth_history_x = yearly_data.index
        yearly_dividend_history_y = yearly_data['Dividends']
        yearly_growth_history_y = yearly_data['growth rate']
        
        sum_fcf = round(sum_of_discounted_future_cashflow, 2)
        price = cost
        
        safety_margin_lv1 = sum_of_lv1
        safety_margin_lv2 = sum_of_lv2
        safety_margin_lv3 = sum_of_lv3
        
        irr = round(irr * 100, 2)
        demanded_return = round(self.d_return * 100, 2)
        
        # 차트 생성
        specs = [[{"secondary_y": True, "colspan": 2}, None],
                [{}, {}],
                [{}, {}]]
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Price & Dividend Yield', "Dividend History", 
                          "Growth History", "Sum of FCF vs Current Price", 
                          "IRR vs Demanded Return"),
            column_widths=[0.7, 0.7],
            row_heights=[0.6, 0.6, 0.4],
            specs=specs
        )
        
        # 차트 추가 (날짜 형식 변환 추가)
        x_dates = [pd.to_datetime(d) for d in yield_price_x]
        
        fig.add_trace(go.Scatter(x=x_dates, y=yield_price_y, 
                               mode='lines', name='Price'),
                     row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=x_dates, y=yield_price_y2, 
                               mode='lines', name='Dividend Yield'),
                     row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=x_dates, 
                               y=[safety_margin_lv1] * len(x_dates),
                               mode='lines', line=dict(color="green"), 
                               name='Safety Margin Lv1'),
                     row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=x_dates, 
                               y=[safety_margin_lv2] * len(x_dates),
                               mode='lines', line=dict(color="yellow"), 
                               name='Safety Margin Lv2'),
                     row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=x_dates, 
                               y=[safety_margin_lv3] * len(x_dates),
                               mode='lines', line=dict(color="red"), 
                               name='Safety Margin Lv3'),
                     row=1, col=1, secondary_y=False)
        
        # 연간 데이터 차트
        yearly_x_dates = [pd.to_datetime(d) for d in yearly_dividend_history_x]
        
        fig.add_trace(go.Bar(x=yearly_x_dates, 
                           y=yearly_dividend_history_y, 
                           name='Dividend History'),
                     row=2, col=1)
        fig.add_trace(go.Bar(x=yearly_x_dates, 
                           y=yearly_growth_history_y, 
                           name='Growth History'),
                     row=2, col=2)
        
        # 요약 차트
        fig.add_trace(go.Bar(x=['Sum of FCF'], y=[sum_fcf], 
                           name='FCF Sum'),
                     row=3, col=1)
        fig.add_trace(go.Bar(x=["Price"], y=[price], 
                           name="Current Price"),
                     row=3, col=1)
        fig.add_trace(go.Bar(x=["IRR"], y=[irr], 
                           name="IRR"),
                     row=3, col=2)
        fig.add_trace(go.Bar(x=["Demanded Return"], 
                           y=[demanded_return], 
                           name="Demanded Return"),
                     row=3, col=2)
        
        # 차트 레이아웃 업데이트
        fig.update_layout(
            width=1000, 
            height=900, 
            title_text="Dividend Info", 
            template="seaborn", 
            bargap=0.01
        )
        
        # x축 날짜 형식 설정
        fig.update_xaxes(tickformat="%Y-%m-%d", row=1, col=1)
        fig.update_xaxes(tickformat="%Y-%m-%d", row=2, col=1)
        fig.update_xaxes(tickformat="%Y-%m-%d", row=2, col=2)
        
        # 딕셔너리에 차트 추가
        metrics_dict['chart'] = fig
        
        self._metrics_cache[cache_key] = metrics_dict
        
        return metrics_dict
    
    def display_analysis(self):
        """수치값과 차트를 함께 보여주는 메소드"""
        # 분석 실행
        metrics = self.cal_metrics(show_metrics=False)
        
        # 수치값 표시
        metrics_df = pd.DataFrame(
            [(k, v) for k, v in metrics.items() if k != 'chart'],
            columns=['지표', '값']
        ).set_index('지표')
        
        print("\n=== 배당 분석 결과 ===")
        print(tabulate(metrics_df, headers='keys', tablefmt='psql'))
        
        # 차트 표시
        if self.plot and 'chart' in metrics:
            metrics['chart'].show()
        
        return metrics
    
    def display_streamlit_analysis(self):
        """Streamlit 화면에 분석 결과를 표시하는 메소드"""
        import streamlit as st
        
        # 로딩 상태 표시
        with st.spinner('데이터 분석 중...'):
            metrics = self.cal_metrics(show_metrics=False)
        
        if metrics is None:
            st.error("데이터 분석 중 오류가 발생했습니다.")
            return None
        
        # 사이드바에 기본 정보 표시
        with st.sidebar:
            st.markdown("### 📊 기본 정보")
            
            # 기본 지표 표시
            basic_metrics = {
                "현재 배당수익률": metrics['Current Dividend Yield'],
                "평균 배당수익률": metrics['Average Dividend Yield'],
                "연속 배당 지급": metrics['Consecutive Dividend Paid']
            }
            
            for key, value in basic_metrics.items():
                st.metric(label=key, value=value)
        
        # 메인 화면에 상세 분석 결과 표시
        st.markdown("## 📈 상세 분석 결과")
        
        # 성장성 분석
        with st.expander("🌱 성장성 분석", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 성장률")
                st.metric(
                    "평균 성장률", 
                    metrics['Average Growth Rate']
                )
                st.metric(
                    f"최근 {self.growth_year_choose}년 평균 성장률",
                    metrics[f'Recent {self.growth_year_choose} Years AVG Growth Rate']
                )
            
            with col2:
                st.markdown("### 배당 안정성")
                st.metric(
                    "연속 배당 성장",
                    metrics['Consecutive Dividend Growth']
                )
                st.metric(
                    "연속 배당 지급",
                    metrics['Consecutive Dividend Paid']
                )
        
        # 수익성 분석
        with st.expander("💰 수익성 분석", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 배당 수익률")
                st.metric(
                    "현재 배당수익률",
                    metrics['Current Dividend Yield']
                )
                st.metric(
                    "평균 배당수익률",
                    metrics['Average Dividend Yield']
                )
            
            with col2:
                st.markdown("### 배당 범위")
                st.metric(
                    "최대 배당수익률",
                    metrics['Max Dividend Yield']
                )
                st.metric(
                    "최소 배당수익률",
                    metrics['Min Dividend Yield']
                )
        
        # 가치평가 분석
        with st.expander("💵 가치평가 분석", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 현금흐름")
                st.metric("FCF 합계", metrics['Sum of FCF'])
                st.metric("NPV", metrics['NPV'])
            
            with col2:
                st.markdown("### 수익률")
                st.metric("IRR", metrics['IRR'])
                st.metric(
                    "요구 수익률",
                    f"{self.d_return * 100:.1f}%"
                )
        
        # 안전마진 분석
        with st.expander("🛡️ 안전마진 분석", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 보수적 시나리오")
                st.metric(
                    "Level 1 (성장률 0%)",
                    metrics['Safety Margin Lv1']
                )
                st.metric(
                    "Level 2 (최저 배당 & 성장률 0%)",
                    metrics['Safety Margin Lv2']
                )
            
            with col2:
                st.markdown("### 비관적 시나리오")
                st.metric(
                    "Level 3 (최저 배당 & 할인 20%)",
                    metrics['Safety Margin Lv3']
                )
        
        # 차트 표시
        if 'chart' in metrics:
            st.markdown("## 📊 차트 분석")
            st.plotly_chart(metrics['chart'], use_container_width=True)
        
        return metrics

def main():
    st.set_page_config(
        page_title="배당주 분석 대시보드",
        page_icon="📈",
        layout="wide"
    )
    
    # 메인 타이틀과 설명
    st.title("📊 배당주 분석 대시보드")
    
    # 기본 설명을 expander로 변경
    with st.expander("📌 대시보드 사용 가이드", expanded=False):
        st.markdown("""
        이 대시보드는 주식의 배당 관련 정보를 분석하여 투자 의사결정을 돕습니다.
        
        ### 🔍 사용 방법
        1. 사이드바에서 분석하고자 하는 주식의 티커를 입력하세요.
        2. 필요한 경우 상세 설정(요구 수익률, 분석 기간 등)을 조정하세요.
        3. '분석 시작' 버튼을 클릭하여 결과를 확인하세요.
        
        ### 💡 티커 입력 예시
        - 한국 주식: 종목코드.KS (예: 005930.KS)
        - 미국 주식: 심볼 (예: AAPL)
        """)
    
    # 지표 설명을 expander로 변경
    with st.expander("📊 배당 분석 지표 설명", expanded=False):
        st.markdown("""
        ### 🌱 성장 지표
        - **평균 성장률**: 기업의 배당금 성장 추세를 보여줍니다.
        - **최근 N년 평균 성장률**: 최근 성과를 중점적으로 반영합니다.
        - **연속 배당 성장/지급**: 배당의 안정성을 나타냅니다.
        
        ### 💰 수익률 지표
        - **현재/평균 배당수익률**: 투자 수익의 기대치를 보여줍니다.
        - **최대/최소 배당수익률**: 배당 수익의 변동 범위를 나타냅니다.
        
        ### 💵 가치평가 지표
        - **FCF 합계**: 기업의 실질적인 현금 창출 능력을 보여줍니다.
        - **NPV/IRR**: 투자의 수익성을 평가합니다.
        
        ### 🛡️ 안전마진 지표
        - **Level 1**: 보수적 가정의 안전마진
        - **Level 2**: 현실적 가정의 안전마진
        - **Level 3**: 비관적 가정의 안전마진
        """)
    
    # 투자 전략 설명을 expander로 변경
    with st.expander("📈 투자 전략 가이드", expanded=False):
        st.markdown("""
        ### 💡 효과적인 배당주 투자 전략
        
        1. **성장성 확인**
           - 평균 성장률과 최근 성장률 비교
           - 연속 배당 기록 확인
        
        2. **수익성 평가**
           - 현재 배당수익률과 과거 평균 비교
           - FCF와 배당금 지급 능력 분석
        
        3. **안전성 검토**
           - 안전마진 레벨별 분석
           - NPV와 IRR을 통한 투자 가치 평가
        
        4. **매수 시점 결정**
           - 안전마진 대비 현재 주가 비교
           - 배당수익률 트렌드 분석
        """)
    
    # 사이드바 구성
    with st.sidebar:
        st.header("⚙️ 분석 설정")
        
        # 티커 입력
        ticker = st.text_input(
            "티커 심볼을 입력하세요",
            value="005930.KS",
            help="- 한국 주식: 종목코드.KS (예: 005930.KS)\n- 미국 주식: 심볼 (예: AAPL)"
        )
        
        # 분석 옵션
        with st.expander("🎯 상세 설정", expanded=False):
            d_return = st.slider(
                "요구 수익률 (%)",
                min_value=1,
                max_value=20,
                value=5,
                help="투자에 대한 기대 수익률을 설정합니다."
            )
            
            years = st.slider(
                "분석 기간 (년)",
                min_value=1,
                max_value=20,
                value=10,
                help="과거 데이터 분석 기간을 설정합니다."
            )
            
            growth_years = st.slider(
                "성장률 계산 기간 (년)",
                min_value=1,
                max_value=10,
                value=7,
                help="평균 성장률을 계산할 기간을 설정합니다."
            )
        
        # 분석 시작 버튼
        start_analysis = st.button("📊 분석 시작", use_container_width=True)
    
    def analyze_stock():
        """주식 분석을 실행하는 함수"""
        try:
            with st.spinner('데이터를 분석중입니다...'):
                # 기본 정보 가져오기
                stock = yf.Ticker(ticker)
                info = stock.info
                
                if not info:
                    st.error("종목 정보를 가져올 수 없습니다.")
                    return
                    
                if info.get('dividendYield') is None:
                    st.warning("배당금 정보가 없습니다.")
                    return
                
                # 기업 개요
                st.markdown("## 📈 기업 개요")
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown("### 기업 정보")
                    st.write(f"**종목명**: {info.get('longName', '-')}")
                    st.write(f"**섹터**: {info.get('sector', '-')}")
                    st.write(f"**산업**: {info.get('industry', '-')}")
                    st.write(f"**국가**: {info.get('country', '-')}")
                
                with col2:
                    st.markdown("### 시장 정보")
                    st.write(f"**현재가**: ${info.get('currentPrice', '-'):,.2f}")
                    st.write(f"**시가총액**: ${info.get('marketCap', 0):,.0f}")
                    st.write(f"**52주 최고**: ${info.get('fiftyTwoWeekHigh', '-'):,.2f}")
                    st.write(f"**52주 최저**: ${info.get('fiftyTwoWeekLow', '-'):,.2f}")
                
                with col3:
                    st.markdown("### 배당 정보")
                    annual_dividend = (
                        info.get('dividendYield', 0) * 
                        info.get('previousClose', 0) / 100
                    )
                    current_price = info.get('previousClose', 0)
                    dividend_yield = (
                        (annual_dividend / current_price) * 100 
                        if current_price > 0 else 0
                    )
                    
                    st.metric("연간 배당금", f"${annual_dividend:.2f}")
                    st.metric("배당률", f"{dividend_yield:.2f}%")
                
                # 주가 히스토리
                with st.expander("📊 주가 히스토리", expanded=False):
                    history_df = stock.history(period="max")
                    if not history_df.empty and 'Close' in history_df.columns:
                        st.dataframe(history_df.tail(10))
                        st.line_chart(history_df['Close'])
                    else:
                        st.warning("히스토리 데이터를 가져올 수 없습니다.")
                
                # 상세 분석 실행
                analysis = DivAnalysis(
                    ticker=ticker,
                    d_return=d_return/100,
                    years=years,
                    growth_year_pick=growth_years
                )
                
                # 상세 분석 결과 표시
                analysis.display_streamlit_analysis()
                
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
            st.info("올바른 티커를 입력했는지 확인해주세요.")
            st.info("예시) 한국 주식: 005930.KS, 미국 주식: AAPL")
    
    # 분석 시작 버튼을 누르면 분석 실행
    if start_analysis:
        analyze_stock()

if __name__ == "__main__":
    main()
