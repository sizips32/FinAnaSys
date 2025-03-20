# 필요한 패키지 임포트 
import pandas as pd
import yfinance as yf  # FinanceDataReader 대신 yfinance 사용
import numpy_financial as num_finance
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate
import os
import streamlit as st

class DivAnalysis():

    def __init__(self, ticker: str, data_feed: bool = False, d_return: float = 0.05, 
                 years: int = 10, growth_year_pick: int = 7, plot: bool = True, 
                 save: bool = True):
        # 입력값 검증 추가
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
        
    def get_data(self):
        """Yahoo Finance에서 주식 데이터와 배당 정보를 가져옵니다."""
        try:
            # yfinance를 사용하여 주식 정보 가져오기
            stock = yf.Ticker(self.ticker)
            
            # 기본 정보 확인
            try:
                info = stock.info
                if not info or len(info) == 0:
                    st.error(f"{self.ticker}에 대한 기본 정보를 가져올 수 없습니다.")
                    return False
            except Exception as e:
                st.error(f"기본 정보 조회 실패: {str(e)}")
                st.info("잠시 후 다시 시도해주세요.")
                return False
            
            # 주가 데이터 가져오기
            try:
                # 먼저 최근 1년 데이터로 시도
                price_data = stock.history(period="1y")
                if price_data.empty:
                    # 전체 기간으로 다시 시도
                    price_data = stock.history(period="max")
                    if price_data.empty:
                        st.error(f"{self.ticker}에 대한 주가 데이터가 없습니다.")
                        return False
                    st.warning("1년치 데이터를 가져올 수 없어 전체 기간의 데이터를 사용합니다.")
            except Exception as e:
                st.error(f"주가 데이터 조회 실패: {str(e)}")
                st.info("잠시 후 다시 시도해주세요.")
                return False
            
            # 배당 데이터 가져오기
            try:
                dividends = stock.dividends
                if len(dividends) > 0:
                    dividends_data = dividends.to_frame(name='Dividends')
                    # 배당 데이터를 일별 데이터로 리샘플링
                    dividends_data = dividends_data.resample('D').ffill()
                    # price_data의 인덱스에 맞추어 배당 데이터 정리
                    dividends_data = dividends_data.reindex(price_data.index, method='ffill')
                    dividends_data = dividends_data.fillna(0)  # NaN 값을 0으로 채우기
                else:
                    st.warning(f"{self.ticker}의 배당 데이터가 없습니다.")
                    # 배당 데이터가 없는 경우 0으로 채운 데이터프레임 생성
                    dividends_data = pd.DataFrame(
                        index=price_data.index,
                        columns=['Dividends'],
                        data=0
                    )
            except Exception as e:
                st.error(f"배당 데이터 조회 실패: {str(e)}")
                st.info("잠시 후 다시 시도해주세요.")
                return False
            
            # 데이터 검증
            if price_data.empty or dividends_data.empty:
                st.error("데이터 검증 실패: 가격 또는 배당 데이터가 비어있습니다.")
                return False
                
            # 데이터 전처리
            self.price_data = price_data
            self.dividends_data = dividends_data
            
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
        data_dict = self.get_data()
        if data_dict is None:
            return None
        
        # 계산 결과를 캐시하여 재사용
        if not hasattr(self, '_metrics_cache'):
            self._metrics_cache = {}
        
        cache_key = f"{use_all}_{show_metrics}"
        if cache_key in self._metrics_cache:
            return self._metrics_cache[cache_key]
        
        """Get Data"""
        yearly_data = self.dividends_data
        dividend_price_data = self.price_data
        
        """Calculate FCF and NPV & IRR"""
        yearly_data['growth rate'] = yearly_data['Dividends'].pct_change() # consecutive growth count 및 plotting 에서 필요하기 때문에 열을 생성해줘야함
        average_growth_rate = yearly_data['growth rate'].median()
        chosen_year_average_growth_rate = yearly_data['growth rate'].iloc[-self.growth_year_choose:].median()        
        
        # 성장률 설정
        expected_growth_rate = average_growth_rate if use_all else chosen_year_average_growth_rate
        
        # 최근 배당금과 주가 가져오기 
        recent_dividend = yearly_data['Dividends'].iloc[-1]
        cost = dividend_price_data['Close'].iloc[-1]
        
        # FCF 계산 
        result = self.cal_sum_of_fcf(recent_dividend=recent_dividend, cost=cost, expected_growth_rate=expected_growth_rate)
        sum_of_discounted_future_cashflow = result['sum_fcf']
        future_dividend_list = result['fcf']
        
        # NPV & IRR 계산
        future_dividend_list.insert(0, -cost) # NPV 계산을 위해 미래 cash flow 리스트의 첫 값에 -투입비용을 넣어줘야 한다.
        npv = num_finance.npv(self.d_return, future_dividend_list)
        irr = num_finance.irr(future_dividend_list)
        
        """Calculate Safety Marign"""
        
        saftey_margin_lv1 = self.cal_sum_of_fcf(recent_dividend=recent_dividend, cost=cost, expected_growth_rate=0)
        sum_of_lv1 = saftey_margin_lv1['sum_fcf'] # 배당성장률이 0%인 경우를 가정 
        safety_margin_lv2 = self.cal_sum_of_fcf(recent_dividend=yearly_data['Dividends'].min(), cost=cost, expected_growth_rate=0)
        sum_of_lv2 = safety_margin_lv2['sum_fcf'] # 시작 배당을 역사적으로 가장 낮은 배당 & 배당성장률 0% 
        safety_margin_lv3 = self.cal_sum_of_fcf(recent_dividend=yearly_data['Dividends'].min(), cost=cost, expected_growth_rate=0, cost_discount_rate=0.2)
        sum_of_lv3 = safety_margin_lv3['sum_fcf'] # 시작 배당을 min 배당 & 성장률 0% & 주가 회수를 discount
        
        """Calculate Other Metrics"""
        max_yield = dividend_price_data['Dividends'].max()
        min_yield = dividend_price_data['Dividends'].min()
        avg_yield = dividend_price_data['Dividends'].mean()
        cur_yield = dividend_price_data['Dividends'].iloc[-1]
        
        # Calcaulte Consecutive Dividend paid and growth
        yearly_data['dividend paid'] = yearly_data['Dividends'].notnull()
        yearly_data['consecutive dividend count'] = yearly_data['dividend paid'].groupby((yearly_data['dividend paid'] != yearly_data['dividend paid'].shift()).cumsum()).cumsum()
        consecutive_dividend_count = yearly_data['consecutive dividend count'].iloc[-1]
        
        yearly_data['dividend growth'] = yearly_data['growth rate'] > 0        
        yearly_data['consecutive growth'] = yearly_data['dividend growth'].groupby((yearly_data['dividend growth'] != yearly_data['dividend growth'].shift()).cumsum()).cumsum()
        consecutive_dividend_growth = yearly_data['consecutive growth'].iloc[-1]
        
        """Print Analysis"""
        # 결과값 데이터프레임 형식으로 만들기 
        metrics_dict = {
                    'Average Growth Rate': f"{round(average_growth_rate * 100, 2)}%",
                    f'Recent {self.growth_year_choose} Years AVG Growth Rate': f"{round(chosen_year_average_growth_rate * 100, 2)}%",
                    'Average Dividend Yield': f"{round(avg_yield * 100, 2)}%",
                    'Max Dividend Yield': f"{round(max_yield * 100, 2)}%",
                    'Min Dividend Yield': f"{round(min_yield * 100, 2)}%",
                    'Current Dividend Yield': f"{round(cur_yield * 100, 2)}%",
                    'Consecutive Dividend Paid': f"{consecutive_dividend_count} years",
                    'Consecutive Dividend Growth': f"{consecutive_dividend_growth} years",
                    'Sum of FCF': round(sum_of_discounted_future_cashflow, 2),
                    'Safety Margin Lv1': round(sum_of_lv1, 2),
                    'Safety Margin Lv2': round(sum_of_lv2, 2),
                    'Safety Margin Lv3': round(sum_of_lv3),
                    'NPV': round(npv, 2),
                    'IRR': f"{round(irr * 100, 2)}%"
        }
        # Convert dictionary to DataFrame
        metrics_value = pd.DataFrame(list(metrics_dict.items()), columns=['Metrics', 'Value']).set_index('Metrics')
        
        if show_metrics:
            print(tabulate(metrics_value, headers= 'keys', tablefmt='psql'))
        
        """Plotting"""
        # 차트 데이터 준비
        yield_price_x = dividend_price_data['Date']
        yield_price_y = dividend_price_data['Close']
        yield_price_y2 = dividend_price_data['Dividends']
        
        yearly_dividend_history_x = yearly_growth_history_x = yearly_data.index
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
        
        # 차트 추가
        fig.add_trace(go.Scatter(x=yield_price_x, y=yield_price_y, 
                               mode='lines', name='Price'),
                     row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=yield_price_x, y=yield_price_y2, 
                               mode='lines', name='Dividend Yield'),
                     row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=yield_price_x, 
                               y=[safety_margin_lv1] * len(yield_price_x),
                               mode='lines', line=dict(color="green"), 
                               name='Safety Margin Lv1'),
                     row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=yield_price_x, 
                               y=[safety_margin_lv2] * len(yield_price_x),
                               mode='lines', line=dict(color="yellow"), 
                               name='Safety Margin Lv2'),
                     row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=yield_price_x, 
                               y=[safety_margin_lv3] * len(yield_price_x),
                               mode='lines', line=dict(color="red"), 
                               name='Safety Margin Lv3'),
                     row=1, col=1, secondary_y=False)
        fig.add_trace(go.Bar(x=yearly_dividend_history_x, 
                             y=yearly_dividend_history_y, 
                             name='Dividend History'),
                     row=2, col=1)
        fig.add_trace(go.Bar(x=yearly_growth_history_x, 
                             y=yearly_growth_history_y, 
                             name='Growth History'),
                     row=2, col=2)
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
        
        fig.update_layout(
            width=1000, 
            height=900, 
            title_text="Dividend Info", 
            template="seaborn", 
            bargap=0.01
        )
        
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
        """Streamlit 사이드바에 지표를 표시하고 메인 화면에 차트를 보여주는 메소드"""
        import streamlit as st
        
        # 로딩 상태 표시
        with st.spinner('데이터 분석 중...'):
            metrics = self.cal_metrics(show_metrics=False)
        
        if metrics is None:
            st.error("데이터 분석 중 오류가 발생했습니다.")
            return None
        
        # 사이드바에 지표 표시
        st.sidebar.header("📊 배당 분석 지표")
        
        # 지표들을 카테고리별로 구분
        growth_metrics = {
            "평균 성장률": metrics['Average Growth Rate'],
            f"최근 {self.growth_year_choose}년 평균 성장률": metrics[f'Recent {self.growth_year_choose} Years AVG Growth Rate'],
            "연속 배당 성장": metrics['Consecutive Dividend Growth'],
            "연속 배당 지급": metrics['Consecutive Dividend Paid']
        }
        
        yield_metrics = {
            "현재 배당수익률": metrics['Current Dividend Yield'],
            "평균 배당수익률": metrics['Average Dividend Yield'],
            "최대 배당수익률": metrics['Max Dividend Yield'],
            "최소 배당수익률": metrics['Min Dividend Yield']
        }
        
        valuation_metrics = {
            "FCF 합계": metrics['Sum of FCF'],
            "NPV": metrics['NPV'],
            "IRR": metrics['IRR']
        }
        
        safety_metrics = {
            "안전마진 Level 1": metrics['Safety Margin Lv1'],
            "안전마진 Level 2": metrics['Safety Margin Lv2'],
            "안전마진 Level 3": metrics['Safety Margin Lv3']
        }
        
        # 각 카테고리별로 expander 생성
        with st.sidebar.expander("🌱 성장 지표", expanded=True):
            for key, value in growth_metrics.items():
                st.metric(label=key, value=value)
                
        with st.sidebar.expander("💰 수익률 지표", expanded=True):
            for key, value in yield_metrics.items():
                st.metric(label=key, value=value)
                
        with st.sidebar.expander("💵 가치평가 지표", expanded=True):
            for key, value in valuation_metrics.items():
                st.metric(label=key, value=value)
                
        with st.sidebar.expander("🛡️ 안전마진 지표", expanded=True):
            for key, value in safety_metrics.items():
                st.metric(label=key, value=value)
        
        # 메인 화면에 차트 표시
        if 'chart' in metrics:
            st.plotly_chart(metrics['chart'], use_container_width=True)
        
        return metrics

<<<<<<< HEAD
if __name__ == '__main__':
    st.title("배당주 분석 대시보드")
    
    # 사이드바에 티커 입력 추가
    ticker = st.sidebar.text_input(
        "티커 심볼을 입력하세요 (예: 005930.KS)",
        value="005930.KS",
        help="한국 주식의 경우 종목코드 뒤에 .KS를 붙여주세요"
    )
    
    # 분석 시작 버튼
    if st.sidebar.button("분석 시작"):
        try:
            analysis = DivAnalysis(ticker=ticker)
            analysis.display_streamlit_analysis()
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {str(e)}")

    # 배당 지표 설명 expander
    with st.expander("배당 지표 설명"):
        st.markdown(
            """
            #### 배당 분석 지표들은 주식 투자에서 기업의 배당 지급 능력과 성장 가능성을 평가하는 데 중요한 역할을 합니다. 각 지표의 의미와 주식 투자에 어떻게 적용할 수 있는지에 대해 설명하겠습니다.

            1. 평균 성장률 (Average Growth Rate)  
               - 의미: 기업의 배당금이 과거에 얼마나 성장했는지를 나타냅니다.  
               - 투자 적용: 평균 성장률이 높을수록 기업이 지속적으로 성장할 가능성이 높습니다.

            2. 최근 N년 평균 성장률 (Recent N Years AVG Growth Rate)  
               - 의미: 최근 N년 동안의 배당금 성장률로 기업의 최근 성과를 반영합니다.  
               - 투자 적용: 최근 성과가 평균보다 높다면, 기업이 최근에 더 좋은 성과를 내고 있음을 의미합니다.

            3. 현재 배당수익률 (Current Dividend Yield)  
               - 의미: 현재 주가 대비 배당금 비율입니다.  
               - 투자 적용: 배당수익률이 높을수록 안정적인 현금 흐름을 기대할 수 있습니다.

            4. 최대 배당수익률 (Max Dividend Yield)  
               - 의미: 과거 데이터에서 가장 높았던 배당수익률입니다.  
               - 투자 적용: 주가가 급락했을 때 높은 배당수익률을 매수 기회로 볼 수 있습니다.

            5. 최소 배당수익률 (Min Dividend Yield)  
               - 의미: 과거 데이터에서 가장 낮았던 배당수익률입니다.  
               - 투자 적용: 배당금 지급이 불안정할 수 있는 신호이므로 유의해야 합니다.

            예시로, A기업의 평균 성장률이 8%, 현재 배당수익률이 4%, 안전마진 Lv1이 30%라고 하면, 안정적인 배당금 지급과 성장 가능성을 가진 기업으로 평가할 수 있습니다.  

            1. FCF 합계 (Free Cash Flow)  
               - 의미: 운영에서 발생한 현금 흐름에서 자본 지출을 제외한 금액입니다.  
               - 투자 적용: FCF가 높으면 재무 건전성이 좋아, 배당금 지급이나 재투자 등에 유리합니다.

            2. NPV (Net Present Value)  
               - 의미: 미래 현금 흐름의 현재 가치 합에서 초기 투자 비용을 뺀 값입니다.  
               - 투자 적용: NPV가 양수이면 수익성이 있다고 보고, 음수이면 신중하게 접근해야 합니다.

            3. IRR (Internal Rate of Return)  
               - 의미: 투자 프로젝트의 수익률을 나타내는 지표입니다.  
               - 투자 적용: IRR이 요구 수익률보다 낮다면, 다른 투자 기회를 고려할 수 있습니다.

            4. 안전마진 (Safety Margin)  
               - 의미: 주가가 하락하더라도 배당금 지급이 가능한 정도를 평가하는 지표입니다.  
               - 투자 적용: 안전마진이 높은 기업은 추가 하락 시에도 버퍼가 있어 안정적입니다.

            종합적으로, FCF가 높고 NPV가 양수이며 IRR이 요구 수익률을 상회하고, 안전마진이 충분한 기업은 배당 투자에 유리합니다. 각 지표는 상호 보완적이며, 모든 지표를 함께 고려하여 투자 결정을 내리는 것이 바람직합니다.
            """
        )
=======
def run_dividend_analysis():
    st.title('배당금 분석 대시보드')
    
    with st.expander("📈 배당금 분석", expanded=True):
        try:
            stock = yf.Ticker(ticker)
            stock_info = stock.info
            
            if stock_info.get('dividendYield') is None:
                st.warning("배당금 정보가 없습니다.")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("기본 정보")
                st.write(f"종목명: {stock_info.get('longName', '-')}")
                st.write(f"섹터: {stock_info.get('sector', '-')}")
                st.write(f"산업: {stock_info.get('industry', '-')}")
            
            with col2:
                st.subheader("배당 정보")
                annual_dividend = stock_info['dividendYield'] * stock_info['previousClose'] / 100
                current_price = stock_info['previousClose']
                dividend_yield = (annual_dividend / current_price) * 100
                
                st.metric("연간 배당금", f"${annual_dividend:.2f}")
                st.metric("배당률", f"{dividend_yield:.2f}%")
            
            with st.expander("📊 주가 및 배당금 히스토리", expanded=False):
                st.dataframe(stock.history(period="max").tail(10))
                st.line_chart(stock.history(period="max")['Close'])
            
        except Exception as e:
            st.error(f"에러 발생: {str(e)}")

if __name__ == "__main__":
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
        if st.button("📊 분석 시작", use_container_width=True):
            try:
                with st.spinner('데이터를 분석중입니다...'):
                    analysis = DivAnalysis(
                        ticker=ticker,
                        d_return=d_return/100,
                        years=years,
                        growth_year_pick=growth_years
                    )
                    
                    # 메인 화면에 분석 결과 표시
                    with st.expander("📈 기본 정보", expanded=True):
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("기업 정보")
                            st.write(f"종목명: {info.get('longName', '-')}")
                            st.write(f"섹터: {info.get('sector', '-')}")
                            st.write(f"산업: {info.get('industry', '-')}")
                            st.write(f"국가: {info.get('country', '-')}")
                        
                        with col2:
                            st.subheader("시장 정보")
                            st.write(f"현재가: ${info.get('currentPrice', '-'):,.2f}")
                            st.write(f"시가총액: ${info.get('marketCap', 0):,.0f}")
                            st.write(f"52주 최고: ${info.get('fiftyTwoWeekHigh', '-'):,.2f}")
                            st.write(f"52주 최저: ${info.get('fiftyTwoWeekLow', '-'):,.2f}")
                    
                    # 상세 분석 결과 표시
                    analysis.display_streamlit_analysis()
                    
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
                st.info("올바른 티커를 입력했는지 확인해주세요.")
>>>>>>> 29dfec2 (업데이트)
