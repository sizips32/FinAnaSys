# 필요한 패키지 임포트 
import pandas as pd
import yfinance as yf
import numpy_financial as num_finance
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate
import os
import streamlit as st

class DivAnalysis():

    def __init__(self, ticker : str, data_feed : bool = False, d_return: float = 0.05, years: int =10, growth_year_pick: int = 7, plot: bool = True, save: bool = True):
        # 입력값 검증 추가
        if not isinstance(ticker, str):
            raise TypeError("ticker는 문자열이어야 합니다.")
        if d_return <= 0:
            raise ValueError("d_return은 0보다 커야 합니다.")
        if years <= 0:
            raise ValueError("years는 0보다 커야 합니다.")
            
        self.ticker = ticker
        self.data_feed = data_feed
        self.d_return = d_return # 요구 수익률(할인율)
        self.years = years # Number of forecast years
        self.growth_year_choose = growth_year_pick # Past years to calculate average growth rate
        self.plot = plot # Flag to show plots 
        
    def statics_analysis(self):
            pass
        
    def get_data(self):
        """Fetch and prepare data"""
        try:
            ticker_obj = yf.Ticker(self.ticker)
            dividends_data = pd.DataFrame(ticker_obj.dividends)
            
            if dividends_data.empty:
                raise ValueError(f"{self.ticker}에 대한 배당 데이터가 없습니다.")
            
            # 주가 데이터 가져오기
            price_data = yf.download(self.ticker, start=dividends_data.index[0])
            
            # 인덱스 레벨 확인 후 처리
            if isinstance(price_data.columns, pd.MultiIndex):
                price_data = price_data.droplevel(level=1, axis=1)
            elif price_data.columns.nlevels > 1:
                raise ValueError("Unexpected MultiIndex structure in price_data.")
            
            # Add year columns
            dividends_data['year'] = dividends_data.index.year
            yearly_dividends_sum = dividends_data.groupby('year')['Dividends'].sum().reset_index(name='dividend sum').set_index('year')
            yearly_dividends_count = dividends_data.groupby('year').size().reset_index(name='dividend count').set_index('year')
            
            most_common_value = yearly_dividends_count['dividend count'].value_counts().idxmax()
            most_common_ratio = yearly_dividends_count['dividend count'].value_counts(normalize=True)[most_common_value]
            
            # 비중이 80% 이상일 경우와 그렇지 않은 경우를 나눠서 처리 
            if most_common_ratio >=0.8:
                # 만약 데이터프레임의 마지막에 해당하는 년도에 받은 배당이 부족하다면
                if yearly_dividends_count['dividend count'].iloc[-1] < most_common_value:
                    # 처리 로직 1 : 배당 지급이 일정하다고 가정하고 부족한 배당을 예상
                    yearly_dividends_sum.iloc[-1, yearly_dividends_sum.columns.get_loc('dividend sum')] = dividends_data['Dividends'].iloc[-1] * most_common_value
                else: # 연배당인 경우
                    yearly_dividends_sum.loc[yearly_dividends_sum.index[-1], 'dividend sum'] = yearly_dividends_sum['dividend sum'].iloc[-2]
                    
            else:
                # 처리 로직 2: 배당 지급의 consistency가 적으므로 이전 년도의 배당으로 대체 
                yearly_dividends_sum.loc[yearly_dividends_sum.index[-1], 'dividend sum'] = yearly_dividends_sum['dividend sum'].iloc[-2]
        
            # Merge Dividend Sum data and Close price data for Dividend Yield & Price Combo chart
            data_close = price_data['Close'].reset_index()
            yearly_dividends = yearly_dividends_sum.reset_index()
            data_close['year'] = pd.to_datetime(data_close['Date']).dt.year
            merged_df = pd.merge(data_close, yearly_dividends, on='year', how='left')
            
            # 날짜별 배당수익률 열 추가하기
            merged_df['dividend yield'] = merged_df['dividend sum'] / merged_df['Close']
            
            merged_df.set_index('year', inplace=True)
            
            # 데이터 딕셔너리 만들기
            data_dict = {
                        'yearly_dividends' : yearly_dividends_sum,
                        'dividend_count' : yearly_dividends_count,
                        'price_dividend_combo' : merged_df, 
            }
            
            return data_dict
        except Exception as e:
            print(f"데이터 가져오기 실패: {str(e)}")
            return None
    
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
        yearly_data = data_dict['yearly_dividends']
        dividend_price_data = data_dict['price_dividend_combo']
        
        """Calculate FCF and NPV & IRR"""
        yearly_data['growth rate'] = yearly_data['dividend sum'].pct_change() # consecutive growth count 및 plotting 에서 필요하기 때문에 열을 생성해줘야함
        average_growth_rate = yearly_data['growth rate'].median()
        chosen_year_average_growth_rate = yearly_data['growth rate'].iloc[-self.growth_year_choose:].median()        
        
        # 성장률 설정
        expected_growth_rate = average_growth_rate if use_all else chosen_year_average_growth_rate
        
        # 최근 배당금과 주가 가져오기 
        recent_dividend = yearly_data['dividend sum'].iloc[-1]
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
        safety_margin_lv2 = self.cal_sum_of_fcf(recent_dividend=yearly_data['dividend sum'].min(), cost=cost, expected_growth_rate=0)
        sum_of_lv2 = safety_margin_lv2['sum_fcf'] # 시작 배당을 역사적으로 가장 낮은 배당 & 배당성장률 0% 
        safety_margin_lv3 = self.cal_sum_of_fcf(recent_dividend=yearly_data['dividend sum'].min(), cost=cost, expected_growth_rate=0, cost_discount_rate=0.2)
        sum_of_lv3 = safety_margin_lv3['sum_fcf'] # 시작 배당을 min 배당 & 성장률 0% & 주가 회수를 discount
        
        """Calculate Other Metrics"""
        max_yield = dividend_price_data['dividend yield'].max()
        min_yield = dividend_price_data['dividend yield'].min()
        avg_yield = dividend_price_data['dividend yield'].mean()
        cur_yield = dividend_price_data['dividend yield'].iloc[-1]
        
        # Calcaulte Consecutive Dividend paid and growth
        yearly_data['dividend paid'] = yearly_data['dividend sum'].notnull()
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
        yield_price_y2 = dividend_price_data['dividend yield']
        
        yearly_dividend_history_x = yearly_growth_history_x = yearly_data.index
        yearly_dividend_history_y = yearly_data['dividend sum']
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
