import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from pypfopt import (
    EfficientFrontier, risk_models, expected_returns, HRPOpt, CLA
)
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

warnings.simplefilter(action='ignore', category=FutureWarning)

# 페이지 설정
st.set_page_config(
    layout="wide",
    page_title="Advanced Portfolio Analytics",
    page_icon="📈"
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# CSS 스타일 적용
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
    }
    
    .stMarkdown {
        font-family: 'Noto Sans KR', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# 사이드바 설정
st.sidebar.title("포트폴리오 분석 설정 🛠️")

# 기본 입력 파라미터
tickers = st.sidebar.text_input(
    "종목 코드 입력 (쉼표로 구분) 📝",
    "005930, 035420, 051910, 035720"  # 기본 예시 종목 (삼성전자, NAVER, LG화학,카카오)
)
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.now().date())

# 최적화 설정
st.sidebar.markdown("### 포트폴리오 최적화 설정 ⚙️")
optimization_method = st.sidebar.selectbox(
    "최적화 방법 선택",
    [
        "최대 샤프 지수 (Maximum Sharpe Ratio) ⚡", 
        "최소 변동성 (Minimum Volatility) 🛡️", 
        "리스크 균형 (Risk Parity) ⚖️"
    ]
)

rebalance_period = st.sidebar.selectbox(
    "Rebalancing Frequency",
    ["Monthly", "Quarterly", "Semi-Annually", "Annually"],
    help="Select the rebalancing frequency for the portfolio."
)

# 사이드바 설정 부분에 추가
st.sidebar.markdown("### 동적 헤지 설정 🛡️")
hedge_params = st.sidebar.expander("헤지 파라미터 설정", expanded=False)

with hedge_params:
    # 기본 임계값
    base_threshold = st.slider(
        "기본 변동성 임계값 (%)",
        min_value=10.0,
        max_value=50.0,
        value=15.0,
        step=1.0,
        format="%.1f%%"
    )
    
    # 헤지 강도
    hedge_intensity = st.slider(
        "헤지 강도 조절",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="1.0 = 기본, 2.0 = 2배 강한 헤지"
    )
    
    # 최대 헤지 비율
    max_hedge_ratio = st.slider(
        "최대 헤지 비율",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="헤지 가능 최대 포트폴리오 비중"
    )

# 분석 실행 버튼
run_analysis = st.sidebar.button("Run Analysis", use_container_width=True)

# 탭 생성
tab1, tab2, tab3, tab4 = st.tabs([
    "포트폴리오 최적화 및 리스크 분석 📊", 
    "리밸런싱 시뮬레이션 🔄",
    "스트레스 테스트 🧪",
    "동적 헤지 전략 🛡️"
])

# 포트폴리오 최적화 결과 설명
method_descriptions = {
    "최대 샤프 지수 (Maximum Sharpe Ratio) ⚡": """
    ### 📈 최대 샤프 지수 전략
    **위험 대비 수익률을 최대화** 하는 포트폴리오 구성
    - 장점: 높은 위험 조정 수익률 기대
    - 단점: 특정 자산에 집중될 수 있음
    - 적합 상황: 시장 예측이 비교적 명확할 때
    """,
    
    "최소 변동성 (Minimum Volatility) 🛡️": """
    ### 📉 최소 변동성 전략 
    **포트폴리오 변동성을 최소화** 하는 안정적 구성
    - 장점: 시장 변동성에 덜 민감
    - 단점: 수익률 상대적으로 낮을 수 있음
    - 적합 상황: 시장 불확실성 높을 때
    """,
    
    "리스크 균형 (Risk Parity) ⚖️": """
    ### ⚖️ 리스크 균형 전략
    **모든 자산의 리스크 기여도를 균등하게** 분배
    - 장점: 우수한 리스크 분산 효과
    - 단점: 계산 복잡도 높음
    - 적합 상황: 자산 간 상관관계 낮을 때
    """
}

# 위험 지표 설명
risk_metric_explanation = """
### 📌 주요 위험 지표 설명
| 지표 | 설명 | 해석 기준 |
|------|------|------|
| **변동성** | 수익률의 표준편차(연율화) | 낮을수록 안정적 |
| **샤프 지수** | 위험 단위당 초과수익 | 높을수록 우수 |
| **소르티노** | 하방 변동성 고려한 위험조정 수익 | 높을수록 우수 |
| **MDD** | 최대 낙폭(역대 최고점 대비 최저점) | 낮을수록 안전 |
| **VaR 95%** | 95% 신뢰구간 최대 예상 손실 | 낮을수록 좋음 |
| **CVaR 95%** | VaR 초과시 평균 손실액 | 낮을수록 좋음 |
"""

# 스트레스 테스트 시나리오
stress_scenarios = {
    "시장 폭락 📉": -0.20,
    "경기 침체 💸": -0.30,
    "보통 하락 📉": -0.10,
    "테크 버블 💣": -0.25,
    "금융 위기 🏚️": -0.35
}

# 동적 헤지 전략 설명
hedge_strategy_explanation = """
### 🛡️ 동적 헤지 전략 원리
1. **실시간 변동성 모니터링**  
   → GARCH 모델로 일일 변동성 추정 📈
2. **헤지 비율 자동 조정**  
   → 변동성 임계치 초과시 헤지 강화 ⚖️
3. **리스크 노출 제한**  
   → 극단적 시장 상황에서 손실 최소화 🛑
"""

def fetch_data(tickers_list, start_date, end_date):
    """
    FinanceDataReader를 사용하여 주가 데이터를 가져오는 함수
    
    Parameters:
    tickers_list (list): 종목 코드 리스트
    start_date (datetime): 시작일
    end_date (datetime): 종료일
    
    Returns:
    tuple: (prices_df, returns_df) - 가격 데이터프레임과 수익률 데이터프레임
    """
    try:
        if not tickers_list or len(tickers_list) == 0:
            raise ValueError("종목 코드를 하나 이상 입력해주세요.")
            
        valid_tickers = []
        data_frames = []
        
        # KRX 종목 목록 가져오기
        krx = fdr.StockListing('KRX')
        
        # 컬럼명 확인 및 처리
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
            raise ValueError("주식 시장 데이터 형식이 올바르지 않습니다.")
        
        for ticker in tickers_list:
            t = ticker.strip().upper()
            try:
                # 한국 주식인지 확인
                is_korean = t in krx[symbol_column].values
                
                if is_korean:
                    # 한국 주식 데이터 가져오기
                    temp_data = fdr.DataReader(t, start_date, end_date)
                    company_name = krx[krx[symbol_column] == t][name_column].iloc[0]
                else:
                    # 미국 주식 데이터 가져오기
                    temp_data = fdr.DataReader(t, start_date, end_date)
                    company_name = t  # 미국 주식은 심볼을 회사명으로 사용
                
                if temp_data is not None and not temp_data.empty:
                    if 'Close' in temp_data.columns:
                        close_series = temp_data['Close'].rename(f"{company_name} ({t})")
                        data_frames.append(close_series)
                        valid_tickers.append(t)
                    else:
                        st.warning(f"{t}의 종가 데이터가 없습니다.")
                else:
                    st.warning(f"{t}의 데이터를 찾을 수 없습니다.")
                    
            except Exception as e:
                st.warning(f"{t} 데이터 다운로드 실패: {str(e)}")
        
        if not valid_tickers:
            raise ValueError("유효한 종목 코드가 없습니다.")
            
        data = pd.concat(data_frames, axis=1)
        data = data.ffill().bfill().dropna(how='all', axis=1)
        
        if data.empty:
            raise ValueError("처리 후 유효한 데이터가 없습니다.")
            
        returns = data.pct_change().dropna()
        
        if len(returns) < 252:
            st.warning("데이터가 부족합니다 (최소 1년 필요)")
            return None, None
            
        return data, returns
        
    except Exception as e:
        st.error(f"🚨 데이터 오류: {str(e)}")
        st.error("문제 해결 방법:\n1. 인터넷 연결 확인\n2. 종목 코드 확인\n3. 날짜 범위 조정")
        return None, None

def calculate_risk_metrics(returns, weights=None):
    """위험 지표 계산"""
    # Series인 경우 바로 처리
    if isinstance(returns, pd.Series):
        portfolio_returns = returns
    # DataFrame인 경우 가중치 적용
    else:
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)
        portfolio_returns = returns.dot(weights)
    
    metrics = {
        'Volatility': portfolio_returns.std() * np.sqrt(252),
        'Sharpe': calculate_sharpe_ratio(portfolio_returns),
        'Sortino': calculate_sortino_ratio(portfolio_returns),
        'MDD': calculate_mdd(portfolio_returns),
        'VaR_95': calculate_var(portfolio_returns),
        'CVaR_95': calculate_cvar(portfolio_returns)
    }
    return metrics

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """샤프 비율 계산"""
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """소르티노 비율 계산"""
    excess_returns = returns - risk_free_rate/252
    downside_returns = returns[returns < 0]
    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

def calculate_mdd(returns):
    """최대 낙폭(MDD) 계산"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_var(returns, confidence_level=0.95):
    """VaR(Value at Risk) 계산"""
    return np.percentile(returns, (1-confidence_level)*100)

def calculate_cvar(returns, confidence_level=0.95):
    """CVaR(Conditional Value at Risk) 계산"""
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def run_stress_test(returns, scenarios):
    """스트레스 테스트 실행"""
    results = {}
    for scenario_name, shock in scenarios.items():
        shocked_returns = returns * (1 + shock)
        metrics = calculate_risk_metrics(shocked_returns)
        results[scenario_name] = metrics
    return results

# 헤지 시뮬레이션 함수에 캐싱 추가
@st.cache_data
def prepare_hedge_data(returns):
    """헤지 계산에 필요한 기본 데이터 준비"""
    portfolio_returns = returns.mean(axis=1)
    window_size = 63
    return portfolio_returns, window_size

def simulate_dynamic_hedge(_portfolio_returns, window_size, params):
    """확장된 헤지 파라미터 적용"""
    hedge_ratio = []
    
    base_threshold = params['base_threshold']
    hedge_intensity = params['intensity']
    max_ratio = params['max_ratio']
    
    for i in range(window_size, len(_portfolio_returns)):
        # 변동성 예측
        train_data = _portfolio_returns.iloc[i-window_size:i]
        model = arch_model(train_data, vol='Garch', p=1, q=1, dist='normal')
        res = model.fit(update_freq=0, disp='off')
        forecast = res.forecast(horizon=21)
        predicted_vol = np.sqrt(forecast.variance.iloc[-1, -1]) * np.sqrt(252)
        
        # 헤지 비율 계산 확장
        if predicted_vol > base_threshold:
            excess_vol = predicted_vol - base_threshold
            ratio = min(max_ratio, (excess_vol / base_threshold) * hedge_intensity)
        else:
            ratio = 0.0
            
        hedge_ratio.append(ratio)
    
    return hedge_ratio

def fetch_exchange_rate():
    """달러/원 환율 정보 가져오기"""
    try:
        usd_krw = fdr.DataReader("USDKRW=X", start_date, end_date)['Close'].iloc[-1]
        return usd_krw
    except Exception as e:
        st.warning("환율 정보를 가져오는데 실패했습니다. 기본값 1,300원을 사용합니다.")
        return 1300.0

def run_rebalancing_simulation():
    # 리밸런싱 주기 매핑 로직 추가
    freq_mapping = {
        "Monthly": 'M',
        "Quarterly": 'Q',
        "Semi-Annually": '6M',
        "Annually": 'A'
    }
    
    try:
        rebalance_dates = pd.date_range(
            start=dates[0],
            end=dates[-1],
            freq=freq_mapping[rebalance_period]
        )
        
        # ... 기존 코드 ...
        
    except KeyError:
        st.error("잘못된 리밸런싱 주기 설정입니다")
    except Exception as e:
        st.error(f"리밸런싱 시뮬레이션 실패: {str(e)}")
        st.error("문제 발생 시 체크사항:\n"
                 "1. 분석 기간이 리밸런싱 주기보다 긴지 확인\n"
                 "2. 포트폴리오 가중치 합계 검증\n"
                 "3. 수익률 데이터 정확성 확인")

# 메인 실행 부분
if run_analysis:
    tickers_list = [t.strip().upper() for t in tickers.split(",")]
    data, returns = fetch_data(tickers_list, start_date, end_date)
    
    if data is not None and returns is not None:
        try:
            with tab1:
                st.header("Portfolio Optimization & Risk Analysis")
                
                # 기본 정보 표시
                st.subheader("Basic Portfolio Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    # 수익률 히트맵
                    corr_matrix = returns.corr()
                    fig_corr = plt.figure(figsize=(10, 6))
                    sns.heatmap(
                        corr_matrix, 
                        annot=True, 
                        cmap='coolwarm', 
                        center=0,
                        fmt='.2f'
                    )
                    plt.title("Asset Correlation")
                    st.pyplot(fig_corr)
                
                with col2:
                    # 개별 자산 성과
                    cumulative_returns = (1 + returns).cumprod()
                    st.line_chart(cumulative_returns)
                    st.caption("Cumulative Returns of Individual Assets")
                
                # 포트폴리오 최적화
                st.subheader(f"Portfolio Optimization Result ({optimization_method})")
                
                # 최적화 결과 저장
                global weights, performance
                if optimization_method == "최대 샤프 지수 (Maximum Sharpe Ratio) ⚡":
                    ef = EfficientFrontier(expected_returns.mean_historical_return(data), risk_models.sample_cov(data))
                    ef.max_sharpe()
                    weights = ef.clean_weights()
                    performance = ef.portfolio_performance()
                    method_description = method_descriptions["최대 샤프 지수 (Maximum Sharpe Ratio) ⚡"]
                elif optimization_method == "최소 변동성 (Minimum Volatility) 🛡️":
                    ef = EfficientFrontier(expected_returns.mean_historical_return(data), risk_models.sample_cov(data))
                    ef.min_volatility()
                    weights = ef.clean_weights()
                    performance = ef.portfolio_performance()
                    method_description = method_descriptions["최소 변동성 (Minimum Volatility) 🛡️"]
                elif optimization_method == "리스크 균형 (Risk Parity) ⚖️":
                    try:
                        # 공분산 행렬 양정치(positive definite) 검증
                        S = risk_models.fix_nonpositive_semidefinite(risk_models.sample_cov(data))
                        
                        # 데이터 전처리
                        clean_returns = returns.copy()
                        
                        # 무한값과 NaN 처리
                        clean_returns = clean_returns.replace([np.inf, -np.inf], np.nan)
                        clean_returns = clean_returns.fillna(method='ffill').fillna(method='bfill')
                        
                        # 극단값 처리 (1~99 percentile로 제한)
                        for col in clean_returns.columns:
                            upper = np.percentile(clean_returns[col].dropna(), 99)
                            lower = np.percentile(clean_returns[col].dropna(), 1)
                            clean_returns[col] = clean_returns[col].clip(lower, upper)
                        
                        # 남은 NaN 처리
                        if clean_returns.isnull().any().any():
                            clean_returns = clean_returns.fillna(clean_returns.mean())
                        
                        # 공분산 행렬 계산 및 검증
                        cov_matrix = clean_returns.cov()
                        if not np.isfinite(cov_matrix.values).all():
                            raise ValueError("공분산 행렬에 무한값이 포함되어 있습니다.")
                        
                        # 상관관계 행렬 검증
                        corr_matrix = clean_returns.corr()
                        if not np.isfinite(corr_matrix.values).all():
                            raise ValueError("상관관계 행렬에 무한값이 포함되어 있습니다.")
                        
                        # 변동성 검증
                        vols = clean_returns.std()
                        if not np.isfinite(vols).all() or (vols == 0).any():
                            raise ValueError("일부 자산의 변동성이 0이거나 무한값입니다.")
                        
                        # HRP 최적화 수행
                        hrp = HRPOpt(clean_returns)
                        raw_weights = hrp.optimize()
                        
                        # 가중치 변환 및 검증
                        if isinstance(raw_weights, dict):
                            weights = {k: float(v) for k, v in raw_weights.items()}
                        else:
                            weights = {asset: float(w) for asset, w in zip(returns.columns, raw_weights)}
                        
                        # 가중치 검증
                        if not all(np.isfinite(list(weights.values()))):
                            raise ValueError("최적화된 가중치에 무한값이 포함되어 있습니다.")
                        
                        # 가중치 합계 확인 및 정규화
                        total_weight = sum(weights.values())
                        if not (0.99 <= total_weight <= 1.01):
                            weights = {k: v/total_weight for k, v in weights.items()}
                        
                        # 포트폴리오 수익률 계산
                        weights_series = pd.Series(weights)
                        portfolio_returns = returns.dot(weights_series)
                        
                        # 성과 지표 계산
                        annual_return = portfolio_returns.mean() * 252
                        annual_vol = portfolio_returns.std() * np.sqrt(252)
                        sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
                        
                        performance = (annual_return, annual_vol, sharpe_ratio)
                        
                        method_description = method_descriptions["리스크 균형 (Risk Parity) ⚖️"]
                    except Exception as e:
                        st.error(f"Risk Parity Optimization Error: {str(e)}")
                        st.error("Troubleshooting:\n1. Increase analysis period (recommended min 1 year)\n"
                                 "2. Remove highly correlated assets\n"
                                 "3. Choose another optimization method")
                        # 최적화 실패 시 폴백(fallback) 전략
                        try:
                            cla = CLA(expected_returns.mean_historical_return(data), risk_models.sample_cov(data))
                            raw_weights = cla.max_sharpe()
                        except Exception as e:
                            st.warning("CLA Optimization failed, using alternative method...")
                            raw_weights = np.ones(len(expected_returns.mean_historical_return(data))) / len(expected_returns.mean_historical_return(data))
                        
                        # 가중치 합계 확인 및 정규화
                        total_weight = sum(raw_weights)
                        if not (0.99 <= total_weight <= 1.01):
                            raw_weights = {k: v/total_weight for k, v in raw_weights.items()}
                        
                        # 포트폴리오 수익률 계산
                        weights_series = pd.Series(raw_weights)
                        portfolio_returns = returns.dot(weights_series)
                        
                        # 성과 지표 계산
                        annual_return = portfolio_returns.mean() * 252
                        annual_vol = portfolio_returns.std() * np.sqrt(252)
                        sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
                        
                        performance = (annual_return, annual_vol, sharpe_ratio)
                
                # 가중치 검증 및 표시
                total_weight = sum(weights.values())
                if not (0.99 <= total_weight <= 1.01):
                    st.warning(
                        f"Portfolio weights sum is not equal to 1: {total_weight:.2f}"
                    )
                    weights = {k: v/total_weight for k, v in weights.items()}
            
                # 결과 표시
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Portfolio Composition")
                    weights_df = pd.DataFrame(
                        list(weights.items()),
                        columns=['Asset', 'Weight']
                    )
                    weights_df['Weight'] = weights_df['Weight'] * 100
                    
                    # 원형 차트로 표시
                    fig_pie = plt.figure(figsize=(10, 6))
                    plt.pie(
                        weights_df['Weight'],
                        labels=weights_df['Asset'],
                        autopct='%1.1f%%',
                        startangle=90
                    )
                    plt.title("Portfolio Asset Allocation")
                    st.pyplot(fig_pie)
                    
                    # 데이터프레임으로도 표시
                    st.dataframe(weights_df.style.format({
                        'Weight': '{:.2f}%'
                    }))
            
                with col2:
                    st.subheader("Portfolio Performance Metrics")
                    metrics = {
                        'Annual Expected Return': performance[0],
                        'Annual Volatility': performance[1],
                        'Sharpe Ratio': performance[2]
                    }
                    metrics_df = pd.DataFrame(
                        metrics.items(),
                        columns=['Metric', 'Value']
                    )
                    st.dataframe(metrics_df.style.format({
                        'Value': '{:.2%}'
                    }))
                    
                    # 위험 지표
                    portfolio_returns = returns.dot(list(weights.values()))
                    risk_metrics = calculate_risk_metrics(portfolio_returns)
                    risk_df = pd.DataFrame(
                        risk_metrics.items(),
                        columns=['Risk Metric', 'Value']
                    )
                    st.dataframe(risk_df.style.format({
                        'Value': '{:.4f}'
                    }))
                
                # 전략 설명
                st.markdown("### Selected Strategy Description")
                st.markdown(method_description)
                
                # 누적 수익률 비교
                st.subheader("Comparison of Optimized Portfolio vs Individual Asset Returns")
                portfolio_cumulative = (1 + returns.dot(list(weights.values()))).cumprod()
                comparison_df = pd.concat([portfolio_cumulative, cumulative_returns], axis=1)
                comparison_df.columns = ['Optimized Portfolio'] + list(cumulative_returns.columns)
                st.line_chart(comparison_df)

            with tab2:
                st.header("리밸런싱 시뮬레이션 🔄")
                
                # 사전 조건 검증
                if 'weights' not in globals() or weights is None:
                    st.error("⚠️ 먼저 '포트폴리오 최적화' 탭에서 분석을 실행해 주세요")
                    st.stop()
                
                initial_value = 10000  # USD 기준
                
                # 리밸런싱 주기 매핑
                freq_mapping = {
                    "Monthly": 'M',
                    "Quarterly": 'Q',
                    "Semi-Annually": '6M',
                    "Annually": 'A'
                }
                
                # 리밸런싱 시뮬레이션
                portfolio_value = initial_value
                dates = returns.index
                rebalance_dates = pd.date_range(
                    start=dates[0],
                    end=dates[-1],
                    freq=freq_mapping[rebalance_period]
                )
                
                portfolio_values = []
                current_weights = weights
                
                for date in dates:
                    if date in rebalance_dates:
                        # 리밸런싱 수행
                        if optimization_method == "최대 샤프 지수 (Maximum Sharpe Ratio) ⚡":
                            ef = EfficientFrontier(expected_returns.mean_historical_return(data), risk_models.sample_cov(data))
                            current_weights = ef.max_sharpe()
                            current_weights = ef.clean_weights()
                        elif optimization_method == "최소 변동성 (Minimum Volatility) 🛡️":
                            ef = EfficientFrontier(expected_returns.mean_historical_return(data), risk_models.sample_cov(data))
                            current_weights = ef.min_volatility()
                            current_weights = ef.clean_weights()
                        else:  # Risk Parity (Risk Parity)
                            hrp = HRPOpt(returns.loc[:date])
                            weights_array = hrp.optimize()
                            current_weights = {
                                asset: weight for asset, weight in 
                                zip(returns.columns, weights_array)
                            }
                    
                    # 포트폴리오 가치 업데이트
                    daily_return = returns.loc[date].dot(
                        list(current_weights.values())
                    )
                    portfolio_value *= (1 + daily_return)
                    portfolio_values.append(portfolio_value)
                
                # 포트폴리오 가치 시계열 생성 (데이터 타입 명시적 변환)
                portfolio_series = pd.Series(
                    data=np.array(portfolio_values, dtype=np.float64),  # 명시적 타입 지정
                    index=pd.to_datetime(dates),  # datetime 변환 보장
                    name="Portfolio Value"
                )
                
                # 리밸런싱 정보 테이블 생성
                rebalance_info = pd.DataFrame({
                    'Rebalancing Date': pd.to_datetime(rebalance_dates)
                })
                
                # 포트폴리오 가치 추출 방식 개선
                rebalance_info['Portfolio Value'] = rebalance_info['Rebalancing Date'].apply(
                    lambda x: portfolio_series.asof(x) if x >= portfolio_series.index[0] else np.nan
                ).dropna().astype(np.float64)
                
                # 결과 시각화
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Performance (USD)")
                    st.metric("Initial", f"${initial_value:,.2f}")
                    final_value = portfolio_series.iloc[-1]
                    total_return = (final_value - initial_value) / initial_value * 100
                    annual_return = (final_value / initial_value) ** (252/len(portfolio_series)) - 1
                    st.metric("Final Value", 
                            f"${final_value:,.2f}", 
                            f"{total_return:+.2f}%")
                
                with col2:
                    st.subheader("Portfolio Value")
                    st.line_chart(portfolio_series)
                
                # 리밸런싱 시점 표시
                st.subheader("Rebalancing Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 테이블 형식으로 표시
                    st.dataframe(rebalance_info.style.format({
                        'Portfolio Value': '${:,.2f}'
                    }))
                
                with col2:
                    # 리밸런싱 시점의 포트폴리오 가치 변화 그래프
                    fig = plt.figure(figsize=(10, 6))
                    plt.plot(
                        rebalance_info['Rebalancing Date'].values.astype('datetime64[s]'),  # matplotlib 호환 형식
                        rebalance_info['Portfolio Value'].values.astype(np.float64),
                        marker='o', 
                        linestyle='-'
                    )
                    plt.title('Portfolio Value Change at Rebalancing Points')
                    plt.xticks(rotation=45)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # 리밸런싱 효과 설명
                    with st.expander("📊 Rebalancing Effect Analysis", expanded=False):
                        # 리밸런싱 전후 수익률 변화 계산
                        rebalance_returns = pd.Series(
                            index=rebalance_info.index[1:],
                            data=np.diff(rebalance_info['Portfolio Value']) / 
                                 rebalance_info['Portfolio Value'].iloc[:-1]
                        )
                        
                        st.markdown("""
                        ### Rebalancing Effect
                        - Portfolio weights are automatically adjusted at each rebalancing point.
                        - This results in automatic 'sell high, buy low' effect.
                        """)
                        
                        # 리밸런싱 구간별 수익률 표시
                        st.markdown("#### Rebalancing Interval Returns")
                        returns_df = pd.DataFrame({
                            'Interval': [f"{start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}"
                                    for start, end in zip(rebalance_info['Rebalancing Date'][:-1], 
                                                        rebalance_info['Rebalancing Date'][1:])],
                            'Interval Return': rebalance_returns
                        })
                        st.dataframe(returns_df.style.format({
                            'Interval Return': '{:.2%}'
                        }))

            with tab3:
                st.header("Stress Test")
                
                with st.expander("📚 What is Stress Test?", expanded=False):
                    st.markdown("""
                    ### What is Stress Test?
                    Stress Test is a method of simulating how a portfolio would react in various market crisis situations.
                    
                    ### Key Risk Metrics Description
                    | Metric | Description | Interpretation |
                    |------|------|------|
                    | **Volatility (Volatility)** | Degree of return change | Lower is better |
                    | **Sharpe (Sharpe Ratio)** | Return to risk ratio | Higher is better |
                    | **Sortino (Sortino Ratio)** | Return to downside risk | Higher is better |
                    | **MDD (Maximum Drawdown)** | Downside from peak to trough | Lower is better |
                    | **VaR (Value at Risk)** | Maximum expected loss at 95% confidence level | Lower is better |
                    | **CVaR (Conditional VaR)** | Average loss in case of VaR exceedance | Lower is better |
                    
                    ### Scenario Description
                    | Scenario | Shock | Description |
                    |----------|------|------|
                    | **Market Crash** | -20% | Sudden market collapse situation |
                    | **Severe Recession** | -30% | Severe economic downturn situation |
                    | **Moderate Downturn** | -10% | General market downturn situation |
                    | **Tech Bubble** | -25% | Tech bubble collapse situation |
                    | **Financial Crisis** | -35% | Financial crisis situation |
                    """)
                
                # 스트레스 시나리오 정의
                scenarios = {
                    "Market Crash": -0.20,
                    "Severe Recession": -0.30,
                    "Moderate Downturn": -0.10,
                    "Tech Bubble": -0.25,
                    "Financial Crisis": -0.35
                }
                
                # 스트레스 테스트 실행
                stress_results = run_stress_test(returns, scenarios)
                
                # 결과 시각화
                stress_df = pd.DataFrame(stress_results).T
                st.dataframe(stress_df.style.format('{:.4f}'))
                
                # 히트맵 생성
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(
                    stress_df, 
                    annot=True, 
                    fmt='.4f',
                    cmap='RdYlGn', 
                    center=0
                )
                st.pyplot(fig)
        
            with tab4:
                st.header("동적 헤지 전략 🛡️")
                
                # 헤지 파라미터 준비
                hedge_config = {
                    'base_threshold': base_threshold / 100,
                    'intensity': hedge_intensity,
                    'max_ratio': max_hedge_ratio
                }
                
                # 데이터 준비
                portfolio_returns, window_size = prepare_hedge_data(returns)
                
                # 헤지 비율 계산
                hedge_ratio = simulate_dynamic_hedge(
                    portfolio_returns,
                    window_size,
                    hedge_config
                )
                
                if hedge_ratio is not None:
                    # 헤지 비율 시계열 생성
                    hedge_ratio_series = pd.Series(
                        hedge_ratio,
                        index=portfolio_returns.index[window_size:],
                        name='Hedge Ratio'
                    )
                    
                    # 헤지 적용 수익률 계산
                    hedged_returns = portfolio_returns[window_size:] * (1 - hedge_ratio_series)
                    
                    # 시각화 부분만 별도 함수로 분리
                    def update_hedge_visualization(hedge_ratio, hedged_returns):
                        """임계값 변경 시 시각화만 업데이트"""
                        comparison = pd.DataFrame({
                            '원본 포트폴리오': portfolio_returns[window_size:],
                            '헤지 포트폴리오': hedged_returns,
                            '헤지 비율': hedge_ratio
                        }).dropna()
                        
                        # 결과 시각화
                        st.subheader("수익률 비교 📈")
                        st.line_chart(comparison[['원본 포트폴리오', '헤지 포트폴리오']])
                        
                        # 헤지 비율 차트 개선
                        st.subheader("헤지 비율 변화 추이 📉")
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.fill_between(
                            comparison.index,
                            comparison['헤지 비율'],
                            color='red',
                            alpha=0.3,
                            label='헤지 비율'
                        )
                        ax.plot(
                            comparison.index,
                            comparison['헤지 비율'],
                            color='darkred',
                            linewidth=1.5
                        )
                        ax.set_ylim(0, 1.0)
                        ax.set_yticks(np.arange(0, 1.1, 0.2))
                        ax.set_ylabel('헤지 비율')
                        ax.legend(loc='upper left')
                        st.pyplot(fig)
                        
                        # 헤지 효과 분석
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("원본 포트폴리오 성과 지표")
                            original_metrics = calculate_risk_metrics(
                                portfolio_returns[window_size:]
                            )
                            metrics_df = pd.DataFrame(
                                original_metrics.items(),
                                columns=['Metric', 'Value']
                            )
                            st.dataframe(metrics_df.style.format({
                                'Value': '{:.4f}'
                            }))
                        
                        with col2:
                            st.subheader("헤지 포트폴리오 성과 지표")
                            hedged_metrics = calculate_risk_metrics(
                                hedged_returns
                            )
                            metrics_df = pd.DataFrame(
                                hedged_metrics.items(),
                                columns=['Metric', 'Value']
                            )
                            st.dataframe(metrics_df.style.format({
                                'Value': '{:.4f}'
                            }))
                    
                    update_hedge_visualization(hedge_ratio_series, hedged_returns)
                    
                    # 헤지 전략 설명 섹션
                    with st.expander("📚 동적 헤지 전략 상세 설명", expanded=True):
                        st.markdown(f"""
                        ### 🛠️ 현재 헤지 설정 값
                        | 파라미터 | 값 | 설명 |
                        |----------|----|------|
                        | **기본 변동성 임계값** | {base_threshold}% | 헤지가 시작되는 변동성 수준 |
                        | **헤지 강도** | {hedge_intensity}x | 변동성 증가에 따른 헤지 반응 강도 |
                        | **최대 헤지 비율** | {max_hedge_ratio*100}% | 포트폴리오의 최대 헤지 가능 비중 |
                        
                        ### 📈 전략 작동 메커니즘
                        1. **실시간 변동성 추적**
                           - GARCH(1,1) 모델을 사용한 롤링 윈도우 예측 (3개월 기준)
                           - 연율화 변동성으로 환산하여 모니터링
                        
                        2. **헤지 비율 결정**
                           - 예측 변동성이 임계값 초과 시:  
                           `헤지 비율 = min(최대헤지비율, (초과변동성 / 기본임계값) * 헤지강도)`
                           - 임계값 미만 시: 헤지 비율 0%
                        
                        3. **리스크 관리**
                           - 일일 헤지 비율 재계산
                           - 급격한 변동성 증가 시 단계적 헤지 증가
                           - 최대 헤지 비율로 과도한 헤지 방지
                        
                        ### 💡 효과적인 사용 가이드
                        - **방어적 전략**: 낮은 임계값(10~15%) + 높은 강도(1.5~2x)
                        - **균형 전략**: 중간 임계값(15~20%) + 기본 강도(1x)
                        - **공격적 전략**: 높은 임계값(20%+) + 낮은 강도(0.5~0.8x)
                        
                        ### ⚠️ 주의 사항
                        - 과도한 헤지 강도는 기회비용 증가로 이어질 수 있음
                        - 변동성 예측 모델의 한계 고려 (과거 데이터 기반)
                        - 실제 거래 시 실행 시차 고려 필요
                        """)
                else:
                    st.warning("헤지 분석에 실패했습니다. 변동성 임계값을 조정해 보세요.")

        except Exception as e:
            st.error(f"분석 오류: {str(e)}")
    else:
        st.error("데이터 로딩 실패. 종목 코드와 기간을 확인해 주세요") 
