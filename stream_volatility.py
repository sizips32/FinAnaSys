import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import datetime as dt

try:
    from arch import arch_model
except ImportError:
    st.error("""
    arch 패키지가 설치되지 않았습니다. 터미널에서 다음 명령어를 실행하세요:
    pip install arch
    """)
    st.stop()

import matplotlib.font_manager as fm


########################################
# 1. 필수 패키지/의존성 확인
########################################
def check_dependencies():
    missing_packages = []
    try:
        import streamlit
    except ImportError:
        missing_packages.append("streamlit")
    try:
        import pandas
    except ImportError:
        missing_packages.append("pandas")
    try:
        import FinanceDataReader
    except ImportError:
        missing_packages.append("FinanceDataReader")
    try:
        import arch
    except ImportError:
        missing_packages.append("arch")

    if missing_packages:
        st.error(f"""
        다음 패키지들이 설치되지 않았습니다: {', '.join(missing_packages)}
        터미널에서 다음 명령어를 실행하세요:
        pip install {' '.join(missing_packages)}
        """)
        st.stop()

check_dependencies()


########################################
# 2. Streamlit 페이지 기본 설정
########################################
st.set_page_config(layout="wide")

# 한글 폰트 설정
try:
    font_path = '/font/NanumGothic.ttf'
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc('font', family=font_prop.get_name())
except:
    try:
        # MacOS
        plt.rc('font', family='AppleGothic')
    except:
        try:
            # Windows
            plt.rc('font', family='Malgun Gothic')
        except:
            plt.rc('font', family='sans-serif')
            st.warning("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

# 음수 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


########################################
# 3. 지표별 정의 (간단 설명)
########################################
st.header("📌 주요 지표 정의 및 해석")

with st.expander("1. Sharpe Ratio (샤프 비율)", expanded=False):
    st.markdown("""
    **정의**: (초과수익률) / (전체 변동성)
    - 전체 변동성 대비, 위험 대비 수익이 얼마나 효율적인지 평가
    
    **일반적인 범위**:
    - < 0: 위험이 수익보다 큼
    - 1~2: 보통
    - 2~3: 우수
    - > 3: 매우 우수
    """)

with st.expander("2. Sortino Ratio (소르티노 비율)", expanded=False):
    st.markdown("""
    **정의**: (초과수익률) / (하방 변동성)
    - 상승 변동성 제외, '하락 위험'만으로 수익 효율을 측정
    
    **일반적인 범위**:
    - < 0: 하락 위험 큼
    - 1~2: 보통
    - > 2: 안정적 수익
    """)

with st.expander("3. MDD (Maximum Drawdown, 최대 낙폭)", expanded=False):
    st.markdown("""
    **정의**: 투자기간 중 **가장 큰 낙폭** (최고점 대비 몇 %까지 하락했는지)
    
    **일반적인 범위**:
    - ~10%: 매우 안정
    - 10~20%: 보통
    - 20~30%: 고위험
    - > 30%: 매우 큼
    """)

with st.expander("4. Alpha (알파)", expanded=False):
    st.markdown("""
    **정의**: 시장(벤치마크) 대비 **초과성과** (베타로 설명되지 않는 추가 수익)
    
    **일반적인 범위**:
    - > 0: 시장 대비 초과수익
    - < 0: 시장 대비 부진
    """)

with st.expander("5. Beta (베타)", expanded=False):
    st.markdown("""
    **정의**: 시장 변동(벤치마크)에 대한 포트폴리오 민감도
    
    **일반적인 범위**:
    - < 1: 시장보다 변동성 낮음(방어적)
    - = 1: 시장과 유사
    - > 1: 시장보다 변동성 큼(공격적)
    """)

with st.expander("6. Information Ratio (정보 비율)", expanded=False):
    st.markdown("""
    **정의**: 벤치마크 대비 초과수익을 **Tracking Error(초과 변동성)** 대비 얼마나 안정적으로 내는지
    
    **일반적인 범위**:
    - <= 0: 벤치마크 대비 초과수익 없음
    - 0.5~1: 의미 있는 초과수익
    - > 1: 우수
    """)

with st.expander("7. Treynor Ratio (트레이너 비율)", expanded=False):
    st.markdown("""
    **정의**: (초과수익) / Beta
    - 시스템적 위험(베타)을 감수한 대가로 얼마만큼 초과이익을 냈는지
    
    **일반적인 범위**:
    - > 0: 베타 대비 초과이익 있음 (수치가 높을수록 유리)
    """)

with st.expander("8. Calmar Ratio (칼마 비율)", expanded=False):
    st.markdown("""
    **정의**: (연간 수익률) / (최대 낙폭(MDD)의 절댓값)
    - 낙폭 대비 **연수익**이 얼마나 되는지 평가 (장기 안정성과 수익성 함께 고려)
    
    **일반적인 범위**:
    - < 1: 낙폭 대비 수익률 작음
    - 1~2: 보통
    - 2~3: 우수
    - > 3: 매우 우수
    """)


########################################
# 4. 지표 계산 함수들
########################################

def fetch_data(ticker, start_date, end_date):
    """FinanceDataReader로 주가 데이터를 다운로드 후 일별 수익률을 계산합니다."""
    try:
        # 데이터 다운로드
        data = fdr.DataReader(ticker, start_date, end_date)
        
        # 데이터 유효성 검사
        if data.empty:
            st.error(f"'{ticker}' 종목의 데이터를 찾을 수 없습니다.")
            return pd.DataFrame()
            
        if 'Close' not in data.columns:
            st.error("종가(Close) 데이터가 없습니다.")
            return pd.DataFrame()
            
        # NaN 값 처리
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # 최소 2개 이상의 데이터 포인트가 필요
        if len(data) < 2:
            st.error("분석을 위한 충분한 데이터가 없습니다. (최소 2일 이상 필요)")
            return pd.DataFrame()
        
        # 수익률 계산
        try:
            returns = data['Close'].pct_change()
            data['Returns'] = returns
            
            # 첫 번째 행의 NaN 제거
            data = data.iloc[1:]
            
            return data
            
        except Exception as e:
            st.error(f"수익률 계산 중 오류 발생: {str(e)}")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"데이터 다운로드 중 오류 발생: {str(e)}")
        return pd.DataFrame()

def calculate_volatility(data):
    """GARCH(1,1) 기반 일별 변동성 추정."""
    from arch import arch_model
    model = arch_model(data['Returns'].dropna(), vol='Garch', p=1, q=1)
    results = model.fit(disp='off')
    data['Volatility'] = np.sqrt(results.conditional_volatility)
    return data

def bayesian_analysis(data, prior_mean=0.10, prior_variance=0.05):
    """
    베이지안 사후 평균 및 분산 추정.
    연역적 추론(If-Then) + 확률적 예측(베이지안) 의사결정 근거.
    """
    try:
        # NaN 값 제거 및 데이터 유효성 검사
        returns = data['Returns'].dropna()
        if len(returns) == 0:
            st.warning("수익률 데이터가 비어있습니다.")
            return prior_mean, prior_variance

        likelihood_variance = np.var(returns)
        if likelihood_variance == 0:
            st.warning("수익률의 분산이 0입니다. 기본값을 사용합니다.")
            return prior_mean, prior_variance

        posterior_mean = (
            (prior_mean / prior_variance) +
            (np.mean(returns) / likelihood_variance)
        ) / (
            (1 / prior_variance) + (1 / likelihood_variance)
        )
        posterior_variance = 1 / (
            (1 / prior_variance) + (1 / likelihood_variance)
        )

        # 결과 유효성 검사
        if not np.isfinite(posterior_mean) or not np.isfinite(posterior_variance):
            st.warning("계산된 값이 유효하지 않습니다. 기본값을 사용합니다.")
            return prior_mean, prior_variance

        return posterior_mean, posterior_variance
    except Exception as e:
        st.error(f"베이지안 분석 중 오류 발생: {str(e)}")
        return prior_mean, prior_variance

def generate_investment_signal(posterior_mean, risk_free=0.05):
    """
    베이지안 사후 평균과 무위험 금리를 비교해
    매수/관망 신호 텍스트를 생성.
    """
    if posterior_mean > risk_free:
        return "💡 **매수/유지 신호: 사후 평균이 무위험 수익률을 상회**"
    else:
        return "🔍 **관망/매도 신호: 사후 평균이 무위험 수익률 이하**"

def calculate_var(data, confidence_level=0.95):
    """VaR(Value at Risk) 계산."""
    var_value = np.percentile(data['Returns'].dropna(), (1 - confidence_level) * 100)
    return var_value

def calculate_mdd(returns_series):
    """MDD(Maximum Drawdown) 계산."""
    cum_returns = (returns_series + 1).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def calculate_sharpe_ratio(returns_series, risk_free_rate=0.05):
    """Sharpe Ratio (연단위)."""
    daily_rf = risk_free_rate / 252
    excess_returns = returns_series - daily_rf
    avg_excess = excess_returns.mean() * 252
    std_excess = excess_returns.std() * np.sqrt(252)
    if std_excess == 0:
        return 0.0
    return avg_excess / std_excess

def calculate_sortino_ratio(returns_series, risk_free_rate=0.05):
    """Sortino Ratio (연단위)."""
    daily_rf = risk_free_rate / 252
    excess_returns = returns_series - daily_rf
    negative_returns = excess_returns[excess_returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252)
    avg_excess = excess_returns.mean() * 252
    if downside_std == 0:
        return 0.0
    return avg_excess / downside_std

def calculate_information_ratio(portfolio_returns, benchmark_returns):
    """Information Ratio = (Rp - Rb) / Tracking Error."""
    diff = portfolio_returns - benchmark_returns
    mean_diff = diff.mean() * 252
    std_diff = diff.std() * np.sqrt(252)
    if std_diff == 0:
        return 0.0
    return mean_diff / std_diff

def calculate_treynor_ratio(portfolio_returns, benchmark_returns, risk_free_rate=0.05):
    """Treynor Ratio = (Rp - Rf) / Beta."""
    daily_rf = risk_free_rate / 252
    ex_portfolio = portfolio_returns - daily_rf
    df = pd.DataFrame({'p': ex_portfolio, 'b': benchmark_returns - daily_rf}).dropna()

    var_bench = np.var(df['b'])
    cov_pb = np.cov(df['p'], df['b'])[0, 1]
    if var_bench == 0:
        return 0.0
    beta = cov_pb / var_bench

    mean_ex_portfolio = df['p'].mean() * 252
    if beta == 0:
        return 0.0
    return mean_ex_portfolio / beta

def calculate_alpha_beta(portfolio_returns, benchmark_returns, risk_free_rate=0.05, freq='Daily'):
    """
    Alpha, Beta를 (일간/주간/월간) 데이터로 계산하고 연간화.
    """
    daily_rf = risk_free_rate / 252
    p_excess = portfolio_returns - daily_rf
    b_excess = benchmark_returns - daily_rf

    df = pd.DataFrame({'p': p_excess, 'b': b_excess}).dropna()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if freq == 'Weekly':
        weekly_df = (1 + df).resample('W-FRI').prod() - 1
        p_excess = weekly_df['p']
        b_excess = weekly_df['b']
        annual_factor = 52
    elif freq == 'Monthly':
        monthly_df = (1 + df).resample('M').prod() - 1
        p_excess = monthly_df['p']
        b_excess = monthly_df['b']
        annual_factor = 12
    else:
        annual_factor = 252

    if len(p_excess) < 2:
        return 0.0, 0.0

    var_b = np.var(b_excess)
    cov_pb = np.cov(p_excess, b_excess)[0, 1]
    if var_b == 0:
        return 0.0, 0.0

    beta = cov_pb / var_b
    alpha = (p_excess.mean() - beta * b_excess.mean()) * annual_factor
    return alpha, beta

def calculate_calmar_ratio(returns_series):
    """
    Calmar Ratio = (연간 수익률) / |MDD|.
    """
    annual_return = returns_series.mean() * 252
    mdd = calculate_mdd(returns_series)
    if mdd == 0:
        return 0.0
    return annual_return / abs(mdd)


########################################
# 5. 구간별 해석 함수 (MDD, Calmar 예시)
########################################

def interpret_mdd(mdd_value: float) -> str:
    """
    MDD를 %로 환산하여 구간별 해석
    예) ~10%: 매우 안정, 10~20%: 보통, 20~30%: 고위험, 30%이상: 매우 큼
    """
    mdd_pct = abs(mdd_value * 100)
    if mdd_pct < 10:
        return f"- **MDD**: {mdd_pct:.2f}% → 낙폭 작고 안정적."
    elif 10 <= mdd_pct < 20:
        return f"- **MDD**: {mdd_pct:.2f}% → 보통 수준 낙폭."
    elif 20 <= mdd_pct < 30:
        return f"- **MDD**: {mdd_pct:.2f}% → 고위험 구간, 낙폭 큼."
    else:
        return f"- **MDD**: {mdd_pct:.2f}% → 매우 큰 낙폭, 주의 요망."

def interpret_calmar_ratio(calmar: float) -> str:
    """
    Calmar Ratio 구간별 해석
    0 이하: 성과 부진
    0~1   : 낙폭 대비 수익률 작음
    1~2   : 보통
    2~3   : 우수
    3 이상: 매우 우수
    """
    if calmar <= 0:
        return f"- **Calmar Ratio**: {calmar:.2f} → 0 이하 (매우 부진)."
    elif 0 < calmar < 1:
        return f"- **Calmar Ratio**: {calmar:.2f} → 낙폭 대비 수익률 작음."
    elif 1 <= calmar < 2:
        return f"- **Calmar Ratio**: {calmar:.2f} → 보통 수준."
    elif 2 <= calmar < 3:
        return f"- **Calmar Ratio**: {calmar:.2f} → 우수, 안정적 성과."
    else:
        return f"- **Calmar Ratio**: {calmar:.2f} → 매우 우수."


########################################
# 6. 동적 헤지 시뮬레이션
########################################
def simulate_hedging(data, hedge_ratio, initial_cash=100000):
    """
    변동성이 특정 임계값(VIX 기준)을 초과하면 헤지 비중을 높이고,
    그렇지 않으면 낮추는 단순 시뮬레이션 예시.
    """
    portfolio_value = initial_cash
    cash = initial_cash * (1 - hedge_ratio/100)
    hedge = initial_cash * (hedge_ratio/100)
    portfolio = []

    for i in range(1, len(data)):
        # 변동성이 임계값보다 높으면 헤지 비중 확대
        if data['Volatility'].iloc[i] > (volatility_threshold / 100):
            hedge *= 1.05
            cash -= (hedge * 0.05)
        else:
            # 변동성이 낮으면 헤지 비중 축소
            hedge *= 0.95
            cash += (hedge * 0.05)

        portfolio_value = cash + hedge
        portfolio.append(portfolio_value)

    data['Portfolio'] = [initial_cash] + portfolio
    return data


########################################
# 7. 사이드바 (사용자 입력)
########################################
st.sidebar.header('설정 옵션')

# 마켓 선택
market = st.sidebar.selectbox(
    '마켓 선택',
    ['KRX', 'NASDAQ', 'NYSE', 'AMEX']
)

# 티커 입력 안내
if market == 'KRX':
    ticker_guide = "종목코드 입력 (예: 005930)"
    default_ticker = "005930"  # 삼성전자
    benchmark_guide = "벤치마크 종목코드 (예: KS11)"  # KOSPI
    default_benchmark = "KS11"
else:
    ticker_guide = "티커 입력 (예: AAPL)"
    default_ticker = "AAPL"  # Apple
    benchmark_guide = "벤치마크 티커 (예: SPY)"  # S&P 500 ETF
    default_benchmark = "SPY"

ticker = st.sidebar.text_input('종목 코드', value=default_ticker, help=ticker_guide)
benchmark_ticker = st.sidebar.text_input('벤치마크', value=default_benchmark, help=benchmark_guide)

start_date = st.sidebar.date_input('시작 날짜', dt.date(2015, 1, 1))
end_date = st.sidebar.date_input('종료 날짜', dt.date.today())

hedge_ratio = st.sidebar.slider('헤지 비율 (%)', 5, 20, 10)
volatility_threshold = st.sidebar.slider('변동성 임계값 (VIX)', 10, 50, 20)

prior_mean = st.sidebar.number_input('사전 기대 수익률', value=0.10, format="%.4f")
prior_variance = st.sidebar.number_input('사전 불확실성', value=0.05, format="%.4f")

threshold = st.sidebar.number_input('변동성 임계값(베이지안)', value=0.05, format="%.4f")
risk_free_rate = st.sidebar.number_input('무위험 수익률 (예: 0.05)', value=0.05, format="%.4f")

freq_option = st.sidebar.selectbox(
    '알파·베타 계산 빈도 (Daily/Weekly/Monthly)',
    ['Daily', 'Weekly', 'Monthly']
)

execute = st.sidebar.button("시뮬레이션 실행")
initial_cash = 100000


########################################
# 8. 메인 실행 로직
########################################
if execute:
    # (1) 분석 대상 데이터 불러오기
    if market == 'KRX' and not ticker.isdigit():
        st.error("KRX 시장의 경우 숫자로 된 종목코드를 입력해주세요.")
    else:
        data = fetch_data(ticker, start_date, end_date)
        # (2) 벤치마크 데이터
        benchmark_data = fetch_data(benchmark_ticker, start_date, end_date)

        if data.empty or data['Returns'].isnull().all():
            st.error("분석 대상 수익률 데이터가 없습니다. 다른 티커를 확인해주세요.")
        else:
            # (3) 변동성 계산 (GARCH)
            data = calculate_volatility(data)

            # (4) 베이지안 분석 (연역적+확률적 추론)
            st.header('🔮 베이지안 변동성 분석')
            posterior_mean, posterior_variance = bayesian_analysis(data, prior_mean, prior_variance)
            st.write(f"**사후 평균 (Posterior Mean)**: {posterior_mean:.4f}")
            st.write(f"**사후 분산 (Posterior Variance)**: {posterior_variance:.4f}")

            # (5) 매수/관망 신호 (generate_investment_signal)
            investment_signal = generate_investment_signal(posterior_mean, risk_free_rate)
            st.write(investment_signal)

            # (6) 투자설명 (연역적 로직 + 사후평균 vs 무위험이자율)
            if posterior_mean > risk_free_rate:
                st.markdown("""
                - **사후 평균**이 무위험 수익률보다 **높음**  
                - 이는 **긍정적 기대수익**을 시사 → **매수/유지** 전략 고려
                """)
            else:
                st.markdown("""
                - **사후 평균**이 무위험 수익률 **이하**  
                - 기대수익이 높지 않음 → **관망/매도** 또는 대체자산 고려
                """)

            # (7) VaR
            var_value = calculate_var(data)
            st.write(f"**VaR (95% 신뢰수준)**: {var_value:.4f}")

            # (8) 변동성 시각화
            st.subheader('📈 변동성 시각화')
            fig, ax = plt.subplots()
            ax.plot(data.index, data['Volatility'], label='Volatility')
            ax.axhline(volatility_threshold/100, color='r', linestyle='--', label='Threshold')
            ax.legend()
            st.pyplot(fig)

            # (9) 동적 헤지 시뮬레이션
            st.header('🛡️ 동적 헤지 전략 시뮬레이션')
            data = simulate_hedging(data, hedge_ratio, initial_cash)

            # (10) 포트폴리오 가치 변화
            st.subheader('💰 포트폴리오 가치 변화')
            fig2, ax2 = plt.subplots()
            ax2.plot(data.index, data['Portfolio'], label='포트폴리오 가치', color='blue')
            ax2.set_title('포트폴리오 가치 변화')
            ax2.set_xlabel('날짜')
            ax2.set_ylabel('포트폴리오 가치 ($)')
            ax2.legend()
            st.pyplot(fig2)

            # (11) 누적 수익률
            data['Cumulative Returns'] = (data['Returns'] + 1).cumprod() - 1
            st.subheader('📊 누적 수익률')
            fig3, ax3 = plt.subplots()
            ax3.plot(data.index, data['Cumulative Returns'], label='누적 수익률', color='green')
            ax3.set_title('누적 수익률')
            ax3.set_xlabel('날짜')
            ax3.set_ylabel('누적 수익률 (%)')
            ax3.legend()
            st.pyplot(fig3)

            # (12) 성과 요약
            st.subheader('📌 성과 요약')
            final_value = data['Portfolio'].iloc[-1]
            total_return = ((final_value / initial_cash) - 1) * 100
            st.write(f"- 최종 포트폴리오 가치: ${final_value:,.2f}")
            st.write(f"- 총 수익률: {total_return:.2f}%")

            # (13) MDD
            st.header('🏳️ MDD (Maximum Drawdown)')
            mdd_value = calculate_mdd(data['Returns'])
            # 구간별 해석
            mdd_text = interpret_mdd(mdd_value)
            st.write(mdd_text)

            # (14) Sharpe & Sortino
            st.header('📈 Sharpe Ratio & Sortino Ratio')
            sharpe_val = calculate_sharpe_ratio(data['Returns'], risk_free_rate)
            sortino_val = calculate_sortino_ratio(data['Returns'], risk_free_rate)
            st.write(f"- Sharpe Ratio: {sharpe_val:.4f}")
            st.write(f"- Sortino Ratio: {sortino_val:.4f}")

            # (15) Information Ratio & Treynor Ratio
            st.header('📊 Information Ratio & Treynor Ratio')
            if not benchmark_data.empty and 'Returns' in benchmark_data.columns:
                info_ratio_val = calculate_information_ratio(data['Returns'], benchmark_data['Returns'])
                treynor_val = calculate_treynor_ratio(data['Returns'], benchmark_data['Returns'], risk_free_rate)
                st.write(f"- Information Ratio: {info_ratio_val:.4f}")
                st.write(f"- Treynor Ratio: {treynor_val:.4f}")

                st.markdown("""
                - **Information Ratio**: 벤치마크 대비 초과수익을 얼마나 '안정적'으로 내는지  
                - **Treynor Ratio**: Beta(시스템적 위험) 1만큼 감수할 때 초과수익이 얼마나 되는지
                """)
            else:
                st.warning("벤치마크 데이터가 부족하여 IR/Treynor 계산 불가.")

            # (16) Alpha & Beta
            st.header('🔎 Alpha & Beta (빈도 선택)')
            if not benchmark_data.empty and 'Returns' in benchmark_data.columns:
                alpha_val, beta_val = calculate_alpha_beta(
                    data['Returns'],
                    benchmark_data['Returns'],
                    risk_free_rate,
                    freq=freq_option
                )
                st.write(f"- Alpha({freq_option}, 연율): {alpha_val:.4f}")
                st.write(f"- Beta({freq_option}): {beta_val:.4f}")
                st.markdown("""
                - Beta > 1 : 시장보다 변동성 큼(공격)  
                - Beta < 1 : 시장보다 변동성 낮음(방어)  
                - Alpha > 0: 시장 대비 초과수익  
                - Alpha < 0: 시장 대비 부진
                """)
            else:
                st.warning("벤치마크 데이터가 없어 Alpha·Beta 계산 불가.")

            # (17) Calmar Ratio
            st.header("🌊 Calmar Ratio")
            calmar_val = calculate_calmar_ratio(data['Returns'])
            calmar_text = interpret_calmar_ratio(calmar_val)
            st.write(calmar_text)

            # (18) 에르고딕 가설
            st.header('🔎 에르고딕 가설 분석')
            st.markdown("""
            **에르고딕 가설**:  
            - 시간 평균 ≈ 표본 평균이면 과거 통계가 미래에도 유효할 가능성 높음
            """)
            cum_ret = data['Cumulative Returns']
            st.line_chart(cum_ret)

            time_avg = cum_ret.mean()  # 시계열의 단순 평균
            ensemble_avg = data['Returns'].mean() * len(data)  # 집합 평균 (단순히 mean x sample_count)
            diff = abs(time_avg - ensemble_avg)

            st.write(f"- 시간 평균 (Time Average): {time_avg:.4f}")
            st.write(f"- 집합 평균 (Ensemble Average): {ensemble_avg:.4f}")
            st.write(f"- 차이값: {diff:.4f}")

            if diff < 0.01:
                st.write("✅ 에르고딕 성질이 **강하게** 성립")
            elif diff < 0.05:
                st.write("🟨 에르고딕 성질이 **약하게** 성립")
            else:
                st.write("❌ 에르고딕 성질이 **성립하지 않음**")

            # (19) 종합 결과
            st.header('🔔 종합 결과 분석')
            st.write(f"**최종 포트폴리오 가치**: ${final_value:,.2f}")
            st.write(f"**총 수익률**: {total_return:.2f}%")
            st.write(f"**MDD**: {mdd_value*100:.2f}%")
            st.write(f"**Sharpe Ratio**: {sharpe_val:.4f}")
            st.write(f"**Sortino Ratio**: {sortino_val:.4f}")
            if not benchmark_data.empty and 'Returns' in benchmark_data.columns:
                st.write(f"**Information Ratio**: {info_ratio_val:.4f}")
                st.write(f"**Treynor Ratio**: {treynor_val:.4f}")
                st.write(f"**Alpha({freq_option})**: {alpha_val:.4f}")
                st.write(f"**Beta({freq_option})**: {beta_val:.4f}")
            st.write(f"**Calmar Ratio**: {calmar_val:.4f}")

            # (20) 🌟 투자 추천 (연역적 로직 + 확률적 예측)
            st.subheader('🌟 투자 의사결정 추천')
            # 간단한 예시 로직: posterior_mean vs risk_free, MDD, Sharpe
            # (실무에서는 훨씬 정교한 조건을 넣어야 함)
            if posterior_mean > risk_free_rate:
                if abs(mdd_value) < 0.2 and sharpe_val > 1.0:
                    st.write("✅ **매수** 추천: 사후 평균이 높고, MDD와 Sharpe Ratio가 괜찮은 편입니다.")
                    st.write(" - 무위험 수익률을 상회하는 기대수익 + 안정적인 변동성 감안.")
                else:
                    st.write("🔄 **유지/추가매수 신중 검토**: 수익률 기대는 높지만, 변동성/리스크도 함께 체크 필요.")
            else:
                if sharpe_val < 0.5 or abs(mdd_value) > 0.2:
                    st.write("❌ **매도** 권장: 기대수익 낮고 변동성(낙폭) 위험도 큼.")
                else:
                    st.write("🔍 **관망**: 기대수익 낮으나, Sharpe Ratio 등이 보통 수준. 시장상황을 좀 더 지켜보세요.")

            # (21) 종합 결과
            st.markdown("---")
            st.success("💡 분석 완료! 위 지표와 해석을 종합해 최종 투자 결정을 내리시길 바랍니다.")
