import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
try:
    from arch import arch_model
except ImportError:
    st.error("""
    arch 패키지가 설치되지 않았습니다. 터미널에서 다음 명령어를 실행하세요:
    pip install arch
    """)
import matplotlib.font_manager as fm
import plotly.graph_objects as go

# 패키지 존재 여부 확인 및 에러 처리
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
        import yfinance
    except ImportError:
        missing_packages.append("yfinance")
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

# 의존성 확인 실행
check_dependencies()

st.set_page_config(layout="wide")

# 한글 폰트 설정 (에러 처리 추가)
try:
    font_path = './font/NanumGothic.ttf'
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc('font', family=font_prop.get_name())
except:
    try:
        # MacOS의 경우
        plt.rc('font', family='AppleGothic')
    except:
        try:
            # Windows의 경우
            plt.rc('font', family='Malgun Gothic')
        except:
            # 모든 시도가 실패할 경우 기본 폰트 사용
            plt.rc('font', family='sans-serif')
            st.warning("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

# 음수 기호 표시 설정
plt.rcParams['axes.unicode_minus'] = False

# Title
st.title('주가 변동성 분석 시스템')

# 투자 의사결정 프로세스 설명 추가
st.markdown("""
### 🎯 투자 의사결정 프로세스

이 시스템은 다음과 같은 단계로 투자 의사결정을 지원합니다:

1. **베이지안 분석을 통한 수익률 예측**
   - 예시: 애플(AAPL)의 사후 평균이 0.03(3%)이고 무위험 수익률이 0.02(2%)인 경우 → 매수 고려
   - 반대로 사후 평균이 0.01(1%)이면 → 매도 또는 관망

2. **에르고딕 가설을 통한 안정성 검증**
   - 예시: 시간평균과 집합평균의 차이가 0.01 미만 → 매우 안정적인 투자
   - 차이가 0.05 이상 → 전략 재검토 필요

3. **변동성 기반 포지션 조절**
   - 예시: VIX > 20 → 헤지 비중 확대
   - VIX < 20 → 주식 비중 확대

### 📊 실제 적용 예시

**시나리오 1: 안정적 상승장**
- 사후 평균: 0.04 (4%)
- 에르고딕 차이: 0.008
- VIX: 15
➡️ 결정: "적극적 매수, 헤지 비중 축소"

**시나리오 2: 불안정한 하락장**
- 사후 평균: 0.01 (1%)
- 에르고딕 차이: 0.06
- VIX: 25
➡️ 결정: "매도 검토, 헤지 비중 확대"

### 📈 분석 개요
""")

st.markdown("""
### 분석 개요
이 시스템은 주식의 변동성과 리스크를 다각도로 분석합니다.

#### 주요 지표:
1. **베이지안 변동성 분석**
   - 시장 대비 변동성 측정
   - 상관관계 분석
   
2. **변동성 지표**
   - 역사적 변동성 계산
   - 내재 변동성 추정

3. **리스크 평가**
   - VaR (Value at Risk) 계산
   - MDD(Maximum Drawdown) 제공
""")

# Sidebar Inputs
st.sidebar.header('설정 옵션')

# 설명 추가
st.sidebar.markdown("### 파라미터 설명")
st.sidebar.markdown("- **주식 티커**: 분석할 주식 또는 ETF 코드(SPY, QQQ 등).")
st.sidebar.markdown("- **시작/종료 날짜**: 분석 기간 설정.")
st.sidebar.markdown("- **헤지 비율**: 포트폴리오에서 변동성 헤지 자산의 비율.")
st.sidebar.markdown("- **변동성 임계값**: VIX 기반으로 추가 헤지를 실행할 변동성 기준치.")
st.sidebar.markdown("- **사전 기대 수익률**: 베이지안 분석의 사전 평균값.")
st.sidebar.markdown("- **사전 불확실성**: 베이지안 분석의 사전 분산값.")

# 입력 파라미터
ticker = st.sidebar.text_input('주식 티커 입력', value='SPY')
start_date = st.sidebar.date_input('시작 날짜', dt.date(2015, 1, 1))
end_date = st.sidebar.date_input('종료 날짜', dt.date.today())
hedge_ratio = st.sidebar.slider('헤지 비율 (%)', 5, 20, 10)
volatility_threshold = st.sidebar.slider('변동성 임계값 (VIX 기준)', 10, 50, 20)

# 베이지안 분석 파라미터
prior_mean = st.sidebar.number_input('사전 기대 수익률', value=0.10, format="%.4f", 
    help="베이지안 분석에서 사용할 사전 평균값입니다. 일반적으로 0.10 (10%) 정도로 설정합니다.")
prior_variance = st.sidebar.number_input('사전 불확실성', value=0.05, format="%.4f",
    help="베이지안 분석에서 사용할 사전 분산값입니다. 불확실성이 클수록 큰 값을 설정합니다.")

# 사용자 입력을 통한 변동성 임계값 설정
threshold = st.sidebar.number_input('변동성 임계값 입력', value=0.05, format="%.4f")  # 기본값 0.05

# 사용자 입력을 통한 무위험 수익률 설정
risk_free_rate = st.sidebar.number_input('무위험 수익률 입력 (예: 0.05)', value=0.05, format="%.4f")  # 기본값 0.05

# 실행 버튼
execute = st.sidebar.button("시뮬레이션 실행")

initial_cash = 100000  # 초기 현금 정의

def fetch_data(ticker, start_date, end_date):
    """주식 데이터를 다운로드하고 수익률을 계산합니다."""
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Adj Close'].pct_change().dropna()
    return data

def calculate_volatility(data):
    """GARCH 모델을 사용하여 변동성을 계산합니다."""
    model = arch_model(data['Returns'].dropna(), vol='Garch', p=1, q=1)
    results = model.fit(disp='off')
    data['Volatility'] = np.sqrt(results.conditional_volatility)
    return data

def bayesian_analysis(data, prior_mean=0.10, prior_variance=0.05):
    """
    Perform Bayesian volatility analysis and calculate the posterior mean and variance.

    Args:
        data (pd.DataFrame): A DataFrame containing a 'Returns' column with return data.
        prior_mean (float, optional): The prior mean for the Bayesian analysis. Defaults to 0.10.
        prior_variance (float, optional): The prior variance for the Bayesian analysis. Defaults to 0.05.

    Returns:
        tuple: A tuple containing the posterior mean and posterior variance.

    Note:
        - `prior_mean` and `prior_variance` are hyperparameters that represent the prior belief about the mean and variance of the returns before observing the data.
        - The function calculates the likelihood variance from the data and combines it with the prior to compute the posterior mean and variance.
    """
    """베이지안 변동성 분석을 수행하고 사후 평균과 분산을 계산합니다."""
    likelihood_variance = np.var(data['Returns'].dropna())
    posterior_mean = (prior_mean / prior_variance + np.mean(data['Returns']) / likelihood_variance) / (1 / prior_variance + 1 / likelihood_variance)
    posterior_variance = 1 / (1 / prior_variance + 1 / likelihood_variance)
    return posterior_mean, posterior_variance

def generate_investment_signal(posterior_mean, threshold=0.05):
    """투자 신호를 생성합니다."""
    if posterior_mean > threshold:
        return "💡 **투자 신호: 사후 평균이 임계값을 초과했습니다. 투자 고려하세요!**"
    else:
        return "🔍 **관망 신호: 사후 평균이 임계값 이하입니다. 신중하게 접근하세요.**"

def calculate_var(data, confidence_level=0.95):
    """VaR (Value at Risk)를 계산합니다."""
    var_value = np.percentile(data['Returns'].dropna(), (1 - confidence_level) * 100)
    return var_value

def simulate_hedging(data, hedge_ratio, initial_cash=100000):
    """동적 헤지 전략을 시뮬레이션합니다."""
    portfolio_value = initial_cash  # 초기 포트폴리오 가치 설정
    cash = initial_cash * (1 - hedge_ratio / 100)  # 현금 설정
    hedge = initial_cash * (hedge_ratio / 100)  # 헤지 자산 설정
    portfolio = []

    for i in range(1, len(data)):
        if data['Volatility'].iloc[i] > volatility_threshold / 100:
            hedge *= 1.05
            cash -= (hedge * 0.05)
        else:
            hedge *= 0.95
            cash += (hedge * 0.05)

        portfolio_value = cash + hedge
        portfolio.append(portfolio_value)

    data['Portfolio'] = [initial_cash] + portfolio
    return data

if execute:
    # 데이터 수집
    data = fetch_data(ticker, start_date, end_date)

    # 수익률 데이터가 비어 있는지 확인
    if data['Returns'].isnull().all():
        st.error("수익률 데이터가 없습니다. 다른 주식 티커를 입력해 주세요.")
    else:
        # 변동성 계산
        data = calculate_volatility(data)

        # 베이지안 변동성 분석 - 사용자 정의 파라미터 사용
        st.header('베이지안 변동성 분석')
        posterior_mean, posterior_variance = bayesian_analysis(data, prior_mean, prior_variance)

        st.write(f"사후 평균 (Posterior Mean): {posterior_mean:.4f}")
        st.write(f"사후 분산 (Posterior Variance): {posterior_variance:.4f}")

        # 투자 신호 생성
        investment_signal = generate_investment_signal(posterior_mean, risk_free_rate)  # 사용자 입력으로 무위험 수익률 전달
        st.write(investment_signal)

        # 투자 설명 추가
        if posterior_mean > risk_free_rate:
            st.write("### 투자 설명")
            st.write("사후 평균이 무위험 수익률을 초과했습니다. 이는 해당 주식이 앞으로 긍정적인 수익을 낼 가능성이 높다는 것을 의미합니다. "
                      "따라서, 이 주식에 투자하는 것이 좋습니다.")
        else:
            st.write("### 투자 설명")
            st.write("사후 평균이 무위험 수익률 이하입니다. 이는 해당 주식의 수익률이 기대에 미치지 못할 가능성이 높다는 것을 의미합니다. "
                      "따라서, 신중하게 접근하고 다른 투자 기회를 고려하는 것이 좋습니다.")

        # VaR 계산
        var_value = calculate_var(data)
        st.write(f"VaR (신뢰 수준 95%): {var_value:.4f}")

        # 변동성 시각화
        st.subheader('변동성 시각화')
        fig, ax = plt.subplots()
        ax.plot(data.index, data['Volatility'], label='Volatility')
        ax.axhline(volatility_threshold / 100, color='r', linestyle='--', label='Threshold')
        ax.legend()
        st.pyplot(fig)

        # 동적 헤지 전략 시뮬레이션
        st.header('헤지 전략 시뮬레이션')
        data = simulate_hedging(data, hedge_ratio, initial_cash)

        # 포트폴리오 성과 시각화
        st.subheader('포트폴리오 가치 변화')
        fig2, ax2 = plt.subplots()
        ax2.plot(data.index, data['Portfolio'], label='포트폴리오 가치', color='blue')
        ax2.set_title('포트폴리오 가치 변화', fontproperties=font_prop)
        ax2.set_xlabel('날짜', fontproperties=font_prop)
        ax2.set_ylabel('포트폴리오 가치 ($)', fontproperties=font_prop)
        ax2.legend(prop=font_prop)
        st.pyplot(fig2)

        # 누적 수익률 계산
        data['Cumulative Returns'] = (data['Returns'] + 1).cumprod() - 1

        # 누적 수익률 시각화
        st.subheader('누적 수익률')
        fig3, ax3 = plt.subplots()
        ax3.plot(data.index, data['Cumulative Returns'], label='누적 수익률', color='green')
        ax3.set_title('누적 수익률', fontproperties=font_prop)
        ax3.set_xlabel('날짜', fontproperties=font_prop)
        ax3.set_ylabel('누적 수익률 (%)', fontproperties=font_prop)
        ax3.legend(prop=font_prop)
        st.pyplot(fig3)

        # 성과 요약
        st.subheader('성과 요약')
        st.write(f"최종 포트폴리오 가치: ${data['Portfolio'].iloc[-1]:,.2f}")
        st.write(f"수익률: {((data['Portfolio'].iloc[-1] / initial_cash - 1) * 100):.2f}%")

        # 에르고딕 가설 분석 추가
        st.header('에르고딕 가설 분석')
        """
        에르고딕 가설 분석은 시계열 데이터의 시간 평균과 집합 평균이 수렴하는지를 검증합니다.
        이는 과거 데이터를 기반으로 한 투자 전략이 미래에도 유효할지 판단하는 중요한 지표입니다.
        
        1. 시간 평균: 특정 기간 동안의 수익률 평균
        2. 집합 평균: 전체 수익률 분포의 평균
        3. 두 평균의 차이가 작을수록 전략의 안정성이 높음
        """

        # 1. 누적 수익률 계산 및 시각화
        cumulative_returns = (data['Returns'] + 1).cumprod() - 1
        st.line_chart(cumulative_returns)  # 누적 수익률 시각화

        # 2. 시간 평균 (Time Average) 계산
        time_avg = np.mean(cumulative_returns)
        st.write(f"시간 평균 (Time Average): {time_avg:.4f}")
        st.markdown("""
        > 시간 평균의 의미:
        > - 양수: 전체 기간 동안 평균적으로 수익을 냄
        > - 음수: 전체 기간 동안 평균적으로 손실을 봄
        > - 절대값이 클수록 수익/손실의 크기가 큼
        """)

        # 3. 집합 평균 (Ensemble Average) 계산
        ensemble_avg = data['Returns'].mean() * len(data)
        st.write(f"집합 평균 (Ensemble Average): {ensemble_avg:.4f}")
        st.markdown("""
        > 집합 평균의 의미:
        > - 개별 거래일의 수익률 평균에 기간을 곱한 값
        > - 장기 투자 시 기대할 수 있는 이론적 수익률
        """)

        # 4. 에르고딕 성질 검증
        difference = abs(time_avg - ensemble_avg)
        st.write(f"시간 평균과 집합 평균의 차이: {difference:.4f}")
        st.markdown("""
        > 차이값의 해석:
        > - 차이 < 0.01: 매우 안정적인 투자 전략
        > - 0.01 ≤ 차이 < 0.05: 비교적 안정적인 전략
        > - 차이 ≥ 0.05: 불안정한 전략, 재검토 필요
        """)

        # 5. 에르고딕 성질의 성립 여부 판단 및 투자 전략 제시
        if difference < 0.10:
            st.write("""
            ✅ **에르고딕 성질이 강하게 성립합니다.**
            - 투자 전략이 장기적으로 매우 안정적일 것으로 예상됩니다.
            - 현재의 투자 전략을 유지하는 것이 좋습니다.
            - 포지션 크기를 점진적으로 확대하는 것을 고려해볼 수 있습니다.
            """)
        elif difference < 0.05:
            st.write("""
            🟨 **에르고딕 성질이 약하게 성립합니다.**
            - 전략이 비교적 안정적이나 주의가 필요합니다.
            - 현재의 포지션을 유지하되, 리스크 관리를 강화하세요.
            - 정기적인 전략 재검토가 필요합니다.
            """)
        else:
            st.write("""
            ❌ **에르고딕 성질이 성립하지 않습니다.**
            - 현재 전략의 장기 안정성이 낮습니다.
            - 포지션 크기를 줄이는 것을 고려하세요.
            - 전략을 전면 재검토하거나 새로운 전략 개발이 필요합니다.
            - 변동성이 큰 구간에서는 거래를 중단하는 것이 좋습니다.
            """)

        # 전략 추천 섹션
        st.write("### 전략 추천")
        if data['Volatility'].iloc[-1] > volatility_threshold / 100:
            st.write("🔥 **변동성이 높은 상황입니다. 헤지 비중을 확대하고 단기 옵션을 고려하세요.**")
        else:
            st.write("📈 **변동성이 안정적입니다. 핵심 자산 비중을 유지하며 장기 성장 전략을 고려하세요.**")

        # 결과 분석 설명
        st.write("### 결과 분석")
        st.write("이 전략은 변동성 임계값을 기반으로 동적 헤지를 수행하여 시장 급변 상황에 대비합니다.")

        # 종합 결과 분석
        st.header('종합 결과 분석')
        final_portfolio_value = data['Portfolio'].iloc[-1] if 'Portfolio' in data.columns else initial_cash
        st.write(f"최종 포트폴리오 가치: ${final_portfolio_value:,.2f}")
        st.write(f"수익률: {((final_portfolio_value / initial_cash - 1) * 100):.2f}%")

        # 투자 추천
        st.subheader('투자 추천')

        # 1. 사후 평균과 임계값 비교
        if posterior_mean > risk_free_rate:  # 무위험 수익률과 비교
            st.write("🔍 **사후 평균이 무위험 수익률을 초과했습니다.**")
            
            # 2. 포트폴리오 가치 평가
            if final_portfolio_value > initial_cash:
                st.write("📈 **추천: 매수!**")
                st.write("변동성이 높고, 사후 평균이 무위험 수익률을 초과했습니다. "
                         "이는 해당 주식이 긍정적인 수익을 낼 가능성이 높다는 것을 의미합니다.")
            else:
                st.write("🔄 **추천: 홀딩!**")
                st.write("포트폴리오 가치가 초기 투자금보다 낮지만, 사후 평균이 무위험 수익률을 초과합니다. "
                         "따라서, 추가적인 관찰이 필요합니다.")

        # 3. 사후 평균과 임계값 비교 (하위 조건)
        else:
            st.write("🔍 **사후 평균이 무위험 수익률 이하입니다.**")
            
            # 4. 포트폴리오 가치 평가
            if final_portfolio_value < initial_cash:
                st.write("🔻 **추천: 매도!**")
                st.write("변동성이 낮고, 사후 평균이 무위험 수익률 이하입니다. "
                         "이는 해당 주식의 수익률이 기대에 미치지 못할 가능성이 높다는 것을 의미합니다.")
            else:
                st.write("🔄 **추천: 홀딩!**")
                st.write("포트폴리오 가치가 초기 투자금보다 높지만, 사후 평균이 무위험 수익률 이하입니다. "
                         "따라서, 신중하게 접근하고 다른 투자 기회를 고려하는 것이 좋습니다.")

# Footer
st.sidebar.markdown("---")
st.sidebar.text("Created by Sean J. Kim")
