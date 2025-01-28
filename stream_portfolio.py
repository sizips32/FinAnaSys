import warnings

# FutureWarning 무시
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
import yfinance as yf
import pandas as pd
from pypfopt import (
    EfficientFrontier, risk_models, expected_returns,
    BlackLittermanModel, HRPOpt
)

st.title("Hedge Fund Portfolio Optimization")

# User input
tickers = st.text_input(
    "Enter tickers (comma-separated):",
    "034950.KS, 000270.KS, 000815.KS, 115570.KQ, 053800.KQ, 001840.KQ, 003690.KS, 049430.KS, 034950.KQ, 024940.KQ, 004060.KS"
)
start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End date", value=pd.to_datetime("2025-1-31"))

# User input for market outlook
market_outlook = st.selectbox(
    "Select long-term economic outlook:",
    options=["Neutral", "Positive", "Negative"]
)

# Sidebar for individual asset outlook
st.sidebar.subheader("Individual Asset Outlook")
st.sidebar.write("각 자산의 기대 수익률 조정은 해당 자산의 미래 수익률에 대한 개인적인 전망을 반영합니다. "
                 "예를 들어, 1.1을 입력하면 해당 자산의 기대 수익률이 10% 증가한다고 가정합니다.")
individual_outlook = {}
for ticker in tickers.split(","):
    individual_outlook[ticker.strip()] = st.sidebar.number_input(
        f"Expected return adjustment for {ticker.strip()} "
        "(e.g., 1.1 for +10%, 0.9 for -10%)",
        value=1.0
    )

# Fetch data
if st.button("Fetch Data and Optimize"):
    tickers_list = [ticker.strip() for ticker in tickers.split(",")]

    try:
        data = yf.download(tickers_list, start=start_date,
                           end=end_date)['Close']
        if data.empty:
            st.error("데이터를 가져오는 데 실패했습니다. 입력한 티커를 확인하세요.")
        else:
            # Calculate daily returns
            returns = data.pct_change(fill_method=None).dropna()

            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod() - 1

            st.write("Cumulative Returns")
            st.line_chart(cumulative_returns)

            # Calculate expected returns and sample covariance
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)

            # Ensure covariance matrix is symmetric
            S = (S + S.T) / 2

            # Optimization method selection
            methods = [
                "Equal Weight", "Maximum Sharpe Ratio", "Minimum Volatility",
                "Risk Parity", "Black-Litterman"
            ]
            selected_methods = st.multiselect(
                "Select optimization methods",
                methods,
                default=methods
            )

            results = {}
            
            if "Equal Weight" in selected_methods:
                n = len(tickers_list)
                weights = {ticker: 1/n for ticker in tickers_list}
                results["Equal Weight"] = weights

            if "Maximum Sharpe Ratio" in selected_methods:
                ef = EfficientFrontier(mu, S)
                weights = ef.max_sharpe()
                cleaned_weights = ef.clean_weights()
                results["Maximum Sharpe Ratio"] = cleaned_weights

            if "Minimum Volatility" in selected_methods:
                ef = EfficientFrontier(mu, S)
                weights = ef.min_volatility()
                cleaned_weights = ef.clean_weights()
                results["Minimum Volatility"] = cleaned_weights

            if "Risk Parity" in selected_methods:
                hrp = HRPOpt(returns)
                weights = hrp.optimize()
                results["Risk Parity"] = weights

            if "Black-Litterman" in selected_methods:
                market_caps = yf.download(
                    tickers_list, start=start_date, end=end_date
                )['Close'].iloc[-1]
                mcaps = market_caps / market_caps.sum()

                # Check if mu is valid
                if mu is not None and not mu.empty:
                    viewdict = {
                        ticker: mu[ticker] * individual_outlook[ticker]
                        for ticker in tickers_list
                    }

                    bl = BlackLittermanModel(S, pi=mu, absolute_views=viewdict)
                    bl_returns = bl.bl_returns()
                    ef = EfficientFrontier(bl_returns, S)
                    weights = ef.max_sharpe()
                    cleaned_weights = ef.clean_weights()
                    results["Black-Litterman"] = cleaned_weights
                else:
                    st.error(
                        "Expected returns (mu) could not be calculated. "
                        "Please check the input data."
                    )

            # Visualize results
            for method, weights in results.items():
                st.subheader(method)
                weights_df = pd.DataFrame(
                    list(weights.items()),
                    columns=['Asset', 'Weight']
                )
                weights_df['Weight'] = weights_df['Weight'] * 100
                st.write(
                    weights_df.to_html(
                        index=False,
                        float_format=lambda x: f'{x:.2f}%'
                    ),
                    unsafe_allow_html=True
                )

                # Use Streamlit's bar_chart for visualization
                st.bar_chart(weights_df.set_index('Asset'))

            # Generate results
            if results:
                st.subheader("Optimization Results")
                for method, weights in results.items():
                    # Method 이름을 검정색 볼드체로 표시하고 밑줄 추가
                    st.markdown(
                        f"<span style='color:black; font-size:24px; font-weight:bold; "
                        "text-decoration: underline;'>{method} Asset Allocation:</span>",
                        unsafe_allow_html=True
                    )

                    weights_df = pd.DataFrame(
                        list(weights.items()),
                        columns=['Asset', 'Weight']
                    )
                    weights_df['Weight'] = weights_df['Weight'] * 100

                    # 상위 비중 2개 색상 표시 및 볼드체 적용
                    top_weights = weights_df.nlargest(2, 'Weight')
                    weights_df['Weight'] = weights_df['Weight'].apply(
                        lambda x: f"{x:.2f}%")  # 비중 포맷팅

                    # 가로 형태로 테이블 표시
                    weights_df = weights_df.set_index('Asset').T  # 전치하여 가로 형태로 변경
                    
                    # map 대신 applymap 사용
                    def style_top_weights(val):
                        return 'color: red; font-weight: bold;' if val in top_weights['Weight'].values else ''
                    
                    styled_weights_df = weights_df.style.applymap(
                        style_top_weights,
                        subset=top_weights['Asset'].tolist()
                    )

                    st.write(styled_weights_df)  # 스타일이 적용된 데이터프레임 표시

    except Exception as e:
        st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")

# 투자 전략 설명을 expander로 변경
with st.expander("📚 포트폴리오 투자 전략 설명"):
    st.markdown("""
    #### 각 포트폴리오 투자 전략에 대한 설명과 적용 방법을 아래에 정리하였습니다.
    
    1. **Equal Weight (동일 비중)**
    - 설명: 모든 자산에 동일한 비중을 할당하는 전략입니다. 이 방법은 간단하고 직관적이며, 특정 자산에 대한 편향을 피할 수 있습니다.
    - 적용 방법:
        - 포트폴리오에 포함된 자산의 수를 n이라고 할 때, 각 자산의 비중은 1/n으로 설정합니다.
        - 예를 들어, 4개의 자산이 있다면 각 자산의 비중은 25%가 됩니다.

    2. **Maximum Sharpe Ratio (최대 샤프 비율)**
    - 설명: 샤프 비율은 포트폴리오의 초과 수익률을 변동성으로 나눈 값으로, 위험 대비 수익을 측정합니다.
    - 적용 방법:
        - 예상 수익률(mu)과 공분산 행렬(S)을 사용하여 효율적 프론티어를 계산합니다.
        - 샤프 비율을 최대화하는 비중을 계산하여 포트폴리오를 구성합니다.

    3. **Minimum Volatility (최소 변동성)**
    - 설명: 포트폴리오의 변동성을 최소화하는 전략입니다. 이 방법은 위험을 줄이면서 안정적인 수익을 추구합니다.
    - 적용 방법:
        - 예상 수익률(mu)과 공분산 행렬(S)을 사용하여 최소 변동성을 달성하는 자산 비중을 계산합니다.
        - 변동성이 가장 낮은 포트폴리오를 선택합니다.

    4. **Risk Parity (위험 균형)**
    - 설명: 각 자산의 위험 기여도를 균형 있게 조정하여 포트폴리오를 구성하는 전략입니다.
    - 적용 방법:
        - 각 자산의 위험을 평가하고, 위험 기여도가 동일하도록 비중을 조정합니다.
        - 변동성이 높은 자산의 비중은 낮추고, 변동성이 낮은 자산의 비중은 높입니다.

    5. **Black-Litterman (블랙-리터먼 모델)**
    - 설명: 시장의 기대 수익률과 개인의 전망을 결합하여 최적의 자산 비중을 찾는 방법입니다.
    - 적용 방법:
        - 시장의 기대 수익률(mu)과 공분산 행렬(S)을 기반으로 개인의 전망을 반영한 비중을 계산합니다.
        - 개인의 기대 수익률 조정값을 사용하여 블랙-리터먼 모델을 통해 최적의 비중을 도출합니다.
    """)
