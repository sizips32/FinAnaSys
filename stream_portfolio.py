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
    "IONQ, JOBY, RXRX, SMR"
)
start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End date", value=pd.to_datetime("2024-12-31"))

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
                st.markdown(
                    "<h3 style='color: white;'>Optimization Results</h3>",
                    unsafe_allow_html=True
                )
                for method, weights in results.items():
                    st.markdown(
                        f"<span style='color:white; font-size:24px; font-weight:bold; "
                        "text-decoration: underline;'>{method} Asset Allocation:</span>",
                        unsafe_allow_html=True
                    )

                    weights_df = pd.DataFrame(
                        list(weights.items()),
                        columns=['Asset', 'Weight']
                    )
                    weights_df['Weight'] = weights_df['Weight'] * 100

                    # 상위 비중 2개 찾기
                    top_weights = weights_df.nlargest(2, 'Weight')
                    
                    # HTML 스타일링을 사용하여 테이블 생성
                    html_table = "<table style='width:100%'><tr>"
                    for asset in weights_df['Asset']:
                        weight = weights_df[weights_df['Asset'] == asset]['Weight'].values[0]
                        style = "color:red;font-weight:bold;" if asset in top_weights['Asset'].values else ""
                        html_table += f"<th style='text-align:center'>{asset}</th>"
                    html_table += "</tr><tr>"
                    for asset in weights_df['Asset']:
                        weight = weights_df[weights_df['Asset'] == asset]['Weight'].values[0]
                        style = "color:red;font-weight:bold;" if asset in top_weights['Asset'].values else ""
                        html_table += f"<td style='text-align:center;{style}'>{weight:.2f}%</td>"
                    html_table += "</tr></table>"
                    
                    st.markdown(html_table, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")

# 설명 섹션을 마크다운 형식의 expander로 변경
with st.expander("📊 포트폴리오 투자 전략 가이드"):
    st.markdown("""
    ### 1️⃣ Equal Weight (동일 비중) 전략
    - **설명**: 모든 자산에 동일한 비중을 할당합니다. 간단하고 직관적이며, 분산 투자의 기본 원칙을 따릅니다.
    - **사용 시점** 💡
        - 시장에 대한 특별한 견해가 없을 때
        - 단순하고 투명한 전략이 필요할 때
        - 장기 투자를 계획할 때
    
    ### 2️⃣ Maximum Sharpe Ratio (최대 샤프 비율) 전략
    - **설명**: 위험 대비 수익률을 최적화하여 가장 효율적인 포트폴리오를 구성합니다.
    - **사용 시점** 💡
        - 위험 조정 수익률을 최대화하고 싶을 때
        - 시장이 안정적이고 예측 가능할 때
        - 과거 데이터의 신뢰도가 높을 때
    
    ### 3️⃣ Minimum Volatility (최소 변동성) 전략
    - **설명**: 포트폴리오의 변동성을 최소화하여 안정적인 수익을 추구합니다.
    - **사용 시점** 💡
        - 시장 변동성이 높을 때
        - 보수적인 투자 성향일 때
        - 원금 보존이 중요할 때
    
    ### 4️⃣ Risk Parity (위험 균형) 전략
    - **설명**: 각 자산의 위험 기여도를 균등하게 분배하여 균형 잡힌 포트폴리오를 구성합니다.
    - **사용 시점** 💡
        - 다양한 자산 클래스에 투자할 때
        - 위험 분산이 중요할 때
        - 시장 환경이 불확실할 때
    
    ### 5️⃣ Black-Litterman (블랙-리터먼 모델) 전략
    - **설명**: 시장 균형과 투자자의 개별 전망을 결합하여 최적의 포트폴리오를 구성합니다.
    - **사용 시점** 💡
        - 특정 자산에 대한 강한 견해가 있을 때
        - 시장 전망과 개인 견해를 결합하고 싶을 때
        - 전문적인 포트폴리오 관리가 필요할 때
    
    > 💡 **팁**: 투자 목적, 위험 성향, 투자 기간에 따라 적절한 전략을 선택하거나 복합적으로 활용하세요.
    """)
