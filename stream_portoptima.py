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
end_date = st.date_input("End date", value=pd.to_datetime("2024-10-31"))

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
                           end=end_date)['Adj Close']
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
                    styled_weights_df = weights_df.style.map(
                        lambda x: 'color: red; font-weight: bold;' if x in top_weights['Weight'].values else '',
                        subset=top_weights['Asset'].tolist()
                    )

                    st.write(styled_weights_df)  # 스타일이 적용된 데이터프레임 표시

    except Exception as e:  # 들여쓰기 수정
        st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")
