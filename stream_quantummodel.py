import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

st.set_page_config(layout="wide")  # 와이드 레이아웃 설정

# 사이드바에 입력 폼 추가
with st.sidebar:
    st.title("Input Parameters")
    ticker = st.text_input("Stock Ticker Symbol (e.g., AAPL, TSLA):", "AAPL")
    period_option = st.selectbox(
        "Time Period:",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y", "Custom"],
        index=3  # 기본값 "1y"
    )
    
    if period_option == "Custom":
        start_date = st.date_input("Start Date", value=date(2010,1,1))
        end_date = st.date_input("End Date", value=date.today())
    else:
        start_date, end_date = None, None
    
    st.subheader("Quantum Simulation Parameters")
    hbar = st.slider("Planck Constant (ℏ)", 0.1, 2.0, 1.0, 0.1)
    st.caption("ℏ: 양자역학적 기본 상수로, '에너지-시간'과 '운동량-위치' 사이의 상관성을 결정합니다. 값이 커지면 파동함수의 진동 특성이 변하며, 가격 분포 변화의 예민도가 달라집니다.")
    
    m = st.slider("Mass Parameter (m)", 0.1, 2.0, 1.0, 0.1)
    st.caption("m: '질량'에 해당하는 매개변수로, 가격 변화를 입자의 운동에 비유했을 때 질량은 가격 변동의 민감도를 좌우합니다. 값이 클수록 변화가 완만해지고, 작을수록 급격한 변화가 가능해집니다.")
    
    dt = st.slider("Time Step (dt)", 0.001, 0.1, 0.01, 0.001)
    st.caption("dt: 시뮬레이션의 한 단계 시간 간격입니다. 값이 작을수록 더 많은 연산이 필요하지만 더 정교한 결과를 얻을 수 있고, 값이 크면 계산 속도는 빠르지만 해석 정확도가 낮아질 수 있습니다.")
    
    M = st.slider("Number of Time Steps", 50, 200, 100, 10)
    st.caption("M: 전체 시간 스텝의 개수로, 시뮬레이션 진행 전체 기간을 결정합니다. 많은 스텝을 사용할수록 파동함수가 더 오래 진화하고, 이에 따른 미래 예측 분포가 더욱 넓게 탐색될 수 있습니다.")
    
    calculate = st.button("Calculate Price Distribution")

st.title("Quantum Finance: PDE-based Stock Price Distribution Prediction")

if calculate:
    with st.spinner('Downloading and processing data...'):
        if period_option == "Custom" and start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date)
        else:
            data = yf.download(ticker, period=period_option)
        
        if len(data) == 0:
            st.error("Could not fetch data. Please check the ticker symbol and try again.")
        else:
            prices = data['Close']
            current_price = float(prices.iloc[-1].item())

            price_min = float(prices.min().item())
            price_max = float(prices.max().item())
            price_range = price_max - price_min
            margin = 0.2
            x_min = price_min - margin * price_range
            x_max = price_max + margin * price_range
            
            log_returns = np.log(prices / prices.shift(1)).dropna()
            sigma = float(log_returns.std())
            if np.isclose(sigma, 0) or np.isnan(sigma):
                sigma = 0.01
            mu = float(log_returns.mean())

            N = 200
            x = np.linspace(x_min, x_max, N)
            dx = x[1] - x[0]

            psi = np.zeros(N)
            for price_val in prices.values:
                psi += np.exp(-(x - price_val)**2/(4*sigma**2))

            psi = psi / np.sqrt(np.sum(np.abs(psi)**2)*dx)
            psi_t = psi.copy()

            price_hist, bins = np.histogram(prices, bins=50, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            V = np.interp(x, bin_centers, -np.log(price_hist + 1e-10))
            V = V / np.max(np.abs(V))

            coeff = -(hbar**2)/(2*m*dx**2)
            diag = 2*coeff + V
            off = -coeff
            i_factor = 1j*dt/(2*hbar)

            aA = -i_factor * off * np.ones(N, dtype=complex)
            bA = 1 + i_factor * diag
            cA = -i_factor * off * np.ones(N, dtype=complex)
            aA[0] = 0; cA[-1] = 0

            aB = i_factor * off * np.ones(N, dtype=complex)
            bB = 1 - i_factor * diag
            cB = i_factor * off * np.ones(N, dtype=complex)
            aB[0] = 0; cB[-1] = 0

            def tridiag_solve(a, b, c, d):
                n = len(d)
                c_ = np.zeros(n, dtype=complex)
                d_ = np.zeros(n, dtype=complex)
                x_ = np.zeros(n, dtype=complex)

                c_[0] = c[0]/b[0]
                d_[0] = d[0]/b[0]

                for i in range(1, n):
                    denom = b[i] - a[i]*c_[i-1]
                    if denom == 0:
                        denom = 1e-12
                    c_[i] = c[i]/denom if i < n-1 else 0
                    d_[i] = (d[i]-a[i]*d_[i-1])/denom

                x_[n-1] = d_[n-1]
                for i in range(n-2, -1, -1):
                    x_[i] = d_[i] - c_[i]*x_[i+1]
                return x_

            for _ in range(M):
                d = np.zeros(N, dtype=complex)
                for i in range(N):
                    val = bB[i]*psi_t[i]
                    if i > 0:
                        val += aB[i]*psi_t[i-1]
                    if i < N-1:
                        val += cB[i]*psi_t[i+1]
                    d[i] = val

                psi_new = tridiag_solve(aA, bA, cA, d)
                norm = np.sqrt(np.sum(np.abs(psi_new)**2)*dx)
                if norm != 0:
                    psi_t = psi_new / norm
                else:
                    psi_t = psi_new

            prob_density = np.abs(psi_t)**2
            mean_future = np.sum(x * prob_density * dx)
            var_future = np.sum((x - mean_future)**2 * prob_density * dx)
            std_future = np.sqrt(var_future)

            # σ 구간 계산
            pred_1sigma_min = mean_future - std_future
            pred_1sigma_max = mean_future + std_future
            pred_2sigma_min = mean_future - 2*std_future
            pred_2sigma_max = mean_future + 2*std_future
            pred_3sigma_min = mean_future - 3*std_future
            pred_3sigma_max = mean_future + 3*std_future
            pred_4sigma_min = mean_future - 4*std_future
            pred_4sigma_max = mean_future + 4*std_future
            pred_5sigma_min = mean_future - 5*std_future
            pred_5sigma_max = mean_future + 5*std_future
            pred_6sigma_min = mean_future - 6*std_future
            pred_6sigma_max = mean_future + 6*std_future

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Price Statistics")
                st.metric("Current Price", f"${current_price:.2f}")
                st.metric("Expected Future Price", f"${mean_future:.2f}")
                st.metric("Price Standard Deviation", f"${std_future:.2f}")

                st.subheader("Confidence Intervals")
                st.write(f"1σ Range: ${pred_1sigma_min:.2f} - ${pred_1sigma_max:.2f}")
                st.write(f"2σ Range: ${pred_2sigma_min:.2f} - ${pred_2sigma_max:.2f}")
                st.write(f"3σ Range: ${pred_3sigma_min:.2f} - ${pred_3sigma_max:.2f}")
                st.write(f"4σ Range: ${pred_4sigma_min:.2f} - ${pred_4sigma_max:.2f}")
                st.write(f"5σ Range: ${pred_5sigma_min:.2f} - ${pred_5sigma_max:.2f}")
                st.write(f"6σ Range: ${pred_6sigma_min:.2f} - ${pred_6sigma_max:.2f}")

            with col2:
                st.subheader("Historical Data")
                st.line_chart(data['Close'])

            st.subheader("Price Distribution Analysis")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            ax1.hist(prices, bins=50, density=True, alpha=0.5, color='blue', label='Historical Price Distribution')
            ax1.set_title(f"{ticker} - Historical Price Distribution")
            ax1.set_xlabel("Price")
            ax1.set_ylabel("Density")
            ax1.legend()

            ax2.plot(x, prob_density, label='Quantum Probability Density')
            ax2.axvspan(pred_1sigma_min, pred_1sigma_max, color='orange', alpha=0.2, label='1σ Range')
            ax2.axvspan(pred_2sigma_min, pred_2sigma_max, color='green', alpha=0.1, label='2σ Range')
            ax2.axvspan(pred_3sigma_min, pred_3sigma_max, color='red', alpha=0.05, label='3σ Range')
            ax2.axvspan(pred_4sigma_min, pred_4sigma_max, color='purple', alpha=0.03, label='4σ Range')
            ax2.axvspan(pred_5sigma_min, pred_5sigma_max, color='brown', alpha=0.02, label='5σ Range')
            ax2.axvspan(pred_6sigma_min, pred_6sigma_max, color='gray', alpha=0.01, label='6σ Range')

            ax2.axvline(x=current_price, color='red', linestyle='--', label='Current Price')
            ax2.set_title(f"{ticker} - PDE-based Price Distribution Prediction")
            ax2.set_xlabel("Price")
            ax2.set_ylabel("Probability Density")
            ax2.legend()

            plt.tight_layout()
            st.pyplot(fig)

            sigma_data = [
                ("±1σ", "약 68.27%", "1/3", "3일 중 하루"),
                ("±2σ", "약 95.45%", "1/22", "3주 중 하루"),
                ("±3σ", "약 99.73%", "1/370", "1년 중 하루"),
                ("±4σ", "약 99.9937%", "1/15,787", "60년(평생) 중 하루"),
                ("±5σ", "약 99.99994%", "1/1,744,278", "5000년(역사시대) 중 하루"),
                ("±6σ", "약 99.9999998%", "1/506,842,372", "150만년 중 하루 (유인원 출현 이전 이래)")
            ]
            df_sigma = pd.DataFrame(sigma_data, columns=["범위", "차지하는 비율", "벗어날 확률(개략)", "비유적 표현"])
            st.table(df_sigma)
else:
    st.info("Please enter parameters in the sidebar and click 'Calculate Price Distribution' to start the analysis.")

""" 
#### quantum simulation parameters`는 주가 흐름을 예측하는 데 중요한 역할을 합니다. 각 매개변수가 주가 흐름에 미치는 영향을 살펴보겠습니다. 

1. Planck Constant (ℏ)
영향: ℏ는 양자역학에서 에너지와 시간, 운동량과 위치 간의 관계를 결정하는 상수입니다. 이 값이 클수록 파동함수의 진동 특성이 강해져 가격 분포의 변화에 대한 민감도가 증가합니다. 즉, 주가의 변동성이 커질 수 있습니다.
시뮬레이션을 통해 알 수 있는 것: ℏ의 값을 조정하여 주가의 변동성을 어떻게 변화시킬 수 있는지, 그리고 그에 따른 가격 예측의 범위를 확인할 수 있습니다.

2. Mass Parameter (m)
영향: 질량 매개변수는 가격 변화를 입자의 운동에 비유할 때, 가격 변동의 민감도를 조절합니다. 값이 클수록 가격 변화가 완만해지고, 작을수록 급격한 변화가 가능해집니다.
시뮬레이션을 통해 알 수 있는 것: m의 값을 조정하여 주가의 안정성과 변동성을 어떻게 조절할 수 있는지, 그리고 특정 상황에서의 가격 예측의 신뢰도를 평가할 수 있습니다.

3. Time Step (dt)
영향: dt는 시뮬레이션의 시간 간격을 결정합니다. 값이 작을수록 더 많은 계산이 필요하지만 더 정교한 결과를 얻을 수 있습니다. 반면, 값이 크면 계산 속도는 빨라지지만 해석의 정확도가 떨어질 수 있습니다.
시뮬레이션을 통해 알 수 있는 것: dt의 크기를 조정하여 시뮬레이션의 정밀도와 속도 간의 균형을 맞추고, 주가 예측의 정확성을 평가할 수 있습니다.

4. Number of Time Steps (M)
영향: M은 전체 시뮬레이션 기간 동안의 시간 스텝 수를 결정합니다. 많은 스텝을 사용할수록 파동함수가 더 오래 진화하고, 이에 따른 미래 예측 분포가 더욱 넓게 탐색될 수 있습니다.
시뮬레이션을 통해 알 수 있는 것: M의 값을 조정하여 시뮬레이션의 기간을 늘리거나 줄여서 주가 예측의 장기적인 경향을 분석할 수 있습니다.
결론
이러한 매개변수들은 주가 예측의 민감도와 정확성에 큰 영향을 미칩니다. 시뮬레이션을 통해 각 매개변수를 조정하면서 주가의 변동성과 예측 범위를 분석할 수 있으며, 이를 통해 투자 전략을 세우는 데 유용한 정보를 얻을 수 있습니다.

#### 주가의 변동성과 예측 범위를 분석하고 이를 통해 투자 전략을 세우는 방법에 대한 예시를 들어 설명하겠습니다.
예시: 주식 투자 전략 수립
1. 변동성 분석
시뮬레이션 결과: 예를 들어, 시뮬레이션을 통해 특정 주식의 ℏ 값을 1.5로 설정했을 때, 주가의 변동성이 높아지는 것을 확인했다고 가정해봅시다. 이 경우, 주가는 급격하게 오르거나 내릴 가능성이 크다는 것을 의미합니다.
투자 전략: 변동성이 높은 주식은 단기 투자에 적합할 수 있습니다. 따라서, 투자자는 이 주식을 단기 매매 전략으로 접근할 수 있습니다. 예를 들어, 주가가 급등할 때 매도하고, 급락할 때 매수하는 전략을 사용할 수 있습니다.
2. 예측 범위 분석
시뮬레이션 결과: 시뮬레이션을 통해 주가의 1σ, 2σ 범위를 계산했을 때, 주가가 $100에서 $120 사이에 있을 확률이 68%라는 결과가 나왔다고 가정해봅시다. 이는 주가가 이 범위 내에서 움직일 가능성이 높다는 것을 의미합니다.
투자 전략: 이 정보를 바탕으로 투자자는 주가가 $100 이하로 떨어질 경우 매수하고, $120 이상으로 오를 경우 매도하는 전략을 세울 수 있습니다. 즉, 예측 범위를 활용하여 매수 및 매도 시점을 결정할 수 있습니다.
3. 리스크 관리
시뮬레이션 결과: 시뮬레이션을 통해 주가의 표준편차가 높아져 리스크가 증가하는 것을 확인했다고 가정해봅시다.
투자 전략: 리스크가 높다는 것은 손실 가능성도 크다는 의미입니다. 따라서, 투자자는 포트폴리오의 일부를 안전 자산(예: 채권)으로 분산 투자하거나, 손절매(Stop-Loss) 주문을 설정하여 손실을 최소화하는 전략을 사용할 수 있습니다.
결론
이와 같이 시뮬레이션을 통해 주가의 변동성과 예측 범위를 분석하면, 투자자는 보다 정보에 기반한 결정을 내릴 수 있습니다. 변동성이 높은 주식은 단기 매매 전략에 적합하고, 예측 범위를 활용하여 매수 및 매도 시점을 결정하며, 리스크 관리 전략을 통해 손실을 최소화할 수 있습니다. 이러한 접근은 투자 성과를 향상시키는 데 큰 도움이 됩니다.

"""
