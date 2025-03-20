import streamlit as st
import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
from scipy.stats import norm

st.set_page_config(layout="wide")  # 와이드 레이아웃 설정

# 사이드바에 입력 폼 추가
with st.sidebar:
    st.title("Input Parameters")
    
    # 시장 선택 추가
    market = st.selectbox(
        "시장 선택",
        options=["한국 주식(KRX)", "미국 주식(NYSE/NASDAQ)"],
        index=0
    )
    
    # 시장별 기본값과 예시 설정
    if market == "한국 주식(KRX)":
        default_ticker = "005930"
        ticker_example = "예: 삼성전자(005930), 네이버(035420), 카카오(035720)"
    else:
        default_ticker = "AAPL"
        ticker_example = "예: AAPL(애플), GOOGL(구글), MSFT(마이크로소프트)"
    
    ticker = st.text_input(f"종목 코드 ({ticker_example})", default_ticker)
    
    period_option = st.selectbox(
        "기간 선택:",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y", "Custom"],
        index=3  # 기본값 "1y"
    )
    
    if period_option == "Custom":
        start_date = st.date_input("Start Date", value=date(2010,1,1))
        end_date = st.date_input("End Date", value=date.today())
    else:
        start_date = None
        end_date = None
    
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
    with st.spinner('데이터를 가져오는 중...'):
        try:
            # 종목 코드에서 공백 제거
            ticker = ticker.strip()
            
            # 기간에 따른 시작일 계산
            end_date = date.today()
            if period_option != "Custom":
                if period_option == "1mo":
                    start_date = end_date - pd.DateOffset(months=1)
                elif period_option == "3mo":
                    start_date = end_date - pd.DateOffset(months=3)
                elif period_option == "6mo":
                    start_date = end_date - pd.DateOffset(months=6)
                elif period_option == "1y":
                    start_date = end_date - pd.DateOffset(years=1)
                elif period_option == "2y":
                    start_date = end_date - pd.DateOffset(years=2)
                elif period_option == "5y":
                    start_date = end_date - pd.DateOffset(years=5)
                elif period_option == "10y":
                    start_date = end_date - pd.DateOffset(years=10)
                elif period_option == "20y":
                    start_date = end_date - pd.DateOffset(years=20)
                
                # datetime.date 객체로 변환
                start_date = start_date.date()
            
            # 시장별 종목 코드 검증
            is_valid_ticker = False
            if market == "한국 주식(KRX)":
                # 한국 주식 종목 코드 형식 검증 (6자리 숫자)
                if len(ticker) == 6 and ticker.isdigit():
                    try:
                        stock_info = fdr.StockListing('KRX')
                        if stock_info is None or stock_info.empty:
                            st.error("주식 시장 데이터를 가져올 수 없습니다. 잠시 후 다시 시도해주세요.")
                            st.stop()
                        
                        # Symbol 또는 Code 컬럼 확인
                        if 'Symbol' in stock_info.columns:
                            symbol_column = 'Symbol'
                            name_column = 'Name'
                        elif 'Code' in stock_info.columns:
                            symbol_column = 'Code'
                            name_column = 'Name'
                        else:
                            st.error("주식 시장 데이터 형식이 변경되었습니다.")
                            st.error("관리자에게 문의하세요.")
                            st.stop()
                        
                        # 종목 검색
                        stock_row = stock_info[stock_info[symbol_column] == ticker]
                        if not stock_row.empty:
                            is_valid_ticker = True
                            company_name = stock_row[name_column].iloc[0]
                            st.success(f"종목: {company_name} ({ticker})")
                            
                            # 시가총액과 거래량 정보 표시 (있는 경우)
                            if 'Market Cap' in stock_info.columns:
                                market_cap = stock_row['Market Cap'].iloc[0]
                                st.info(f"시가총액: {market_cap:,.0f}원")
                            if 'Volume' in stock_info.columns:
                                volume = stock_row['Volume'].iloc[0]
                                st.info(f"거래량: {volume:,.0f}주")
                                
                    except Exception as e:
                        st.error("한국 주식 시장 데이터를 가져오는 중 오류가 발생했습니다.")
                        st.error(f"에러 상세: {str(e)}")
                        st.info("인터넷 연결을 확인하고 잠시 후 다시 시도해주세요.")
                        st.stop()
                
                if not is_valid_ticker:
                    st.error(f"'{ticker}'는 유효한 한국 주식 종목 코드가 아닙니다.")
                    st.info("한국 주식 종목 코드는 6자리 숫자입니다.")
                    st.info("예시: 삼성전자(005930), 네이버(035420), 카카오(035720)")
                    st.stop()
                    
            else:  # 미국 주식
                try:
                    # 미국 주식은 심볼 형식이 다양하므로 직접 데이터를 조회해서 확인
                    temp_data = fdr.DataReader(ticker, date.today(), date.today())
                    if temp_data is not None:
                        is_valid_ticker = True
                        st.success(f"종목: {ticker} (US)")
                except Exception:
                    st.error(f"'{ticker}'는 유효한 미국 주식 심볼이 아닙니다.")
                    st.info("대문자로 된 심볼을 입력해주세요.")
                    st.info("예시: AAPL(애플), GOOGL(구글), MSFT(마이크로소프트)")
                    st.stop()
            
            # 주가 데이터 가져오기
            try:
                data = fdr.DataReader(ticker, start_date, end_date)
                
                if data is None or data.empty:
                    st.error(f"{ticker}에 대한 주가 데이터가 없습니다.")
                    st.info("다른 기간을 선택하거나 다른 종목을 시도해보세요.")
                    st.stop()
                    
            except Exception as e:
                st.error("주가 데이터를 가져오는 중 오류가 발생했습니다.")
                st.error(f"에러 상세: {str(e)}")
                st.info("다른 기간을 선택하거나 잠시 후 다시 시도해주세요.")
                st.stop()
                
        except Exception as e:
            st.error(f"예상치 못한 오류가 발생했습니다: {str(e)}")
            st.info("문제가 지속되면 관리자에게 문의해주세요.")
            st.stop()
            data = None

    if data is not None:
        prices = data['Close']
        
        # 기간 정보 표시
        st.sidebar.markdown("---")
        st.sidebar.subheader("선택된 기간 정보")
        date_range = data.index
        st.sidebar.write(f"시작일: {date_range[0].strftime('%Y-%m-%d')}")
        st.sidebar.write(f"종료일: {date_range[-1].strftime('%Y-%m-%d')}")
        st.sidebar.write(f"총 거래일: {len(date_range)}일")
        
        # 주가 통계 정보 표시
        st.sidebar.markdown("---")
        st.sidebar.subheader("주가 통계")
        price_stats = prices.describe()
        st.sidebar.write(f"평균가: ${price_stats['mean']:.2f}")
        st.sidebar.write(f"최저가: ${price_stats['min']:.2f}")
        st.sidebar.write(f"최고가: ${price_stats['max']:.2f}")
        st.sidebar.write(f"표준편차: ${price_stats['std']:.2f}")
        
        # float 변환 시 item() 메서드 호출 전에 타입 체크
        try:
            current_price = prices.iloc[-1]
            current_price = float(current_price.item()) if hasattr(current_price, 'item') else float(current_price)

            price_min = prices.min()
            price_min = float(price_min.item()) if hasattr(price_min, 'item') else float(price_min)
            
            price_max = prices.max()
            price_max = float(price_max.item()) if hasattr(price_max, 'item') else float(price_max)
            
            price_range = price_max - price_min
            margin = 0.2
            x_min = price_min - margin * price_range
            x_max = price_max + margin * price_range
        except Exception as e:
            st.error("주가 데이터 처리 중 오류가 발생했습니다.")
            st.error(f"에러 상세: {str(e)}")
            st.info("다른 종목이나 기간을 선택해보세요.")
            st.stop()

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

with st.expander("양자역학 기반 주가 예측 모델 설명"):
    st.markdown("""
    ### 모델 설명
    이 분석은 슈뢰딩거 방정식을 기반으로 한 양자역학적 접근법을 통해 주가의 확률 분포를 예측합니다.

    #### 주요 기능:
    1. **양자 파동함수 기반 분석**
       - 주가의 움직임을 양자역학적 파동함수로 모델링
       - 불확실성 원리를 활용한 가격 예측 범위 도출

    2. **신뢰구간 분석**
       - 1σ ~ 6σ 범위의 상세한 가격 예측 구간 제공
       - 각 신뢰구간별 발생 확률 계산

    3. **가격 확률 분포**
       - 현재 가격 기준 확률 밀도 함수 생성
       - 미래 가격 분포의 시각화
    """)
