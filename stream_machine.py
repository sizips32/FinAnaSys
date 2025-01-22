import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    mean_absolute_error
)
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
import plotly.express as px
import time

# 페이지 설정
st.set_page_config(page_title="Financial Machine Learning App", layout="wide")
st.title("Financial Machine Learning Analysis")

# 모델 설명 섹션
with st.expander("📚 모델 설명 및 파라미터 가이드", expanded=True):
    st.markdown("""
    ### 🤖 모델 종류별 특징
    
    #### 1. Random Forest
    - **특징**: 여러 개의 의사결정 트리를 생성하여 앙상블하는 모델
    - **장점**: 과적합에 강하고, 특성 중요도를 파악할 수 있음
    - **단점**: 모델이 복잡하고 학습/예측 시간이 김
    
    #### 2. 선형 회귀
    - **특징**: 입력 변수와 출력 변수 간의 선형 관계를 모델링
    - **장점**: 해석이 쉽고 학습이 빠름
    - **단점**: 비선형 관계를 포착하기 어려움
    
    #### 3. LSTM
    - **특징**: 시계열 데이터 분석에 특화된 딥러닝 모델
    - **장점**: 장기 의존성을 잘 포착하고 복잡한 패턴 학습 가능
    - **단점**: 많은 데이터와 계산 자원이 필요
    
    ### 📊 파라미터 설명
    
    #### Random Forest 파라미터
    - **트리 개수**: 생성할 의사결정 트리의 수 (많을수록 안정적이나 느려짐)
    - **최대 깊이**: 각 트리의 최대 깊이 (깊을수록 과적합 위험)
    
    #### 선형 회귀 파라미터
    - **회귀 유형**: 
        - Linear: 기본 선형 회귀
        - Ridge: L2 규제 적용
        - Lasso: L1 규제 적용
    - **알파**: 규제 강도 (높을수록 모델이 단순해짐)
    
    #### LSTM 파라미터
    - **시퀀스 길이**: 예측에 사용할 과거 데이터 기간
    - **LSTM 유닛 수**: 모델의 복잡도 결정
    - **Dropout 비율**: 과적합 방지를 위한 비율
    - **학습률**: 모델 학습 속도 조절
    
    ### 📈 결과 해석 가이드
    
    #### 성능 지표
    - **MSE (Mean Squared Error)**: 예측값과 실제값의 차이를 제곱한 평균
        - 낮을수록 좋음
        - 실제 주가 단위의 제곱
    - **R² Score**: 모델이 설명하는 분산의 비율
        - 1에 가까울수록 좋음
        - 0~1 사이의 값
    
    #### 시각화
    - **특성 중요도**: 각 입력 변수가 예측에 미치는 영향력
    - **학습 곡선**: 모델의 학습 진행 상황
    - **예측 결과**: 실제 가격과 예측 가격 비교
    """)

# 사이드바 파라미터 설정
st.sidebar.header("모델 파라미터 설정")

# 주식 심볼 입력
ticker = st.sidebar.text_input("주식 심볼 입력", "AAPL")

# 날짜 범위 선택
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("시작일", datetime.now() - timedelta(days=365*3))
with col2:
    end_date = st.date_input("종료일", datetime.now())

# 모델 선택
model_type = st.sidebar.selectbox(
    "모델 선택",
    ["Random Forest", "선형 회귀", "LSTM", "XGBoost", "LightGBM"]
)

# 하이퍼파라미터 자동 튜닝 옵션
enable_auto_tuning = st.sidebar.checkbox("하이퍼파라미터 자동 튜닝", value=False)

# Random Forest 파라미터
if model_type == "Random Forest":
    st.sidebar.subheader("Random Forest 파라미터")
    n_estimators = st.sidebar.slider(
        "트리 개수 (n_estimators)", 
        min_value=10, 
        max_value=500, 
        value=100,
        help="더 많은 트리를 사용하면 모델의 안정성이 향상되지만 학습 시간이 증가합니다."
    )
    
    max_depth = st.sidebar.slider(
        "최대 깊이 (max_depth)", 
        min_value=1, 
        max_value=50, 
        value=10,
        help="트리의 최대 깊이를 제한하여 과적합을 방지합니다."
    )

# 선형 회귀 파라미터
elif model_type == "선형 회귀":
    st.sidebar.subheader("선형 회귀 파라미터")
    regression_type = st.sidebar.selectbox(
        "회귀 모델 유형",
        ["Linear", "Ridge", "Lasso"]
    )
    
    if regression_type in ["Ridge", "Lasso"]:
        alpha = st.sidebar.slider(
            "알파 (규제 강도)", 
            min_value=0.0, 
            max_value=10.0, 
            value=1.0,
            help="높은 값은 더 강한 규제를 의미합니다."
        )

# LSTM 파라미터
elif model_type == "LSTM":
    st.sidebar.subheader("LSTM 파라미터")
    sequence_length = st.sidebar.slider(
        "시퀀스 길이", 
        min_value=5, 
        max_value=60, 
        value=30,
        help="예측에 사용할 과거 데이터의 기간"
    )
    
    lstm_units = st.sidebar.slider(
        "LSTM 유닛 수", 
        min_value=32, 
        max_value=256, 
        value=128,
        help="모델의 복잡도를 결정합니다."
    )
    
    dropout_rate = st.sidebar.slider(
        "Dropout 비율", 
        min_value=0.0, 
        max_value=0.5, 
        value=0.2,
        help="과적합 방지를 위한 dropout 비율"
    )
    
    learning_rate = st.sidebar.slider(
        "학습률", 
        min_value=0.0001, 
        max_value=0.01, 
        value=0.001,
        format="%.4f",
        help="모델의 학습 속도를 조절합니다."
    )

# 공통 파라미터
test_size = st.sidebar.slider(
    "테스트 데이터 비율", 
    min_value=0.1, 
    max_value=0.4, 
    value=0.2,
    help="전체 데이터 중 테스트에 사용할 비율을 설정합니다."
)

# 데이터 다운로드 함수
@st.cache_data
def get_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            st.error(f"'{ticker}' 심볼에 대한 데이터를 찾을 수 없습니다. 올바른 주식 심볼인지 확인해주세요.")
            return None
        data.reset_index(inplace=True)
        st.success(f"{ticker} 데이터를 성공적으로 다운로드했습니다. ({len(data)} 개의 데이터 포인트)")
        return data
    except Exception as e:
        st.error(f"데이터 다운로드 중 오류 발생: {str(e)}")
        st.info("다음을 확인해주세요:\n- 인터넷 연결 상태\n- 주식 심볼의 정확성\n- 선택한 날짜 범위의 유효성")
        return None

def validate_parameters(model_type, **params):
    """모델 파라미터 유효성 검사"""
    try:
        if model_type == "Random Forest":
            if params.get('n_estimators', 0) < 10:
                st.warning(
                    "트리 개수가 너무 적습니다. "
                    "최소 10개 이상을 권장합니다."
                )
            if params.get('max_depth', 0) > 30:
                st.warning(
                    "트리 깊이가 깊습니다. "
                    "과적합의 위험이 있습니다."
                )
        
        elif model_type == "LSTM":
            if params.get('sequence_length', 0) < 10:
                st.warning(
                    "시퀀스 길이가 너무 짧습니다. "
                    "예측 정확도가 낮을 수 있습니다."
                )
            if params.get('dropout_rate', 0) > 0.5:
                st.warning(
                    "Dropout 비율이 높습니다. "
                    "학습이 불안정할 수 있습니다."
                )
        
        return True
    except Exception as e:
        st.error(f"파라미터 검증 중 오류 발생: {str(e)}")
        return False

# LSTM 데이터 준비 함수
def prepare_lstm_data(data, sequence_length):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length), 0])
        y.append(scaled_data[i + sequence_length, 0])
    
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def get_model_and_params(model_type):
    """모델과 하이퍼파라미터 그리드 반환"""
    if model_type == "Random Forest":
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
    elif model_type == "XGBoost":
        model = XGBRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    elif model_type == "LightGBM":
        model = LGBMRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'num_leaves': [31, 62, 127],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    else:
        return None, None
    
    return model, param_grid

def auto_tune_model(model, param_grid, X_train, y_train):
    """GridSearchCV를 사용한 하이퍼파라미터 튜닝"""
    with st.spinner("하이퍼파라미터 튜닝 중..."):
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='neg_mean_squared_error'
        )
        grid_search.fit(X_train, y_train)
        
        st.success("최적의 하이퍼파라미터를 찾았습니다!")
        st.write("최적 파라미터:", grid_search.best_params_)
        st.write("최적 점수:", -grid_search.best_score_)
        
        return grid_search.best_estimator_

def plot_feature_importance(model, feature_names):
    """특성 중요도 시각화"""
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(
            importances,
            x='feature',
            y='importance',
            title='특성 중요도'
        )
        st.plotly_chart(fig)
    else:
        st.info("이 모델은 특성 중요도를 제공하지 않습니다.")

def plot_learning_curves(history):
    """학습 곡선 시각화"""
    if history is not None and hasattr(history, 'history'):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=history.history['loss'],
                name='Train Loss'
            )
        )
        if 'val_loss' in history.history:
            fig.add_trace(
                go.Scatter(
                    y=history.history['val_loss'],
                    name='Validation Loss'
                )
            )
        fig.update_layout(
            title='학습 곡선',
            xaxis_title='Epoch',
            yaxis_title='Loss'
        )
        st.plotly_chart(fig)

def evaluate_model(model, X_test, y_test, scaler=None):
    """모델 성능 평가"""
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    # 성능 지표 계산
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2 Score': r2_score(y_test, y_pred),
        '추론 시간': f"{inference_time:.4f}초"
    }
    
    # 성능 지표 표시
    st.subheader("📊 모델 성능 평가")
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ['값']
    st.table(metrics_df)
    
    # 예측 vs 실제 그래프
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='예측 vs 실제',
            marker=dict(color='blue', opacity=0.5)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='이상적인 예측',
            line=dict(color='red', dash='dash')
        )
    )
    fig.update_layout(
        title='예측 vs 실제 값 비교',
        xaxis_title='실제 값',
        yaxis_title='예측 값'
    )
    st.plotly_chart(fig)
    
    return metrics
# 메인 프로세스
if st.sidebar.button("분석 시작"):
    # 변수 초기화
    metrics = None
    n_estimators = None
    max_depth = None
    sequence_length = None
    dropout_rate = None
    lstm_units = None
    learning_rate = None
    
    # 모델별 파라미터 설정
    if model_type == "Random Forest":
        n_estimators = st.sidebar.slider(
            "트리 개수 (n_estimators)",
            min_value=10,
            max_value=500,
            value=100
        )
        max_depth = st.sidebar.slider(
            "최대 깊이 (max_depth)",
            min_value=1,
            max_value=50,
            value=10
        )
    elif model_type == "LSTM":
        sequence_length = st.sidebar.slider(
            "시퀀스 길이",
            min_value=5,
            max_value=60,
            value=30
        )
        lstm_units = st.sidebar.slider(
            "LSTM 유닛 수",
            min_value=32,
            max_value=256,
            value=128
        )
        dropout_rate = st.sidebar.slider(
            "Dropout 비율",
            min_value=0.0,
            max_value=0.5,
            value=0.2
        )
        learning_rate = st.sidebar.slider(
            "학습률",
            min_value=0.0001,
            max_value=0.01,
            value=0.001,
            format="%.4f"
        )
    
    with st.spinner("주식 데이터를 다운로드 중입니다..."):
        stock_data = get_stock_data(ticker, start_date, end_date)
    
    if stock_data is not None:
        try:
            # 데이터 유효성 검사
            if len(stock_data) < 100:
                st.warning(
                    "데이터 포인트가 적습니다. "
                    "분석 결과의 신뢰도가 낮을 수 있습니다."
                )
            
            # 파라미터 초기화
            model_params = {}
            if model_type == "Random Forest":
                model_params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth
                }
            elif model_type == "LSTM":
                model_params = {
                    'sequence_length': sequence_length,
                    'dropout_rate': dropout_rate,
                    'lstm_units': lstm_units,
                    'learning_rate': learning_rate
                }
            
            # 파라미터 유효성 검사
            if not validate_parameters(model_type, **model_params):
                st.stop()
            
            # 데이터 시각화
            st.subheader("주가 차트")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=stock_data['Date'],
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close']
            ))
            st.plotly_chart(fig)

            if model_type in ["Random Forest", "선형 회귀"]:
                # 데이터 전처리
                stock_data['Return'] = stock_data['Close'].pct_change()
                stock_data.dropna(inplace=True)
                
                X = stock_data[['Open', 'High', 'Low', 'Volume', 'Return']]
                y = stock_data['Close']
                
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42
                )

                if model_type == "Random Forest":
                    model, param_grid = get_model_and_params(model_type)
                    
                    if enable_auto_tuning and param_grid is not None:
                        model = auto_tune_model(model, param_grid, X_train, y_train)
                    else:
                        model.fit(X_train, y_train)
                else:  # 선형 회귀
                    with st.spinner("선형 회귀 모델을 학습 중입니다..."):
                        if regression_type == "Linear":
                            model = LinearRegression()
                        elif regression_type == "Ridge":
                            model = Ridge(alpha=alpha)
                        else:  # Lasso
                            model = Lasso(alpha=alpha)
                
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                # 성능 평가
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                
                st.subheader("모델 성능")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Squared Error", f"{mse:.2f}")
                with col2:
                    st.metric("R² Score", f"{r2:.2f}")

                if model_type == "Random Forest":
                    # 특성 중요도 시각화
                    st.subheader("🎯 특성 중요도")
                    plot_feature_importance(model, X.columns)

            else:  # LSTM
                with st.spinner("LSTM 모델을 학습 중입니다..."):
                    # LSTM 데이터 준비
                    X, y, scaler = prepare_lstm_data(stock_data, model_params['sequence_length'])
                    
                    # 데이터 분할
                    train_size = int(len(X) * (1 - test_size))
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    # LSTM 모델 구성
                    model = Sequential([
                        LSTM(
                            model_params['lstm_units'],
                            return_sequences=True,
                            input_shape=(model_params['sequence_length'], 1)
                        ),
                        Dropout(model_params['dropout_rate']),
                        LSTM(model_params['lstm_units']//2),
                        Dropout(model_params['dropout_rate']),
                        Dense(1)
                    ])
                    
                    model.compile(
                        optimizer=Adam(
                            learning_rate=model_params['learning_rate']
                        ),
                        loss='mse'
                    )
                    
                    # 모델 학습
                    history = model.fit(
                        X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.1,
                        verbose=0
                    )
                    
                    # 예측
                    predictions = model.predict(X_test)
                    
                    # 성능 평가
                    mse = mean_squared_error(y_test, predictions)
                    
                    # 결과 표시
                    st.subheader("LSTM 모델 성능")
                    st.metric("Mean Squared Error", f"{mse:.6f}")
                    
                    # 학습 곡선 시각화
                    st.subheader("📈 학습 곡선")
                    plot_learning_curves(history)
                    
                    # 예측 결과 시각화
                    st.subheader("예측 결과")
                    
                    # 스케일 복원
                    predictions_unscaled = scaler.inverse_transform(predictions)
                    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=y_test_unscaled.flatten(),
                        name='실제 가격'
                    ))
                    fig.add_trace(go.Scatter(
                        y=predictions_unscaled.flatten(),
                        name='예측 가격'
                    ))
                    fig.update_layout(
                        title="가격 예측 결과",
                        xaxis_title="시간",
                        yaxis_title="가격"
                    )
                    st.plotly_chart(fig)

            # 모델 모니터링 섹션
            if metrics is not None:
                st.sidebar.markdown("---")
                st.sidebar.subheader("📊 모델 모니터링")
                
                if 'metrics_history' not in st.session_state:
                    st.session_state.metrics_history = []
                
                st.session_state.metrics_history.append({
                    'timestamp': datetime.now(),
                    'model_type': model_type,
                    **metrics
                })
                
                # 성능 지표 히스토리 표시
                if st.session_state.metrics_history:
                    history_df = pd.DataFrame(
                        st.session_state.metrics_history
                    )
                    st.sidebar.write("최근 실험 결과:")
                    st.sidebar.dataframe(
                        history_df[[
                            'timestamp',
                            'model_type',
                            'R2 Score',
                            'RMSE'
                        ]]
                    )

        except Exception as e:
            st.error(
                f"분석 중 오류가 발생했습니다: {str(e)}"
            )
            st.info(
                "문제가 지속되면 다음을 시도해보세요:\n"
                "- 페이지 새로고침\n"
                "- 다른 주식 심볼 선택\n"
                "- 날짜 범위 조정"
            )

    else:
        st.error(
            "데이터를 불러오는데 실패했습니다. "
            "티커 심볼을 확인해주세요."
        ) 

