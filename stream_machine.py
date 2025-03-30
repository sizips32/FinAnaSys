import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import openai
from typing import Dict, Any

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

# 상수 정의
US_STOCKS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'TSLA': 'Tesla Inc.',
    'NVDA': 'NVIDIA Corporation'
}

# TensorFlow 가용성 확인
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow를 찾을 수 없습니다. LSTM 모델은 사용할 수 없습니다.")

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# LightGBM 가용성 확인
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    st.warning("LightGBM을 사용할 수 없습니다. 설치하려면: pip install lightgbm")

# XGBoost 가용성 확인
try:
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost를 사용할 수 없습니다. 설치하려면: pip install xgboost")

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

ticker = st.sidebar.text_input("주식 심볼 입력", "AAPL")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("시작일", datetime.now() - timedelta(days=365*3))
with col2:
    end_date = st.date_input("종료일", datetime.now())

model_type = st.sidebar.selectbox(
    "모델 선택",
    ["Random Forest", "선형 회귀", "LSTM", "XGBoost", "LightGBM"]
)

enable_auto_tuning = st.sidebar.checkbox("하이퍼파라미터 자동 튜닝", value=False)

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

test_size = st.sidebar.slider(
    "테스트 데이터 비율", 
    min_value=0.1, 
    max_value=0.4, 
    value=0.2,
    help="전체 데이터 중 테스트에 사용할 비율을 설정합니다."
)

@st.cache_data(ttl=3600)
def get_stock_data(ticker, start, end):
    """주식 데이터를 가져오는 함수"""
    try:
        if ticker.upper() in US_STOCKS:
            try:
                df = fdr.DataReader(ticker, start, end)
                if df.empty:
                    st.error(f"{ticker}에 대한 주가 데이터가 없습니다.")
                    if ticker.upper() in US_STOCKS:
                        st.info(
                            f"'{US_STOCKS[ticker.upper()]}' "
                            f"({ticker.upper()})의 데이터를 찾을 수 없습니다."
                        )
                    else:
                        st.info("미국 주식 심볼을 확인하고 다시 시도해주세요.")
                    return None

                stock_name = US_STOCKS.get(ticker.upper(), ticker.upper())
                st.success(
                    f"미국 주식 '{stock_name}' ({ticker.upper()}) "
                    "데이터를 성공적으로 불러왔습니다."
                )

            except Exception as us_error:
                st.error(
                    f"미국 주식 데이터를 가져오는데 실패했습니다: {str(us_error)}"
                )
                return None
        else:
            try:
                # KRX 데이터 가져오기 (최대 3번 재시도)
                max_retries = 3
                retry_count = 0
                stock_info = None
                
                while retry_count < max_retries:
                    try:
                        stock_info = fdr.StockListing('KRX')
                        break
                    except Exception as retry_error:
                        retry_count += 1
                        if retry_count == max_retries:
                            st.error("KRX 데이터를 가져오는데 실패했습니다.")
                            st.info(f"오류 내용: {str(retry_error)}")
                            return None
                        time.sleep(1)
                
                if stock_info is None or stock_info.empty:
                    st.error("KRX 종목 정보가 비어있습니다.")
                    st.info("잠시 후 다시 시도해주세요.")
                    return None
                
                if 'Symbol' not in stock_info.columns or 'Name' not in stock_info.columns:
                    available_columns = ', '.join(stock_info.columns)
                    st.error("KRX 데이터에서 필수 정보(Symbol/Name)를 찾을 수 없습니다.")
                    st.info(f"사용 가능한 컬럼: {available_columns}")
                    return None
                
                # 종목 코드로 검색
                matching_stocks = stock_info[stock_info['Symbol'] == ticker]
                if matching_stocks.empty:
                    # 종목명으로 검색 시도
                    name_match = stock_info[stock_info['Name'].str.contains(ticker, case=False, na=False)]
                    if not name_match.empty:
                        st.warning(f"입력하신 '{ticker}'는 종목명입니다. 다음 종목들을 찾았습니다:")
                        for _, row in name_match.iterrows():
                            st.write(f"- {row['Name']}: {row['Symbol']}")
                        return None
                    else:
                        st.error(f"'{ticker}'에 해당하는 한국 주식을 찾을 수 없습니다.")
                        st.info("올바른 종목 코드를 입력해주세요. (예: 삼성전자 - 005930)")
                        return None
                
                stock_name = matching_stocks['Name'].iloc[0]
                
                # 주가 데이터 가져오기
                df = fdr.DataReader(ticker, start, end)
                if df is None or df.empty:
                    st.error(f"{stock_name} ({ticker})에 대한 주가 데이터가 없습니다.")
                    st.info("종목 코드와 날짜를 확인하고 다시 시도해주세요.")
                    return None
                
                st.success(f"한국 주식 '{stock_name}' ({ticker}) 데이터를 성공적으로 불러왔습니다.")
                
            except Exception as kr_error:
                st.error(f"한국 주식 데이터를 가져오는데 실패했습니다: {str(kr_error)}")
                st.info("종목 코드를 확인하고 다시 시도해주세요.")
                return None
        
        # 필요한 컬럼 존재 여부 확인
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"필수 데이터가 누락되었습니다: {', '.join(missing_columns)}")
            return None
        
        # 필요한 컬럼만 선택하고 결측치 처리
        df = df[required_columns].fillna(method='ffill').fillna(method='bfill')
        return df
        
    except Exception as e:
        st.error(f"예상치 못한 오류가 발생했습니다: {str(e)}")
        st.info("잠시 후 다시 시도해주세요.")
        return None

def validate_parameters(model_type, **params):
    try:
        if model_type == "Random Forest":
            if params.get('n_estimators', 0) < 10:
                st.warning("트리 개수가 너무 적습니다. 최소 10개 이상을 권장합니다.")
            if params.get('max_depth', 0) > 30:
                st.warning("트리 깊이가 깊습니다. 과적합의 위험이 있습니다.")
        elif model_type == "LSTM":
            if params.get('sequence_length', 0) < 10:
                st.warning("시퀀스 길이가 너무 짧습니다. 예측 정확도가 낮을 수 있습니다.")
            if params.get('dropout_rate', 0) > 0.5:
                st.warning("Dropout 비율이 높습니다. 학습이 불안정할 수 있습니다.")
        return True
    except Exception as e:
        st.error(f"파라미터 검증 중 오류 발생: {str(e)}")
        return False

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
    if model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
    elif model_type == "XGBoost":
        model = XGBClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    elif model_type == "LightGBM":
        model = LGBMClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'num_leaves': [31, 62, 127],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    else:
        return None, None
    
    return model, param_grid

def auto_tune_model(model, param_grid, X_train, y_train):
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

def plot_feature_importance(self):
    try:
        st.write("### 모델별 특성 중요도 분석")
        for name, model in self.models.items():
            try:
                if name == 'Linear Regression':
                    # 특성 이름을 문자열로 적절히 변환
                    raw_feature_names = self.poly_features.get_feature_names_out(self.X_train.columns)
                    feature_names = []
                    for feat in raw_feature_names:
                        feat_str = str(feat)
                        if isinstance(feat, (tuple, list)):
                            feat_str = ' * '.join(map(str, feat))
                        feature_names.append(feat_str)
                    importance = np.abs(model.coef_)
                elif name == 'LSTM':
                    continue  # LSTM은 특성 중요도를 제공하지 않음
                elif name in ['Random Forest', 'XGBoost', 'LightGBM']:
                    if not hasattr(model, 'feature_importances_'):
                        continue
                    feature_names = [str(col) for col in self.X_train.columns]
                    importance = model.feature_importances_
                else:
                    continue
                
                # 데이터프레임 생성 및 정렬
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                })
                importance_df = importance_df.sort_values('Importance', ascending=False).head(15)
                
                # Plotly 그래프 생성
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=importance_df['Importance'],
                        y=importance_df['Feature'],
                        orientation='h',
                        marker=dict(
                            color=importance_df['Importance'],
                            colorscale='YlOrRd'
                        )
                    )
                )
                
                fig.update_layout(
                    title=f'{name} 모델의 특성 중요도',
                    xaxis_title='중요도',
                    yaxis_title='특성',
                    height=600,
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig)
                
                # 상세 데이터 표시
                st.write(f"#### {name} 모델의 특성 중요도 상세")
                st.dataframe(
                    importance_df.style
                    .background_gradient(cmap='YlOrRd', subset=['Importance'])
                )
                
            except Exception as model_error:
                st.warning(f"{name} 모델의 특성 중요도를 계산할 수 없습니다: {str(model_error)}")
                continue
            
    except Exception as e:
        st.error(f"특성 중요도 분석 중 오류 발생: {str(e)}")

def plot_learning_curves(history):
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
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2 Score': r2_score(y_test, y_pred),
        '추론 시간': f"{inference_time:.4f}초"
    }
    
    st.subheader("📊 모델 성능 평가")
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ['값']
    st.table(metrics_df)
    
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

def calculate_risk_metrics(data, signals):
    try:
        risk_metrics = {}
        
        # 변동성 계산
        returns = data['Close'].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        risk_metrics['Volatility'] = float(volatility.iloc[-1])
        
        # 샤프 비율 계산
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate/252
        if len(returns) > 0 and returns.std() > 0:
            risk_metrics['Sharpe_ratio'] = float((np.sqrt(252) * excess_returns.mean() / returns.std()))
        else:
            risk_metrics['Sharpe_ratio'] = 0.0
        
        # 최대 낙폭 계산
        rolling_max = data['Close'].rolling(252, min_periods=1).max()
        drawdown = (data['Close'] - rolling_max) / rolling_max
        risk_metrics['Max_drawdown'] = float(drawdown.min())
        
        return risk_metrics
        
    except Exception as e:
        st.error(f"리스크 메트릭스 계산 중 오류 발생: {str(e)}")
        return None

class TechnicalAnalyzer:
    def __init__(self, data):
        self.data = data
        self.features = []
    
    def calculate_indicators(self):
        try:
            df = self.data.copy()
            
            # 이동평균선
            for period in [5, 10, 20, 50, 200]:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            
            # 볼린저 밴드
            for period in [20]:
                df[f'BB_middle_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'BB_upper_{period}'] = df[f'BB_middle_{period}'] + 2 * df['Close'].rolling(window=period).std()
                df[f'BB_lower_{period}'] = df[f'BB_middle_{period}'] - 2 * df['Close'].rolling(window=period).std()
                df[f'BB_width_{period}'] = (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}']) / df[f'BB_middle_{period}']
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
            
            # 스토캐스틱
            low_14 = df['Low'].rolling(window=14).min()
            high_14 = df['High'].rolling(window=14).max()
            df['K_percent'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
            df['D_percent'] = df['K_percent'].rolling(window=3).mean()
            
            # ADX
            plus_dm = df['High'].diff()
            minus_dm = df['Low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift(1))
            tr3 = abs(df['Low'] - df['Close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['ADX'] = dx.rolling(window=14).mean()
            
            # OBV
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            
            # 모멘텀 지표
            df['ROC'] = df['Close'].pct_change(periods=12) * 100
            df['MOM'] = df['Close'].diff(periods=10)
            
            # 추가 변동성 지표
            df['ATR'] = self.calculate_atr(df)
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # 거래량 지표
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # 가격 모멘텀
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_Ratio'] = df['Close'] / df['Close'].rolling(window=20).mean()
            
            # 피보나치 레벨 계산
            high = df['High'].rolling(window=20).max()
            low = df['Low'].rolling(window=20).min()
            diff = high - low
            df['Fib_23.6'] = high - (diff * 0.236)
            df['Fib_38.2'] = high - (diff * 0.382)
            df['Fib_50.0'] = high - (diff * 0.500)
            df['Fib_61.8'] = high - (diff * 0.618)
            
            # 결측치 처리
            df.fillna(method='bfill', inplace=True)
            df.fillna(method='ffill', inplace=True)
            
            self.data = df
            self.features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            
            return df
            
        except Exception as e:
            st.error(f"기술적 지표 계산 중 오류 발생: {str(e)}")
            return None

    def calculate_atr(self, df, period=14):
        try:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(period).mean()
            return atr
        except Exception as e:
            st.error(f"ATR 계산 중 오류 발생: {str(e)}")
            return pd.Series(0, index=df.index)

    def get_trading_signals(self):
        """기술적 지표 기반 매매 신호 생성"""
        try:
            signals = pd.DataFrame(index=self.data.index)
            
            # RSI 기반 신호
            signals['RSI_Signal'] = 0
            signals.loc[self.data['RSI'] < 30, 'RSI_Signal'] = 1  # 매수
            signals.loc[self.data['RSI'] > 70, 'RSI_Signal'] = -1  # 매도
            
            # MACD 기반 신호
            signals['MACD_Signal'] = 0
            signals.loc[self.data['MACD'] > self.data['Signal_Line'], 'MACD_Signal'] = 1
            signals.loc[self.data['MACD'] < self.data['Signal_Line'], 'MACD_Signal'] = -1
            
            # 볼린저 밴드 기반 신호
            signals['BB_Signal'] = 0
            signals.loc[self.data['Close'] < self.data['BB_lower_20'], 'BB_Signal'] = 1
            signals.loc[self.data['Close'] > self.data['BB_upper_20'], 'BB_Signal'] = -1
            
            # 이동평균 크로스 신호
            signals['MA_Cross_Signal'] = 0
            signals.loc[self.data['SMA_5'] > self.data['SMA_20'], 'MA_Cross_Signal'] = 1
            signals.loc[self.data['SMA_5'] < self.data['SMA_20'], 'MA_Cross_Signal'] = -1
            
            # 종합 신호 계산
            signals['Total_Signal'] = (signals['RSI_Signal'] + 
                                     signals['MACD_Signal'] + 
                                     signals['BB_Signal'] + 
                                     signals['MA_Cross_Signal'])
            
            # 최종 매매 신호
            signals['Final_Signal'] = 'HOLD'
            signals.loc[signals['Total_Signal'] >= 2, 'Final_Signal'] = 'BUY'
            signals.loc[signals['Total_Signal'] <= -2, 'Final_Signal'] = 'SELL'
            
            return signals
            
        except Exception as e:
            st.error(f"매매 신호 생성 중 오류 발생: {str(e)}")
            return None

class ProbabilisticAnalyzer:
    def __init__(self, data, test_size=0.2, sequence_length=10):
        self.data = data
        self.test_size = test_size
        self.sequence_length = sequence_length
        self.models = {}
        self.predictions = {}
        self.signal_probabilities = {}
        
        # 데이터 준비
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # 스케일링된 데이터
        self.X_train_scaled = None
        self.X_test_scaled = None
        
        # 시퀀스 데이터
        self.X_train_seq = None
        self.X_test_seq = None
        self.y_train_seq = None
        self.y_test_seq = None
        
        # 회귀 분석용 데이터
        self.X_train_reg = None
        self.X_test_reg = None
        self.y_train_reg = None
        self.y_test_reg = None
        
        # 스케일러와 특성 변환기
        self.scaler = None
        self.poly_features = None
        
        # 초기 특성 준비
        self.prepare_features()

    def prepare_features(self):
        try:
            features = []
            df = self.data.copy()
            
            # Target 변수 생성 (다음 날의 종가 방향)
            df['Target'] = df['Close'].shift(-1) > df['Close']
            df['Target'] = df['Target'].astype(float)
            
            # 기술적 지표 계산
            df['Price_Change'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()
            df['ROC'] = df['Close'].pct_change(periods=12) * 100
            df['MOM'] = df['Close'].diff(periods=10)
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            
            # MA Cross 신호 생성
            df['MA_Cross'] = 0
            df.loc[df['SMA_5'] > df['SMA_20'], 'MA_Cross'] = 1
            df.loc[df['SMA_5'] < df['SMA_20'], 'MA_Cross'] = -1
            
            features = ['Price_Change', 'Volume_Change', 'ROC', 'MOM', 'Volatility', 'MA_Cross']
            
            # 결측치 제거 및 데이터 분할
            df = df.dropna()
            train_size = int(len(df) * (1 - self.test_size))
            train_data = df[:train_size]
            test_data = df[train_size:]
            
            # 학습/테스트 데이터 준비
            self.X_train = train_data[features]
            self.X_test = test_data[features]
            self.y_train = train_data['Target']
            self.y_test = test_data['Target']
            
            # 스케일링
            self.scaler = MinMaxScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            # 시퀀스 데이터 생성
            self.X_train_seq = self.create_sequences(self.X_train_scaled)
            self.X_test_seq = self.create_sequences(self.X_test_scaled)
            self.y_train_seq = self.y_train[self.sequence_length:].values
            self.y_test_seq = self.y_test[self.sequence_length:].values
            
            # 회귀 분석용 데이터 준비 (다음 날의 수익률)
            train_returns = train_data['Close'].pct_change().shift(-1).dropna()
            test_returns = test_data['Close'].pct_change().shift(-1).dropna()
            
            # 다항 특성 생성
            self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
            X_train_poly = self.poly_features.fit_transform(self.X_train_scaled)
            X_test_poly = self.poly_features.transform(self.X_test_scaled)
            
            # 회귀 분석용 데이터 최종 준비
            self.X_train_reg = X_train_poly[:-1]
            self.X_test_reg = X_test_poly[:-1]
            self.y_train_reg = train_returns
            self.y_test_reg = test_returns
            
            # 데이터 길이 맞추기
            min_train_len = min(len(self.X_train_reg), len(self.y_train_reg))
            min_test_len = min(len(self.X_test_reg), len(self.y_test_reg))
            
            self.X_train_reg = self.X_train_reg[:min_train_len]
            self.y_train_reg = self.y_train_reg[:min_train_len]
            self.X_test_reg = self.X_test_reg[:min_test_len]
            self.y_test_reg = self.y_test_reg[:min_test_len]
            
            # 데이터 유효성 검사
            if (self.y_train_reg is None or len(self.y_train_reg) == 0 or 
                self.y_test_reg is None or len(self.y_test_reg) == 0):
                raise ValueError("회귀 분석용 타겟 데이터가 유효하지 않습니다.")
            
            st.success("특성 준비가 완료되었습니다.")
            return features
            
        except Exception as e:
            st.error(f"특성 준비 중 오류 발생: {str(e)}")
            st.write("데이터 형태:", self.data.shape)
            return None
    
    def create_sequences(self, data):
        try:
            sequences = []
            for i in range(len(data) - self.sequence_length):
                sequence = data[i:(i + self.sequence_length)]
                sequences.append(sequence)
            return np.array(sequences)
        except Exception as e:
            st.error(f"시퀀스 생성 중 오류 발생: {str(e)}")
            return np.array([])
    
    def calculate_model_probabilities(self):
        try:
            for name, model in self.models.items():
                predictions_df = pd.DataFrame(index=self.X_test.index)
                
                if name == 'Linear Regression':
                    raw_predictions = model.predict(self.X_test_reg)
                    predictions_df['Value'] = raw_predictions
                    predictions_df['Signal'] = 'HOLD'
                    
                    for idx in predictions_df.index:
                        value = predictions_df.at[idx, 'Value']
                        if value > 0.01:
                            predictions_df.at[idx, 'Signal'] = 'BUY'
                        elif value < -0.01:
                            predictions_df.at[idx, 'Signal'] = 'SELL'
                
                elif name == 'LSTM':
                    raw_predictions = model.predict(self.X_test_seq)
                    predictions_df['Value'] = raw_predictions.flatten()
                    predictions_df['Signal'] = 'HOLD'
                    
                    for idx in predictions_df.index:
                        value = predictions_df.at[idx, 'Value']
                        if value > 0.6:
                            predictions_df.at[idx, 'Signal'] = 'BUY'
                        elif value < 0.4:
                            predictions_df.at[idx, 'Signal'] = 'SELL'
                
                else:
                    probabilities = model.predict_proba(self.X_test_scaled)
                    predictions_df['Value'] = probabilities[:, 1]
                    predictions_df['Signal'] = 'HOLD'
                    
                    for idx in predictions_df.index:
                        value = predictions_df.at[idx, 'Value']
                        if value > 0.6:
                            predictions_df.at[idx, 'Signal'] = 'BUY'
                        elif value < 0.4:
                            predictions_df.at[idx, 'Signal'] = 'SELL'
                
                self.predictions[name] = predictions_df['Signal']
                
                # 신호 확률 계산
                signal_counts = predictions_df['Signal'].value_counts()
                total_signals = len(predictions_df)
                
                self.signal_probabilities[name] = {
                    'BUY': float(signal_counts.get('BUY', 0)) / total_signals * 100,
                    'SELL': float(signal_counts.get('SELL', 0)) / total_signals * 100,
                    'HOLD': float(signal_counts.get('HOLD', 0)) / total_signals * 100
                }
            
            # 신호 확률 시각화
            self.plot_signal_probabilities()
            return True
        
        except Exception as e:
            st.error(f"확률 계산 중 오류 발생: {str(e)}")
            return False
    
    def plot_signal_probabilities(self):
        try:
            # 데이터 준비
            models = list(self.signal_probabilities.keys())
            signals = ['BUY', 'SELL', 'HOLD']
            
            fig = go.Figure()
            
            for signal in signals:
                values = [self.signal_probabilities[model][signal] for model in models]
                
                fig.add_trace(go.Bar(
                    name=signal,
                    x=models,
                    y=values,
                    text=[f"{v:.1f}%" for v in values],
                    textposition='auto',
                ))
            
            fig.update_layout(
                title='모델별 신호 확률 분포',
                xaxis_title='모델',
                yaxis_title='확률 (%)',
                barmode='group',
                height=500,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            st.plotly_chart(fig)
            
            # 상세 확률 표시
            st.write("### 모델별 신호 확률")
            prob_df = pd.DataFrame(self.signal_probabilities).round(2)
            st.dataframe(
                prob_df.style
                .format("{:.2f}%")
                .background_gradient(cmap='YlOrRd')
            )
            
        except Exception as e:
            st.error(f"신호 확률 시각화 중 오류 발생: {str(e)}")

    def train_models(self):
        try:
            self.models = {}
            
            # Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(self.X_train_reg, self.y_train_reg)
            self.models['Linear Regression'] = lr_model
            
            # Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            rf_model.fit(self.X_train_scaled, self.y_train)
            self.models['Random Forest'] = rf_model
            
            # XGBoost
            xgb_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            xgb_model.fit(self.X_train_scaled, self.y_train)
            self.models['XGBoost'] = xgb_model
            
            # LightGBM
            lgbm_model = LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            lgbm_model.fit(self.X_train_scaled, self.y_train)
            self.models['LightGBM'] = lgbm_model
            
            # LSTM
            lstm_model = self.build_lstm_model()
            lstm_model.fit(
                self.X_train_seq,
                self.y_train_seq,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            self.models['LSTM'] = lstm_model
            
            st.success("모든 모델 학습이 완료되었습니다.")
            return True
            
        except Exception as e:
            st.error(f"모델 학습 중 오류 발생: {str(e)}")
            return False

    def predict_future(self, days=30):
        try:
            future_predictions = {}
            last_data = self.data.iloc[-self.sequence_length:]
            
            for name, model in self.models.items():
                predictions = []
                current_data = last_data.copy()
                
                for _ in range(days):
                    # 특성 준비
                    features = self.prepare_prediction_features(current_data)
                    
                    if name == 'Linear Regression':
                        features_poly = self.poly_features.transform(features)
                        pred = model.predict(features_poly)[-1]
                        
                    elif name == 'LSTM':
                        seq = self.create_sequences(features)
                        if len(seq) > 0:
                            pred = model.predict(seq)[-1][0]
                        else:
                            continue
                            
                    else:
                        features_scaled = self.scaler.transform(features)
                        pred = model.predict_proba(features_scaled)[-1][1]
                    
                    predictions.append(pred)
                    
                    # 다음 예측을 위한 데이터 업데이트
                    new_row = current_data.iloc[-1].copy()
                    new_row['Close'] = current_data['Close'].iloc[-1] * (1 + pred)
                    current_data = pd.concat([current_data[1:], pd.DataFrame([new_row])])
                    current_data.index = pd.date_range(
                        start=current_data.index[0],
                        periods=len(current_data),
                        freq='D'
                    )
                
                future_predictions[name] = predictions
            
            # 예측 결과 시각화
            self.plot_future_predictions(future_predictions, days)
            
            return future_predictions
            
        except Exception as e:
            st.error(f"미래 예측 중 오류 발생: {str(e)}")
            return None

    def prepare_prediction_features(self, data):
        try:
            df = data.copy()
            
            # 기술적 지표 계산
            df['Price_Change'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()
            df['ROC'] = df['Close'].pct_change(periods=12) * 100
            df['MOM'] = df['Close'].diff(periods=10)
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            
            # MA Cross 신호
            df['MA_Cross'] = 0
            df.loc[df['SMA_5'] > df['SMA_20'], 'MA_Cross'] = 1
            df.loc[df['SMA_5'] < df['SMA_20'], 'MA_Cross'] = -1
            
            features = ['Price_Change', 'Volume_Change', 'ROC', 'MOM', 'Volatility', 'MA_Cross']
            
            return df[features].fillna(method='ffill')
            
        except Exception as e:
            st.error(f"예측 특성 준비 중 오류 발생: {str(e)}")
            return None

    def plot_future_predictions(self, future_predictions, days):
        try:
            # 과거 데이터 준비
            historical_dates = self.data.index[-30:]  # 최근 30일
            historical_prices = self.data['Close'].iloc[-30:]
            
            # 미래 날짜 생성
            last_date = self.data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
            
            # 그래프 생성
            fig = go.Figure()
            
            # 과거 데이터 플롯
            fig.add_trace(go.Scatter(
                x=historical_dates,
                y=historical_prices,
                name='과거 데이터',
                line=dict(color='gray')
            ))
            
            # 각 모델의 예측값 플롯
            for name, predictions in future_predictions.items():
                # 예측값을 실제 가격으로 변환
                last_price = historical_prices.iloc[-1]
                predicted_prices = []
                current_price = last_price
                
                for pred in predictions:
                    if name == 'Linear Regression':
                        current_price = current_price * (1 + pred)
                    else:
                        # 분류 모델의 경우 확률을 가격 변화로 변환
                        price_change = (pred - 0.5) * 0.02  # 2% 최대 변화
                        current_price = current_price * (1 + price_change)
                    predicted_prices.append(current_price)
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=predicted_prices,
                    name=f'{name} 예측',
                    line=dict(dash='dash')
                ))
            
            # 레이아웃 설정
            fig.update_layout(
                title='주가 예측 결과',
                xaxis_title='날짜',
                yaxis_title='가격',
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # 그리드 추가
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            st.plotly_chart(fig)
            
            # 예측 신뢰도 표시
            confidence_df = pd.DataFrame(columns=['모델', '예측 방향', '신뢰도'])
            for name, predictions in future_predictions.items():
                if len(predictions) > 0:
                    last_pred = predictions[-1]
                    if name == 'Linear Regression':
                        direction = '상승' if last_pred > 0 else '하락'
                        confidence = abs(last_pred) * 100
                    else:
                        direction = '상승' if last_pred > 0.5 else '하락'
                        confidence = abs(last_pred - 0.5) * 200
                    
                    confidence_df = pd.concat([confidence_df, pd.DataFrame({
                        '모델': [name],
                        '예측 방향': [direction],
                        '신뢰도': [f"{min(confidence, 100):.1f}%"]
                    })])
            
            st.write("### 예측 신뢰도")
            st.dataframe(confidence_df.set_index('모델'))
            
        except Exception as e:
            st.error(f"미래 예측 시각화 중 오류 발생: {str(e)}")
            return None

    def build_lstm_model(self):
        try:
            # 입력 형태 검증
            if not hasattr(self, 'sequence_length') or not hasattr(self, 'X_train'):
                raise ValueError("시퀀스 길이 또는 학습 데이터가 설정되지 않았습니다.")
            
            # 입력 차원 계산
            input_dim = self.X_train.shape[1] if hasattr(self, 'X_train') else 1
            
            # LSTM 모델 구성 - 단순화된 버전
            model = Sequential([
                LSTM(
                    units=64,
                    activation='tanh',  # ReLU 대신 tanh 사용
                    input_shape=(self.sequence_length, input_dim),
                    return_sequences=False,
                    kernel_initializer='glorot_uniform'
                ),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            # 모델 컴파일 - 학습률 조정
            optimizer = Adam(learning_rate=0.0005)
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Early Stopping 콜백 설정
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,  # 인내심 감소
                restore_best_weights=True,
                mode='min'
            )
            
            # 모델 속성에 콜백 저장
            self.callbacks = [early_stopping]
            
            return model
            
        except Exception as e:
            st.error(f"LSTM 모델 생성 중 오류 발생: {str(e)}")
            st.info("기본 LSTM 모델을 생성합니다.")
            
            # 더 단순한 기본 LSTM 모델 생성
            default_model = Sequential([
                LSTM(32, input_shape=(self.sequence_length, input_dim), activation='tanh'),
                Dense(1, activation='sigmoid')
            ])
            default_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return default_model

    def compare_models(self):
        """모델별 성능 비교 시각화"""
        try:
            if not hasattr(self, 'models'):
                st.warning("학습된 모델이 없습니다.")
                return None
            
            # 각 모델별 성능 분석
            for name, model in self.models.items():
                with st.expander(f"📊 {name} 모델 분석 결과"):
                    try:
                        # 기본 성능 지표 계산
                        metrics = {}
                        if name == 'Linear Regression':
                            y_pred = model.predict(self.X_test_reg)
                            metrics['R2'] = r2_score(self.y_test_reg, y_pred)
                            metrics['MAE'] = mean_absolute_error(self.y_test_reg, y_pred)
                            metrics['RMSE'] = np.sqrt(mean_squared_error(self.y_test_reg, y_pred))
                            
                            # 방향성 정확도 계산
                            correct_direction = np.sum(
                                (y_pred > 0) == (self.y_test_reg.values > 0)
                            )
                            metrics['방향성 정확도'] = (correct_direction / len(y_pred)) * 100
                            
                        elif name == 'LSTM':
                            y_pred = model.predict(self.X_test_seq)
                            metrics['Accuracy'] = accuracy_score(self.y_test_seq, (y_pred > 0.5).astype(int))
                            metrics['Precision'] = precision_score(self.y_test_seq, (y_pred > 0.5).astype(int))
                            metrics['Recall'] = recall_score(self.y_test_seq, (y_pred > 0.5).astype(int))
                            metrics['F1'] = f1_score(self.y_test_seq, (y_pred > 0.5).astype(int))
                            
                        elif name in ['Random Forest', 'XGBoost', 'LightGBM']:
                            y_pred = model.predict(self.X_test)
                            metrics['Accuracy'] = accuracy_score(self.y_test, y_pred)
                            metrics['Precision'] = precision_score(self.y_test, y_pred)
                            metrics['Recall'] = recall_score(self.y_test, y_pred)
                            metrics['F1'] = f1_score(self.y_test, y_pred)
                            
                            if hasattr(model, 'feature_importances_'):
                                feature_importance = pd.DataFrame({
                                    'Feature': self.X_train.columns,
                                    'Importance': model.feature_importances_
                                })
                                feature_importance = feature_importance.sort_values('Importance', ascending=False)
                                
                                st.write("#### 주요 특성 중요도")
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=feature_importance['Feature'][:10],
                                    y=feature_importance['Importance'][:10],
                                    name='특성 중요도'
                                ))
                                fig.update_layout(
                                    title='상위 10개 특성 중요도',
                                    xaxis_title='특성',
                                    yaxis_title='중요도',
                                    xaxis=dict(tickangle=45)
                                )
                                st.plotly_chart(fig)
                        
                        # 성능 지표 표시
                        st.write("#### 모델 성능 지표")
                        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['값'])
                        st.dataframe(
                            metrics_df.style
                            .format({'값': '{:.4f}'})
                            .background_gradient(cmap='YlOrRd')
                        )
                        
                        # AI 전문가 분석
                        st.write("#### AI 전문가 분석")
                        analysis = analyze_model_performance(metrics)
                        st.markdown(analysis)
                        
                    except Exception as model_error:
                        st.error(f"{name} 모델 분석 중 오류 발생: {str(model_error)}")
                        continue
            
            # 모델 간 성능 비교 시각화
            st.write("### 모델 간 성능 비교")
            comparison_metrics = {}
            for name, model in self.models.items():
                if name == 'Linear Regression':
                    y_pred = model.predict(self.X_test_reg)
                    comparison_metrics[name] = {
                        '방향성 정확도': (np.sum((y_pred > 0) == (self.y_test_reg.values > 0)) / len(y_pred)) * 100
                    }
                else:
                    y_pred = model.predict(self.X_test if name not in ['LSTM'] else self.X_test_seq)
                    y_true = self.y_test if name not in ['LSTM'] else self.y_test_seq
                    comparison_metrics[name] = {
                        'Accuracy': accuracy_score(y_true, (y_pred > 0.5).astype(int) if name == 'LSTM' else y_pred) * 100
                    }
            
            # 비교 차트 생성
            comparison_df = pd.DataFrame(comparison_metrics).T
            fig = go.Figure()
            for col in comparison_df.columns:
                fig.add_trace(go.Bar(
                    name=col,
                    x=comparison_df.index,
                    y=comparison_df[col],
                    text=comparison_df[col].round(2).astype(str) + '%',
                    textposition='auto',
                ))
            
            fig.update_layout(
                title='모델별 성능 비교',
                yaxis_title='정확도 (%)',
                barmode='group',
                showlegend=True
            )
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"모델 비교 분석 중 오류 발생: {str(e)}")
            return None

    def plot_feature_importance(self):
        try:
            st.write("### 모델별 특성 중요도 분석")
            for name, model in self.models.items():
                try:
                    if name == 'Linear Regression':
                        # 특성 이름을 문자열로 적절히 변환
                        raw_feature_names = self.poly_features.get_feature_names_out(self.X_train.columns)
                        feature_names = []
                        for feat in raw_feature_names:
                            feat_str = str(feat)
                            if isinstance(feat, (tuple, list)):
                                feat_str = ' * '.join(map(str, feat))
                            feature_names.append(feat_str)
                        importance = np.abs(model.coef_)
                    elif name == 'LSTM':
                        continue  # LSTM은 특성 중요도를 제공하지 않음
                    elif name in ['Random Forest', 'XGBoost', 'LightGBM']:
                        if not hasattr(model, 'feature_importances_'):
                            continue
                        feature_names = [str(col) for col in self.X_train.columns]
                        importance = model.feature_importances_
                    else:
                        continue
                    
                    # 데이터프레임 생성 및 정렬
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importance
                    })
                    importance_df = importance_df.sort_values('Importance', ascending=False).head(15)
                    
                    # Plotly 그래프 생성
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=importance_df['Importance'],
                            y=importance_df['Feature'],
                            orientation='h',
                            marker=dict(
                                color=importance_df['Importance'],
                                colorscale='YlOrRd'
                            )
                        )
                    )
                    
                    fig.update_layout(
                        title=f'{name} 모델의 특성 중요도',
                        xaxis_title='중요도',
                        yaxis_title='특성',
                        height=600,
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig)
                    
                    # 상세 데이터 표시
                    st.write(f"#### {name} 모델의 특성 중요도 상세")
                    st.dataframe(
                        importance_df.style
                        .background_gradient(cmap='YlOrRd', subset=['Importance'])
                    )
                    
                except Exception as model_error:
                    st.warning(f"{name} 모델의 특성 중요도를 계산할 수 없습니다: {str(model_error)}")
                    continue
                
        except Exception as e:
            st.error(f"특성 중요도 분석 중 오류 발생: {str(e)}")

    def plot_roc_curves(self):
        try:
            st.write("### 모델별 ROC 곡선 비교")
            fig = go.Figure()
            auc_scores = {}
            
            # 기준선 추가
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    line=dict(dash='dash', color='gray'),
                    name='Random Classifier'
                )
            )
            
            for name, model in self.models.items():
                try:
                    if name == 'Linear Regression':
                        y_pred = model.predict(self.X_test_reg)
                        y_true = (self.y_test_reg.values > 0).astype(int)  # 수익률의 방향을 기준으로 이진 분류
                        y_score = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))
                    
                    elif name == 'LSTM':
                        y_true = self.y_test_seq
                        y_score = model.predict(self.X_test_seq).ravel()
                    
                    else:
                        y_true = self.y_test.values  # numpy array로 변환
                        y_score = model.predict_proba(self.X_test_scaled)[:, 1]
                    
                    # ROC 곡선 계산
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    auc_score = auc(fpr, tpr)
                    auc_scores[name] = auc_score
                    
                    # ROC 곡선 추가
                    fig.add_trace(
                        go.Scatter(
                            x=fpr,
                            y=tpr,
                            name=f'{name} (AUC = {auc_score:.3f})',
                            mode='lines'
                        )
                    )
                    
                except Exception as model_error:
                    st.warning(f"{name} 모델의 ROC 곡선을 생성할 수 없습니다: {str(model_error)}")
                    continue
                
            # 그래프 레이아웃 설정
            fig.update_layout(
                title='ROC Curves Comparison',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=600,
                width=800,
                showlegend=True,
                legend=dict(
                    yanchor="bottom",
                    y=0.01,
                    xanchor="right",
                    x=0.99
                )
            )
            
            # 그리드 추가
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            # 그래프 표시
            st.plotly_chart(fig)
            
            # AUC 점수 표시
            if auc_scores:
                st.write("### AUC 점수 비교")
                auc_df = pd.DataFrame(
                    auc_scores.items(),
                    columns=['모델', 'AUC 점수']
                ).sort_values('AUC 점수', ascending=False)
                
                st.dataframe(
                    auc_df.style
                    .background_gradient(cmap='YlOrRd', subset=['AUC 점수'])
                    .format({'AUC 점수': '{:.4f}'})
                )
        except Exception as e:
            st.error(f"ROC 곡선 생성 중 오류 발생: {str(e)}")
            st.info("일부 모델에서 ROC 곡선을 생성할 수 없습니다.")

    def plot_regression_analysis(self):
        """회귀 분석 결과를 시각화합니다."""
        try:
            if not hasattr(self, 'models') or 'Linear Regression' not in self.models:
                st.warning("선형 회귀 모델이 학습되지 않았습니다.")
                return None
            
            # 기존 회귀 분석 시각화 코드
            X_test_reg = self.X_test_reg
            y_test_reg = self.y_test_reg
            
            if X_test_reg is None or y_test_reg is None:
                st.warning("테스트 데이터가 준비되지 않았습니다.")
                return None
            
            # 예측값 계산
            try:
                y_pred = self.models['Linear Regression'].predict(X_test_reg)
                
                # 차원 일치 확인 및 조정
                if len(y_pred.shape) > 1:
                    y_pred = y_pred.flatten()
                
                # 산점도 및 회귀선 그래프
                fig = go.Figure()
                
                # 실제 값과 예측값 산점도
                fig.add_trace(go.Scatter(
                    x=y_test_reg,
                    y=y_pred,
                    mode='markers',
                    name='실제 vs 예측',
                    marker=dict(
                        size=8,
                        color='blue',
                        opacity=0.6
                    )
                ))
                
                # 이상적인 예측선 (y=x)
                min_val = min(min(y_test_reg), min(y_pred))
                max_val = max(max(y_test_reg), max(y_pred))
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='이상적인 예측',
                    line=dict(color='red', dash='dash')
                ))
                
                # 그래프 레이아웃 설정
                fig.update_layout(
                    title='실제 값 vs 예측 값 비교',
                    xaxis_title='실제 값',
                    yaxis_title='예측 값',
                    showlegend=True
                )
                
                # 그리드 추가
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                # 그래프 표시
                st.plotly_chart(fig)
                
                # 회귀 분석 성능 지표 계산
                metrics = {
                    'R2': r2_score(y_test_reg, y_pred),
                    'MAE': mean_absolute_error(y_test_reg, y_pred),
                    'MSE': mean_squared_error(y_test_reg, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred))
                }
                
                # 회귀 분석 결과 해석 expander 추가
                with st.expander("📊 선형 회귀 분석 결과 해석"):
                    st.write("#### 모델 성능 지표")
                    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['값'])
                    st.dataframe(
                        metrics_df.style
                        .format({'값': '{:.4f}'})
                        .background_gradient(cmap='YlOrRd')
                    )
                    
                    # GPT-4를 사용한 상세 분석
                    st.write("#### 전문가 분석")
                    analysis = analyze_model_performance(metrics)
                    st.markdown(analysis)
                    
                    st.write("#### 주요 특성 중요도")
                    if hasattr(self.models['Linear Regression'], 'coef_'):
                        coef = self.models['Linear Regression'].coef_
                        # 다항 특성의 이름을 가져옵니다
                        feature_names = self.poly_features.get_feature_names_out(self.X_train.columns)
                        
                        # 특성 중요도 데이터프레임 생성
                        feature_importance = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': np.abs(coef)
                        })
                        feature_importance = feature_importance.sort_values('Importance', ascending=False)
                        
                        # 상위 10개 특성만 선택
                        top_10_features = feature_importance.head(10)
                        
                        fig_importance = go.Figure()
                        fig_importance.add_trace(go.Bar(
                            x=top_10_features['Feature'],
                            y=top_10_features['Importance'],
                            name='특성 중요도'
                        ))
                        fig_importance.update_layout(
                            title='상위 10개 특성 중요도',
                            xaxis_title='특성',
                            yaxis_title='중요도',
                            xaxis=dict(tickangle=45)
                        )
                        st.plotly_chart(fig_importance)

            except Exception as e:
                st.error(f"회귀 분석 시각화 중 오류 발생: {str(e)}")
                return None
                
        except Exception as e:
            st.error(f"회귀 분석 처리 중 오류 발생: {str(e)}")
            return None

    def plot_model_comparison(self):
        """모델별 성능 비교 시각화"""
        try:
            if not self.performance_metrics:
                return None, pd.DataFrame()

            # 성능 지표 데이터프레임 생성
            comparison_data = []
            for name, metrics in self.performance_metrics.items():
                row_data = {
                    'Model': name,
                    'Current Signal': self.predictions[name].iloc[-1] if name in self.predictions else 'HOLD',
                    'Accuracy': f"{metrics.get('Win_Rate', 0):.2f}%",
                    'Cumulative Return': f"{metrics.get('Cumulative_Return', 0)*100:.2f}%",
                    'Sharpe Ratio': f"{metrics.get('Sharpe_Ratio', 0):.2f}",
                    'Risk Level': metrics.get('Risk_Level', 'MEDIUM')
                }
                comparison_data.append(row_data)

            comparison_df = pd.DataFrame(comparison_data)

            # 시각화
            fig = go.Figure()
            colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}

            for signal in ['BUY', 'SELL', 'HOLD']:
                mask = comparison_df['Current Signal'] == signal
                if not any(mask):
                    continue

                fig.add_trace(go.Bar(
                    name=signal,
                    x=comparison_df[mask]['Model'],
                    y=[float(acc.strip('%')) for acc in comparison_df[mask]['Accuracy']],
                    marker_color=colors[signal]
                ))

            fig.update_layout(
                title='모델별 성능 비교',
                xaxis_title='모델',
                yaxis_title='정확도 (%)',
                barmode='group',
                showlegend=True,
                height=500
            )

            return fig, comparison_df

        except Exception as e:
            st.error(f"모델 비교 시각화 중 오류 발생: {str(e)}")
            return None, pd.DataFrame()

class ModelSignalAnalyzer:
    def __init__(self, models, data, predictions):
        self.models = models
        self.data = data
        self.predictions = predictions
        self.performance_metrics = {}
        self.risk_free_rate = 0.02  # 연간 무위험 수익률 (예: 2%)
        self.trading_days = 252  # 연간 거래일수
        self.returns = self.data['Close'].pct_change().fillna(0)

    def analyze_signals(self):
        """모델별 매매 신호 분석"""
        try:
            metrics = {}
            
            for name, signals in self.predictions.items():
                if signals is None or len(signals) == 0:
                    st.warning(f"{name} 모델의 예측 신호가 없습니다.")
                    continue
                
                # 포지션 설정 (1: 매수, -1: 매도, 0: 관망)
                positions = pd.Series(0, index=signals.index)
                positions[signals == 'BUY'] = 1
                positions[signals == 'SELL'] = -1
                
                # 수익률 계산
                strategy_returns = positions.shift(1) * self.returns  # 다음날의 수익률에 적용
                strategy_returns = strategy_returns.fillna(0)
                
                # 고급 지표 계산
                try:
                    advanced_metrics = self.calculate_advanced_metrics(strategy_returns)
                    if advanced_metrics:
                        metrics[name] = advanced_metrics
                except Exception as e:
                    st.warning(f"{name} 모델의 성능 지표 계산 중 오류 발생: {str(e)}")
                    continue
            
            if not metrics:
                st.warning("모든 모델의 성능 지표 계산에 실패했습니다.")
                return None
            
            self.performance_metrics = metrics
            return metrics
            
        except Exception as e:
            st.error(f"매매 신호 분석 중 오류 발생: {str(e)}")
            return None

    def calculate_advanced_metrics(self, strategy_returns):
        """고급 성능 지표 계산"""
        try:
            metrics = {}
            
            # 누적 수익률
            cumulative_return = (1 + strategy_returns).cumprod().iloc[-1] - 1
            metrics['Cumulative_Return'] = cumulative_return
            
            # 변동성 (연율화)
            volatility = strategy_returns.std() * np.sqrt(self.trading_days)
            metrics['Volatility'] = volatility
            
            # 샤프 비율
            excess_returns = strategy_returns - (self.risk_free_rate / self.trading_days)
            if len(strategy_returns) > 0 and strategy_returns.std() > 0:
                sharpe_ratio = np.sqrt(self.trading_days) * excess_returns.mean() / strategy_returns.std()
                metrics['Sharpe_Ratio'] = sharpe_ratio
            else:
                metrics['Sharpe_Ratio'] = 0
            
            # 최대 낙폭
            cumulative = (1 + strategy_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            metrics['Max_Drawdown'] = float(drawdown.min())
            
            # 승률
            total_trades = len(strategy_returns[strategy_returns != 0])
            if total_trades > 0:
                winning_trades = len(strategy_returns[strategy_returns > 0])
                metrics['Win_Rate'] = winning_trades / total_trades
            else:
                metrics['Win_Rate'] = 0
            
            # 손익비
            gains = strategy_returns[strategy_returns > 0]
            losses = strategy_returns[strategy_returns < 0]
            
            avg_gain = gains.mean() if len(gains) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
            
            metrics['Profit_Loss_Ratio'] = avg_gain / avg_loss if avg_loss != 0 else 0
            
            # 위험 수준 평가
            risk_score = (abs(volatility) * 0.5 + abs(metrics['Max_Drawdown']) * 0.5)
            if risk_score < 0.15:
                metrics['Risk_Level'] = 'LOW'
            elif risk_score < 0.25:
                metrics['Risk_Level'] = 'MEDIUM'
            else:
                metrics['Risk_Level'] = 'HIGH'
            
            return metrics
            
        except Exception as e:
            st.error(f"고급 지표 계산 중 오류 발생: {str(e)}")
            return None

    def plot_model_comparison(self):
        """모델별 성능 비교 시각화"""
        try:
            if not self.performance_metrics:
                st.warning("성능 지표가 계산되지 않았습니다. 먼저 analyze_signals를 실행하세요.")
                return None, pd.DataFrame()

            # 성능 지표 데이터프레임 생성
            comparison_data = []
            for name, metrics in self.performance_metrics.items():
                if name not in self.predictions or self.predictions[name] is None:
                    continue
                    
                current_signal = self.predictions[name].iloc[-1] if len(self.predictions[name]) > 0 else 'HOLD'
                
                row_data = {
                    'Model': name,
                    'Current Signal': current_signal,
                    'Accuracy': f"{metrics.get('Win_Rate', 0)*100:.2f}%",
                    'Cumulative Return': f"{metrics.get('Cumulative_Return', 0)*100:.2f}%",
                    'Sharpe Ratio': f"{metrics.get('Sharpe_Ratio', 0):.2f}",
                    'Risk Level': metrics.get('Risk_Level', 'MEDIUM')
                }
                comparison_data.append(row_data)

            if not comparison_data:
                st.warning("표시할 모델 비교 데이터가 없습니다.")
                return None, pd.DataFrame()

            comparison_df = pd.DataFrame(comparison_data)

            # 시각화
            fig = go.Figure()
            colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}

            for signal in ['BUY', 'SELL', 'HOLD']:
                mask = comparison_df['Current Signal'] == signal
                if not any(mask):
                    continue

                fig.add_trace(go.Bar(
                    name=signal,
                    x=comparison_df[mask]['Model'],
                    y=[float(acc.strip('%')) for acc in comparison_df[mask]['Accuracy']],
                    marker_color=colors[signal]
                ))

            fig.update_layout(
                title='모델별 성능 비교',
                xaxis_title='모델',
                yaxis_title='정확도 (%)',
                barmode='group',
                showlegend=True,
                height=500
            )

            return fig, comparison_df

        except Exception as e:
            st.error(f"모델 비교 시각화 중 오류 발생: {str(e)}")
            return None, pd.DataFrame()

class MarketAnalyzer:
    def __init__(self, data, benchmark_ticker='^GSPC'):  # S&P 500을 기본 벤치마크로 사용
        self.data = data
        self.benchmark_ticker = benchmark_ticker
        self.benchmark_data = None
        self.market_regime = None
        self.correlation_matrix = None
        self.sector_performance = None
        
    def analyze_market_regime(self):
        """시장 국면 분석"""
        try:
            df = self.data.copy()
            
            # 변동성 계산
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # 추세 강도 계산
            df['Trend_Strength'] = abs(df['Close'].pct_change(20))
            
            # 시장 국면 분류
            df['Market_Regime'] = 'Neutral'
            
            # 고변동성 & 강한 상승추세 = 과열
            mask_overheated = (df['Volatility'] > df['Volatility'].quantile(0.8)) & \
                            (df['Trend_Strength'] > df['Trend_Strength'].quantile(0.8))
            df.loc[mask_overheated, 'Market_Regime'] = 'Overheated'
            
            # 고변동성 & 강한 하락추세 = 공포
            mask_fear = (df['Volatility'] > df['Volatility'].quantile(0.8)) & \
                       (df['Trend_Strength'] < df['Trend_Strength'].quantile(0.2))
            df.loc[mask_fear, 'Market_Regime'] = 'Fear'
            
            # 저변동성 & 완만한 상승추세 = 안정
            mask_stable = (df['Volatility'] < df['Volatility'].quantile(0.2)) & \
                         (df['Trend_Strength'] > df['Trend_Strength'].median())
            df.loc[mask_stable, 'Market_Regime'] = 'Stable'
            
            self.market_regime = df['Market_Regime']
            return df['Market_Regime']
            
        except Exception as e:
            st.error(f"시장 국면 분석 중 오류 발생: {str(e)}")
            return None
    
    def calculate_market_correlation(self):
        """시장과의 상관관계 분석"""
        try:
            if self.benchmark_data is None:
                self.benchmark_data = fdr.DataReader(self.benchmark_ticker, 
                                                   self.data.index[0], 
                                                   self.data.index[-1])
            
            # 수익률 계산
            stock_returns = self.data['Close'].pct_change()
            market_returns = self.benchmark_data['Close'].pct_change()
            
            # 상관관계 계산
            correlation = stock_returns.corr(market_returns)
            
            # 베타 계산
            covariance = stock_returns.cov(market_returns)
            market_variance = market_returns.var()
            beta = covariance / market_variance
            
            return {
                'Correlation': correlation,
                'Beta': beta
            }
            
        except Exception as e:
            st.error(f"시장 상관관계 분석 중 오류 발생: {str(e)}")
            return None
    
    def analyze_sector_performance(self, sector_tickers):
        """섹터 성과 분석"""
        try:
            sector_data = {}
            for ticker in sector_tickers:
                sector_data[ticker] = fdr.DataReader(ticker, 
                                                   self.data.index[0], 
                                                   self.data.index[-1])
            
            # 섹터별 수익률 계산
            returns = {}
            for ticker, data in sector_data.items():
                returns[ticker] = (data['Close'][-1] / data['Close'][0] - 1) * 100
            
            self.sector_performance = returns
            return returns
            
        except Exception as e:
            st.error(f"섹터 성과 분석 중 오류 발생: {str(e)}")
            return None
    
    def calculate_risk_metrics(self):
        """리스크 메트릭스 계산"""
        try:
            metrics = {}
            returns = self.data['Close'].pct_change().dropna()
            
            # 변동성
            metrics['Volatility'] = returns.std() * np.sqrt(252)
            
            # 샤프 비율
            risk_free_rate = 0.02  # 연간 2% 가정
            excess_returns = returns - risk_free_rate/252
            metrics['Sharpe_Ratio'] = np.sqrt(252) * returns.mean() / returns.std()
            
            # 최대 낙폭
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            metrics['Max_Drawdown'] = drawdowns.min()
            
            # Value at Risk (95% 신뢰수준)
            metrics['VaR_95'] = np.percentile(returns, 5)
            
            # Conditional VaR (Expected Shortfall)
            metrics['CVaR_95'] = returns[returns <= metrics['VaR_95']].mean()
            
            return metrics
            
        except Exception as e:
            st.error(f"리스크 메트릭스 계산 중 오류 발생: {str(e)}")
            return None
    
    def plot_market_analysis(self):
        """시장 분석 결과 시각화"""
        try:
            # 시장 국면 분포 파이 차트
            regime_counts = self.market_regime.value_counts()
            fig1 = go.Figure(data=[go.Pie(labels=regime_counts.index, 
                                        values=regime_counts.values)])
            fig1.update_layout(title='시장 국면 분포')
            st.plotly_chart(fig1)
            
            # 리스크 메트릭스 레이더 차트
            risk_metrics = self.calculate_risk_metrics()
            if risk_metrics:
                fig2 = go.Figure(data=go.Scatterpolar(
                    r=[risk_metrics['Volatility'],
                       risk_metrics['Sharpe_Ratio'],
                       abs(risk_metrics['Max_Drawdown']),
                       abs(risk_metrics['VaR_95']),
                       abs(risk_metrics['CVaR_95'])],
                    theta=['변동성', '샤프비율', '최대낙폭', 'VaR', 'CVaR'],
                    fill='toself'
                ))
                fig2.update_layout(title='리스크 프로파일')
                st.plotly_chart(fig2)
            
            # 섹터 성과 비교 바 차트
            if self.sector_performance:
                fig3 = go.Figure(data=[go.Bar(
                    x=list(self.sector_performance.keys()),
                    y=list(self.sector_performance.values())
                )])
                fig3.update_layout(title='섹터별 성과 비교')
                st.plotly_chart(fig3)
            
        except Exception as e:
            st.error(f"시장 분석 시각화 중 오류 발생: {str(e)}")

class BacktestSystem:
    def __init__(self, data, initial_capital=10000000):
        self.data = data
        self.initial_capital = initial_capital
        self.positions = pd.DataFrame(index=data.index).fillna(0)
        self.portfolio_value = pd.Series(index=data.index).fillna(0)
        self.trades = []
        
    def run_backtest(self, signals, commission=0.0015):
        """백테스트 실행"""
        try:
            # 포트폴리오 초기화
            capital = self.initial_capital
            position = 0
            self.trades = []
            
            for i in range(len(self.data)):
                date = self.data.index[i]
                close_price = self.data['Close'].iloc[i]
                
                # 매매 신호 확인
                if i > 0:  # 첫날은 제외
                    signal = signals['Final_Signal'].iloc[i-1]  # 전날 신호로 오늘 매매
                    
                    # 매수 신호
                    if signal == 'BUY' and position == 0:
                        shares = int(capital * (1 - commission) / close_price)
                        cost = shares * close_price * (1 + commission)
                        if cost <= capital:
                            position = shares
                            capital -= cost
                            self.trades.append({
                                'Date': date,
                                'Type': 'BUY',
                                'Shares': shares,
                                'Price': close_price,
                                'Cost': cost
                            })
                    
                    # 매도 신호
                    elif signal == 'SELL' and position > 0:
                        revenue = position * close_price * (1 - commission)
                        capital += revenue
                        self.trades.append({
                            'Date': date,
                            'Type': 'SELL',
                            'Shares': position,
                            'Price': close_price,
                            'Revenue': revenue
                        })
                        position = 0
                
                # 포지션 및 포트폴리오 가치 기록
                self.positions.loc[date] = position
                self.portfolio_value.loc[date] = capital + (position * close_price)
            
            return self.calculate_backtest_metrics()
            
        except Exception as e:
            st.error(f"백테스트 실행 중 오류 발생: {str(e)}")
            return None
    
    def calculate_backtest_metrics(self):
        """백테스트 성과 지표 계산"""
        try:
            metrics = {}
            
            # 일간 수익률
            daily_returns = self.portfolio_value.pct_change().dropna()
            
            # 총 수익률
            metrics['Total_Return'] = (self.portfolio_value.iloc[-1] / self.initial_capital - 1) * 100
            
            # 연간 수익률
            years = (self.data.index[-1] - self.data.index[0]).days / 365
            metrics['Annual_Return'] = ((1 + metrics['Total_Return']/100) ** (1/years) - 1) * 100
            
            # 변동성
            metrics['Volatility'] = daily_returns.std() * np.sqrt(252) * 100
            
            # 샤프 비율
            risk_free_rate = 0.02  # 연간 2% 가정
            excess_returns = daily_returns - risk_free_rate/252
            metrics['Sharpe_Ratio'] = np.sqrt(252) * excess_returns.mean() / daily_returns.std()
            
            # 최대 낙폭
            cum_returns = (1 + daily_returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            metrics['Max_Drawdown'] = drawdowns.min() * 100
            
            # 승률
            winning_trades = len([t for t in self.trades if t['Type'] == 'SELL' and 
                                t['Revenue'] > t['Shares'] * self.trades[self.trades.index(t)-1]['Price']])
            total_trades = len([t for t in self.trades if t['Type'] == 'SELL'])
            metrics['Win_Rate'] = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # 손익비
            gains = [t['Revenue'] - t['Shares'] * self.trades[self.trades.index(t)-1]['Price'] 
                    for t in self.trades if t['Type'] == 'SELL']
            if gains:
                avg_gain = sum([g for g in gains if g > 0]) / len([g for g in gains if g > 0]) \
                    if len([g for g in gains if g > 0]) > 0 else 0
                avg_loss = abs(sum([g for g in gains if g < 0]) / len([g for g in gains if g < 0])) \
                    if len([g for g in gains if g < 0]) > 0 else 1
                metrics['Profit_Loss_Ratio'] = avg_gain / avg_loss if avg_loss != 0 else 0
            
            return metrics
            
        except Exception as e:
            st.error(f"백테스트 지표 계산 중 오류 발생: {str(e)}")
            return None
    
    def plot_backtest_results(self):
        """백테스트 결과 시각화"""
        try:
            # 포트폴리오 가치 변화
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=self.portfolio_value.index,
                y=self.portfolio_value.values,
                name='포트폴리오 가치'
            ))
            fig1.update_layout(title='포트폴리오 가치 변화',
                             xaxis_title='날짜',
                             yaxis_title='포트폴리오 가치')
            st.plotly_chart(fig1)
            
            # 수익률 분포
            daily_returns = self.portfolio_value.pct_change().dropna()
            fig2 = go.Figure(data=[go.Histogram(x=daily_returns, nbinsx=50)])
            fig2.update_layout(title='일간 수익률 분포',
                             xaxis_title='수익률',
                             yaxis_title='빈도')
            st.plotly_chart(fig2)
            
            # 드로다운 차트
            cum_returns = (1 + daily_returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=drawdowns.index,
                y=drawdowns.values * 100,
                fill='tozeroy',
                name='드로다운'
            ))
            fig3.update_layout(title='드로다운 차트',
                             xaxis_title='날짜',
                             yaxis_title='드로다운 (%)')
            st.plotly_chart(fig3)
            
            # 성과 지표 표시
            metrics = self.calculate_backtest_metrics()
            if metrics:
                st.write("### 백테스트 성과 지표")
                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['값'])
                st.dataframe(
                    metrics_df.style
                    .format({
                        '값': '{:.2f}' if metrics_df.index != 'Profit_Loss_Ratio' else '{:.2f}:1'
                    })
                    .background_gradient(cmap='YlOrRd')
                )
            
        except Exception as e:
            st.error(f"백테스트 결과 시각화 중 오류 발생: {str(e)}")
            return None

# OpenAI API 설정
from openai import OpenAI

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    st.warning("OpenAI API 키가 설정되지 않았습니다. 모델 성능 상세 분석이 제한됩니다.")

def format_metrics(metrics: Dict[str, Any]) -> Dict[str, str]:
    """성능 지표를 보기 좋게 포맷팅합니다."""
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ['Win_Rate', 'Total_Return', 'Annual_Return']:
                formatted[key] = f"{value:.2f}%"
            elif key in ['Sharpe_Ratio', 'Profit_Loss_Ratio']:
                formatted[key] = f"{value:.2f}"
            else:
                formatted[key] = f"{value:.4f}"
        else:
            formatted[key] = str(value)
    return formatted

def analyze_model_performance(metrics: Dict[str, Any]) -> str:
    """GPT-4를 사용하여 모델 성능을 분석하고 설명을 생성합니다."""
    if not OPENAI_API_KEY:
        return "API 키가 설정되지 않아 상세 분석을 수행할 수 없습니다."
    
    try:
        # 메트릭스를 포맷팅
        formatted_metrics = format_metrics(metrics)
        metrics_str = "\n".join([f"{k}: {v}" for k, v in formatted_metrics.items()])
        
        # GPT-4에 전송할 프롬프트 작성
        prompt = f"""
        다음은 금융 머신러닝 모델의 성능 지표입니다:
        {metrics_str}
        
        이 성능 지표들을 바탕으로 다음 사항들을 포함하여 전문가적인 분석을 제공해주세요:
        1. 모델의 전반적인 성능 평가
        2. 강점과 약점
        3. 실전 트레이딩에서의 활용 가능성
        4. 개선이 필요한 부분
        5. 투자자들이 주의해야 할 점
        
        분석은 전문적이면서도 이해하기 쉽게 작성해주세요.
        """
        
        # GPT-4 API 호출 (새로운 방식)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 금융 머신러닝 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"모델 성능 분석 중 오류가 발생했습니다: {str(e)}"

if st.sidebar.button("분석 시작"):
    stock_data = get_stock_data(ticker, start_date, end_date)
    if stock_data is not None:
        with st.spinner("데이터 분석 중..."):
            try:
                tech_analyzer = TechnicalAnalyzer(stock_data)
                prob_analyzer = ProbabilisticAnalyzer(tech_analyzer.data)
                if prob_analyzer.train_models():
                    st.subheader("📊 모델 성능 비교 분석")
                    prob_analyzer.compare_models()
                    st.subheader("📈 특성 중요도 분석")
                    prob_analyzer.plot_feature_importance()
                    st.subheader("📉 ROC 곡선 분석")
                    prob_analyzer.plot_roc_curves()
                    st.subheader("🤖 모델별 매매 신호 분석")
                    
                    try:
                        model_analyzer = ModelSignalAnalyzer(prob_analyzer.models, tech_analyzer.data, prob_analyzer.predictions)
                        signals_result = model_analyzer.analyze_signals()
                        
                        if signals_result is not None:
                            try:
                                fig, matrix_df = model_analyzer.plot_model_comparison()
                                if fig is not None and not matrix_df.empty:
                                    st.write("### 모델별 매매 신호 확률")
                                    try:
                                        # 기존 데이터프레임 표시
                                        st.dataframe(
                                            matrix_df.style.apply(
                                                lambda x: [
                                                    'background-color: #e6ffe6' if v == 'BUY'
                                                    else 'background-color: #ffe6e6' if v == 'SELL'
                                                    else 'background-color: #f2f2f2'
                                                    for v in x
                                                ],
                                                subset=['Current Signal']
                                            )
                                        )
                                        st.plotly_chart(fig)
                                        
                                        # 각 모델별 상세 분석
                                        st.write("### 모델별 상세 분석")
                                        for name, metrics in self.performance_metrics.items():
                                            with st.expander(f"{name} 모델 상세 분석"):
                                                # 기본 메트릭스 표시
                                                st.write("#### 기본 성능 지표")
                                                metrics_df = pd.DataFrame(
                                                    metrics.items(),
                                                    columns=['지표', '값']
                                                ).set_index('지표')
                                                st.dataframe(metrics_df)
                                                
                                                # GPT-4를 사용한 상세 분석
                                                st.write("#### AI 전문가 분석")
                                                analysis = analyze_model_performance(metrics)
                                                st.markdown(analysis)
                                    except Exception as e:
                                        st.error(f"매매 신호 시각화 중 오류 발생: {str(e)}")
                                else:
                                    st.warning("모델 비교 결과가 없습니다.")
                            except Exception as e:
                                st.error(f"모델 비교 시각화 중 오류 발생: {str(e)}")
                        else:
                            st.warning("매매 신호 분석 결과가 없습니다.")
                    except Exception as e:
                        st.error(f"매매 신호 분석 중 오류 발생: {str(e)}")
                    
                    # 회귀 분석 시각화
                    try:
                        prob_analyzer.plot_regression_analysis()
                    except Exception as e:
                        st.error(f"회귀 분석 시각화 중 오류 발생: {str(e)}")
            except Exception as e:
                st.error(f"분석 중 오류 발생: {str(e)}")
    else:
        st.error("데이터를 불러오는데 실패했습니다. 티커 심볼을 확인해주세요.")
