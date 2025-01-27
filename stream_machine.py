import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    mean_absolute_error,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor, XGBClassifier
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    st.warning("LightGBM을 사용할 수 없습니다. 설치하려면: pip install lightgbm")
import plotly.express as px
import time
import sys
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb

# 전역 변수 설정
XGBOOST_AVAILABLE = False

# XGBoost 가용성 확인
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    st.warning("XGBoost를 사용할 수 없습니다. 설치하려면: pip install xgboost")

# LightGBM 가용성 확인
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    st.warning("LightGBM을 사용할 수 없습니다. 설치하려면: pip install lightgbm")

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
@st.cache_data(ttl=3600)  # 1시간 캐시
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
        ))
    fig.add_trace(
        go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='이상적인 예측',
            line=dict(color='red', dash='dash')
        ))
    fig.update_layout(
        title='예측 vs 실제 값 비교',
        xaxis_title='실제 값',
        yaxis_title='예측 값'
    )
    st.plotly_chart(fig)
    
    return metrics

# 기술적 지표 계산 함수 추가
def calculate_technical_indicators(data):
    """기술적 지표 계산"""
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['BB_upper'] = data['MA20'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_lower'] = data['MA20'] - 2 * data['Close'].rolling(window=20).std()
    
    return data

# 확률적 예측 모델 클래스 추가
class ProbabilisticPredictor:
    def __init__(self, data, sequence_length=30):
        self.data = data
        self.sequence_length = sequence_length
        self.models = []
        self.prepare_data()
        
    def prepare_data(self):
        """데이터 전처리"""
        # 특성 생성
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()
        self.data['MA20'] = self.data['Close'].rolling(window=20).mean()
        
        # NaN 제거
        self.data.dropna(inplace=True)
        
        # 입력 특성 선택
        self.features = ['Open', 'High', 'Low', 'Volume', 'Returns', 'Volatility', 'MA20']
        self.X = self.data[self.features]
        self.y = self.data['Close']
        
        # 데이터 스케일링
        self.scaler = MinMaxScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # 학습/테스트 분할
        train_size = int(len(self.X_scaled) * 0.8)
        self.X_train = self.X_scaled[:train_size]
        self.X_test = self.X_scaled[train_size:]
        self.y_train = self.y[:train_size]
        self.y_test = self.y[train_size:]
    
    def train_ensemble(self):
        """앙상블 모델 학습"""
        self.models = []
        base_models = [
            RandomForestRegressor(n_estimators=100, random_state=42),
            XGBRegressor(random_state=42)
        ]
        
        if LIGHTGBM_AVAILABLE:
            base_models.append(LGBMRegressor(random_state=42))
        
        for model in base_models:
            model.fit(self.X_train, self.y_train)
            self.models.append(model)
        
        return self.models
    
    def predict_probability(self, X=None):
        """확률적 예측 수행"""
        if X is None:
            X = self.X_test
            
        if len(self.models) == 0:
            self.train_ensemble()
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred

# 투자 신호 생성 함수 추가
def generate_trading_signals(data, pred_mean, pred_std):
    """매수/매도/관망 신호 생성"""
    try:
        signals = pd.DataFrame(index=data.index[-len(pred_mean):])
        
        # 기술적 지표 기반 신호
        signals['RSI_signal'] = np.where(data['RSI'].iloc[-len(pred_mean):] < 30, 'BUY',
                                       np.where(data['RSI'].iloc[-len(pred_mean):] > 70, 'SELL', 'HOLD'))
        
        signals['MACD_signal'] = np.where(data['MACD'].iloc[-len(pred_mean):] > data['Signal'].iloc[-len(pred_mean):], 'BUY',
                                        np.where(data['MACD'].iloc[-len(pred_mean):] < data['Signal'].iloc[-len(pred_mean):], 'SELL', 'HOLD'))
        
        # 볼린저 밴드 기반 신호
        signals['BB_signal'] = np.where(data['Close'].iloc[-len(pred_mean):] < data['BB_lower'].iloc[-len(pred_mean):], 'BUY',
                                      np.where(data['Close'].iloc[-len(pred_mean):] > data['BB_upper'].iloc[-len(pred_mean):], 'SELL', 'HOLD'))
        
        # 확률적 예측 기반 신호
        current_price = data['Close'].iloc[-len(pred_mean):]
        confidence_interval = 1.96 * pred_std
        
        signals['Pred_signal'] = np.where(pred_mean - confidence_interval > current_price, 'BUY',
                                        np.where(pred_mean + confidence_interval < current_price, 'SELL', 'HOLD'))
        
        # 종합 신호 (최빈값 기준)
        signals['Final_signal'] = signals.mode(axis=1)[0]
        
        return signals['Final_signal']
        
    except Exception as e:
        st.error(f"매매 신호 생성 중 오류 발생: {str(e)}")
        return None

# 리스크 관리 지표 계산 함수 추가
def calculate_risk_metrics(data, signals):
    """리스크 관리 지표 계산"""
    risk_metrics = {}
    
    # 변동성 (20일)
    volatility = data['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    risk_metrics['Volatility'] = volatility.iloc[-1]  # 수정된 부분
    
    # 샤프 비율
    returns = data['Close'].pct_change()
    risk_free_rate = 0.02  # 연간 기준
    excess_returns = returns - risk_free_rate/252
    risk_metrics['Sharpe_ratio'] = (np.sqrt(252) * excess_returns.mean() / returns.std())
    
    # 최대 낙폭
    rolling_max = data['Close'].rolling(252, min_periods=1).max()
    drawdown = (data['Close'] - rolling_max) / rolling_max
    risk_metrics['Max_drawdown'] = drawdown.min()
    
    return risk_metrics

class TechnicalAnalyzer:
    def __init__(self, data):
        self.data = data
        self.features = []
        self.calculate_indicators()
        
    def calculate_indicators(self):
        """기술적 지표 계산"""
        df = self.data.copy()
        
        # 기본 이동평균
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_60'] = df['Close'].rolling(window=60).mean()
        
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
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 볼린저 밴드
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # 모멘텀 지표
        df['ROC'] = df['Close'].pct_change(periods=12) * 100
        df['MOM'] = df['Close'].diff(periods=10)
        
        # 변동성 지표
        df['ATR'] = self.calculate_atr(df)
        
        # 거래량 지표
        df['OBV'] = self.calculate_obv(df)
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # 추가 파생 지표
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        # NaN 값 처리
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        
        self.data = df
        self.features = [
            'SMA_5', 'SMA_20', 'SMA_60', 'RSI', 'MACD', 'Signal',
            'BB_Width', 'ROC', 'MOM', 'ATR', 'OBV', 'Volume_MA',
            'Price_Change', 'Volatility', 'BB_upper', 'BB_lower'
        ]
        
    def calculate_atr(self, df):
        """ATR(Average True Range) 계산"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(14).mean()
    
    def calculate_obv(self, df):
        """OBV(On Balance Volume) 계산"""
        obv = np.zeros(len(df))
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv[i] = obv[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv[i] = obv[i-1] - df['Volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        return obv
    
    def get_features(self):
        """분석에 사용할 특성 반환"""
        return self.data[self.features]

class ProbabilisticAnalyzer:
    def __init__(self, data, test_size=0.2, sequence_length=10):
        self.data = data
        self.test_size = test_size
        self.sequence_length = sequence_length
        self.models = {}
        self.predictions = {}
        self.signal_probabilities = {}
        self.linear_model = None
        self.poly_features = None
        
    def prepare_features(self):
        """예측을 위한 특성 준비"""
        try:
            # 기본 특성 선택
            features = [
                'RSI', 'MACD', 'Signal', 'BB_upper', 'BB_lower', 'MA20',
                'Volume', 'Close', 'High', 'Low', 'Open'
            ]
            
            # 데이터 복사본 생성
            df = self.data.copy()
            
            # 기술적 지표 계산
            df['Target'] = df['Close'].shift(-1) > df['Close']
            df['BB_Position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
            df['Price_Change'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()
            df['ROC'] = df['Close'].pct_change(periods=12) * 100
            df['MOM'] = df['Close'].diff(periods=10)
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_Cross'] = np.where(df['SMA_5'] > df['SMA_20'], 1, -1)
            
            # 추가 특성 목록에 추가
            features.extend(['BB_Position', 'Price_Change', 'Volume_Change', 
                           'ROC', 'MOM', 'Volatility', 'MA_Cross'])
            
            # NaN 제거
            df = df.dropna()
            
            # 마지막 행 제거 (다음 날 종가를 알 수 없음)
            df = df.iloc[:-1]
            
            # 데이터 분할
            train_size = int(len(df) * (1 - self.test_size))
            
            # 학습/테스트 데이터 분할
            train_data = df[:train_size]
            test_data = df[train_size:]
            
            # X, y 데이터 준비
            self.X_train = train_data[features]
            self.X_test = test_data[features]
            self.y_train = train_data['Target']
            self.y_test = test_data['Target']
            
            # 스케일링
            self.scaler = MinMaxScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            # LSTM용 시퀀스 데이터
            self.X_train_seq = self.create_sequences(self.X_train_scaled)
            self.X_test_seq = self.create_sequences(self.X_test_scaled)
            self.y_train_seq = self.y_train[self.sequence_length:].values
            self.y_test_seq = self.y_test[self.sequence_length:].values
            
            # 선형 회귀를 위한 데이터 준비
            self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
            X_train_poly = self.poly_features.fit_transform(self.X_train_scaled)
            X_test_poly = self.poly_features.transform(self.X_test_scaled)
            
            # 선형 회귀용 타겟 (다음 날의 종가 변화율)
            train_returns = train_data['Close'].pct_change().shift(-1).iloc[:-1]
            test_returns = test_data['Close'].pct_change().shift(-1).iloc[:-1]
            
            # 회귀 데이터 준비
            self.X_train_reg = X_train_poly[:-1]  # 마지막 행 제외
            self.X_test_reg = X_test_poly[:-1]    # 마지막 행 제외
            self.y_train_reg = train_returns.dropna()
            self.y_test_reg = test_returns.dropna()
            
            # 데이터 길이 확인 및 조정
            min_train_len = min(len(self.X_train_reg), len(self.y_train_reg))
            min_test_len = min(len(self.X_test_reg), len(self.y_test_reg))
            
            self.X_train_reg = self.X_train_reg[:min_train_len]
            self.y_train_reg = self.y_train_reg[:min_train_len]
            self.X_test_reg = self.X_test_reg[:min_test_len]
            self.y_test_reg = self.y_test_reg[:min_test_len]
            
            # 데이터 준비 상태 확인
            assert len(self.X_train_reg) == len(self.y_train_reg), "학습 데이터 길이 불일치"
            assert len(self.X_test_reg) == len(self.y_test_reg), "테스트 데이터 길이 불일치"
            
            st.success("특성 준비가 완료되었습니다.")
            return features
            
        except Exception as e:
            st.error(f"특성 준비 중 오류 발생: {str(e)}")
            st.write("데이터 형태:", self.data.shape)
            return None
    
    def create_sequences(self, data):
        """LSTM을 위한 시퀀스 데이터 생성"""
        sequences = []
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:(i + self.sequence_length)])
        return np.array(sequences)
    
    def calculate_model_probabilities(self):
        """각 모델별 매수/매도/관망 확률 계산"""
        try:
            for name, model in self.models.items():
                if name == 'LSTM':
                    pred_proba = model.predict(self.X_test_seq, verbose=0)
                elif name == 'Linear Regression':
                    # 선형 회귀의 경우 예측값을 확률로 변환
                    pred = model.predict(self.X_test_reg)
                    
                    # Min-Max 스케일링으로 0~1 범위로 변환
                    pred_scaled = (pred - pred.min()) / (pred.max() - pred.min())
                    pred_proba = pred_scaled
                else:
                    pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
                # dtype을 명시적으로 지정하여 경고 메시지 제거
                signals = pd.Series(
                    index=self.X_test.index[-len(pred_proba):],
                    dtype='object'
                )
                
                if name == 'Linear Regression':
                    # 선형 회귀의 경우 변화율 기반으로 신호 생성
                    signals.loc[pred > 0.01] = 'BUY'     # 1% 이상 상승 예측
                    signals.loc[pred < -0.01] = 'SELL'   # 1% 이상 하락 예측
                    signals.loc[(pred >= -0.01) & (pred <= 0.01)] = 'HOLD'
                else:
                    # 다른 모델들의 경우 확률 기반으로 신호 생성
                    signals.loc[pred_proba > 0.66] = 'BUY'
                    signals.loc[pred_proba < 0.33] = 'SELL'
                    signals.loc[(pred_proba >= 0.33) & (pred_proba <= 0.66)] = 'HOLD'
                
                self.predictions[name] = signals
                
                # 신호별 확률 계산
                total_signals = len(signals)
                signal_counts = signals.value_counts()
                
                self.signal_probabilities[name] = {
                    'BUY': signal_counts.get('BUY', 0) / total_signals * 100,
                    'SELL': signal_counts.get('SELL', 0) / total_signals * 100,
                    'HOLD': signal_counts.get('HOLD', 0) / total_signals * 100
                }
                
        except Exception as e:
            st.error(f"확률 계산 중 오류 발생: {str(e)}")
            return False
        
        return True

    def train_all_models(self):
        """모든 모델 학습"""
        try:
            features = self.prepare_features()
            if features is None:
                return False
            
            # 선형 회귀 모델 학습
            try:
                linear_model = LinearRegression()
                linear_model.fit(self.X_train_reg, self.y_train_reg)
                self.models['Linear Regression'] = linear_model
                
                # 선형 회귀 예측을 신호로 변환
                y_pred_reg = linear_model.predict(self.X_test_reg)
                signals = pd.Series(index=self.y_test_reg.index, dtype='object')
                
                # 변화율에 따른 신호 생성
                signals.loc[y_pred_reg > 0.01] = 'BUY'    # 1% 이상 상승 예측
                signals.loc[y_pred_reg < -0.01] = 'SELL'  # 1% 이상 하락 예측
                signals.loc[(y_pred_reg >= -0.01) & (y_pred_reg <= 0.01)] = 'HOLD'
                
                self.predictions['Linear Regression'] = signals
                
            except Exception as e:
                st.warning(f"선형 회귀 모델 학습 중 오류 발생: {str(e)}")
                
            # Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced'
            )
            rf_model.fit(self.X_train_scaled, self.y_train)
            self.models['Random Forest'] = rf_model
            
            # XGBoost
            if XGBOOST_AVAILABLE:
                xgb_model = XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                    scale_pos_weight=1
                )
                xgb_model.fit(self.X_train_scaled, self.y_train)
                self.models['XGBoost'] = xgb_model
            
            # LightGBM - 파라미터 수정
            if LIGHTGBM_AVAILABLE:
                # 경고 메시지 임시 비활성화
                import warnings
                warnings.filterwarnings('ignore', category=UserWarning)
                warnings.filterwarnings('ignore', category=FutureWarning)
                
                lgb_model = LGBMClassifier(
                    n_estimators=1000,  # 증가된 트리 수
                    num_leaves=15,      # 감소된 잎 노드 수
                    max_depth=4,        # 감소된 트리 깊이
                    learning_rate=0.05, # 감소된 학습률
                    min_child_samples=5,  # 감소된 최소 샘플 수
                    min_child_weight=1,  # 증가된 최소 가중치
                    min_split_gain=0.1,  # 증가된 분할 임계값
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    class_weight='balanced',
                    force_col_wise=True  # 컬럼 방식 멀티스레딩 강제
                )
                
                # 검증 세트 분리
                X_train_lgb, X_val_lgb, y_train_lgb, y_val_lgb = train_test_split(
                    self.X_train_scaled,
                    self.y_train,
                    test_size=0.2,
                    random_state=42,
                    stratify=self.y_train  # 클래스 비율 유지
                )
                
                # 모델 학습
                try:
                    lgb_model.fit(
                        X_train_lgb,
                        y_train_lgb,
                        eval_set=[(X_val_lgb, y_val_lgb)],
                        eval_metric='auc',  # 변경된 평가 메트릭
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=50),  # 증가된 조기 종료 라운드
                            lgb.log_evaluation(period=0)
                        ]
                    )
                    
                    self.models['LightGBM'] = lgb_model
                    
                except Exception as e:
                    st.warning(f"LightGBM 학습 중 오류 발생: {str(e)}")
                    st.info("다른 모델들로 계속 진행합니다.")
                
                # 경고 메시지 다시 활성화
                warnings.resetwarnings()
            
            # LSTM 모델 수정
            try:
                input_shape = (self.sequence_length, len(features))
                inputs = Input(shape=input_shape)
                
                # LSTM 레이어 구성 수정
                x = LSTM(64, return_sequences=True, 
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal')(inputs)
                x = BatchNormalization()(x)
                x = Dropout(0.3)(x)
                
                x = LSTM(32, 
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal')(x)
                x = BatchNormalization()(x)
                x = Dropout(0.3)(x)
                
                x = Dense(16, activation='relu')(x)
                x = BatchNormalization()(x)
                outputs = Dense(1, activation='sigmoid')(x)
                
                lstm_model = Model(inputs=inputs, outputs=outputs)
                
                # 컴파일 설정 수정
                optimizer = Adam(
                    learning_rate=0.001,
                    clipnorm=1.0  # 그래디언트 클리핑 추가
                )
                
                lstm_model.compile(
                    optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # 클래스 가중치 계산
                class_weights = compute_class_weight(
                    'balanced',
                    classes=np.unique(self.y_train_seq),
                    y=self.y_train_seq
                )
                class_weight_dict = dict(enumerate(class_weights))
                
                # Early Stopping 콜백 추가
                early_stopping = EarlyStopping(
                    monitor='loss',
                    patience=5,
                    restore_best_weights=True
                )
                
                # 모델 학습
                lstm_model.fit(
                    self.X_train_seq, 
                    self.y_train_seq,
                    epochs=50, 
                    batch_size=32,
                    class_weight=class_weight_dict,
                    callbacks=[early_stopping],
                    validation_split=0.2,
                    verbose=0
                )
                
                self.models['LSTM'] = lstm_model
                
            except Exception as e:
                st.warning(f"LSTM 학습 중 오류 발생: {str(e)}")
                st.info("다른 모델들로 계속 진행합니다.")
            
            return self.calculate_model_probabilities()
            
        except Exception as e:
            st.error(f"모델 학습 중 오류 발생: {str(e)}")
            return False
    
    def compare_models(self):
        """모델 성능 비교 분석"""
        try:
            # 모델별 성능 지표 저장
            performance_metrics = {}
            
            for name, model in self.models.items():
                metrics = {}
                
                if name == 'Linear Regression':
                    # 선형 회귀 모델의 성능 지표
                    y_pred = model.predict(self.X_test_reg)
                    metrics['R2'] = r2_score(self.y_test_reg, y_pred)
                    metrics['MAE'] = mean_absolute_error(self.y_test_reg, y_pred)
                    metrics['RMSE'] = np.sqrt(mean_squared_error(self.y_test_reg, y_pred))
                    
                    # 방향성 예측 정확도 계산
                    direction_accuracy = np.mean(
                        (y_pred > 0) == (self.y_test_reg > 0)
                    ) * 100
                    metrics['방향성 정확도'] = direction_accuracy
                    
                elif name == 'LSTM':
                    # LSTM 모델의 성능 지표
                    y_pred = (model.predict(self.X_test_seq) > 0.5).astype(int)
                    metrics['정확도'] = accuracy_score(self.y_test_seq, y_pred)
                    metrics['정밀도'] = precision_score(self.y_test_seq, y_pred)
                    metrics['재현율'] = recall_score(self.y_test_seq, y_pred)
                    metrics['F1 점수'] = f1_score(self.y_test_seq, y_pred)
                    
                else:
                    # 분류 모델들의 성능 지표
                    y_pred = model.predict(self.X_test_scaled)
                    metrics['정확도'] = accuracy_score(self.y_test, y_pred)
                    metrics['정밀도'] = precision_score(self.y_test, y_pred)
                    metrics['재현율'] = recall_score(self.y_test, y_pred)
                    metrics['F1 점수'] = f1_score(self.y_test, y_pred)
                
                performance_metrics[name] = metrics
            
            # 성능 지표 시각화
            st.write("### 모델별 성능 비교")
            
            # 1. 성능 지표 테이블
            metrics_df = pd.DataFrame(performance_metrics).round(4)
            st.dataframe(
                metrics_df.style.background_gradient(cmap='YlOrRd')
            )
            
            # 2. 모델별 주요 지표 시각화
            fig = go.Figure()
            
            for name, metrics in performance_metrics.items():
                if name == 'Linear Regression':
                    # 선형 회귀 모델은 방향성 정확도만 표시
                    fig.add_trace(
                        go.Bar(
                            name=name,
                            x=['방향성 정확도'],
                            y=[metrics['방향성 정확도']],
                            text=[f"{metrics['방향성 정확도']:.2f}%"],
                            textposition='auto'
                        )
                    )
                else:
                    # 다른 모델들은 정확도와 F1 점수 표시
                    fig.add_trace(
                        go.Bar(
                            name=name,
                            x=['정확도', 'F1 점수'],
                            y=[metrics['정확도'], metrics['F1 점수']],
                            text=[f"{metrics['정확도']:.2f}", f"{metrics['F1 점수']:.2f}"],
                            textposition='auto'
                        )
                    )
            
            fig.update_layout(
                title='모델별 주요 성능 지표',
                xaxis_title='지표',
                yaxis_title='점수',
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig)
            
            # 3. 모델 순위 계산
            model_ranks = {}
            for name, metrics in performance_metrics.items():
                if name == 'Linear Regression':
                    model_ranks[name] = metrics['방향성 정확도']
                else:
                    # 정확도와 F1 점수의 평균으로 순위 계산
                    model_ranks[name] = (metrics['정확도'] + metrics['F1 점수']) / 2
            
            ranks_df = pd.DataFrame(
                model_ranks.items(), 
                columns=['모델', '종합 점수']
            ).sort_values('종합 점수', ascending=False)
            
            st.write("### 모델 순위")
            st.dataframe(
                ranks_df.style.background_gradient(
                    cmap='YlOrRd',
                    subset=['종합 점수']
                )
            )
            
            return performance_metrics
            
        except Exception as e:
            st.error(f"모델 비교 분석 중 오류 발생: {str(e)}")
            return None

    def plot_feature_importance(self):
        """특성 중요도 시각화"""
        try:
            st.write("### 모델별 특성 중요도 분석")
            
            for name, model in self.models.items():
                if name == 'Linear Regression':
                    # 선형 회귀 계수의 절대값을 중요도로 사용
                    feature_names = self.poly_features.get_feature_names_out(self.X_train.columns)
                    importance = np.abs(model.coef_)
                    
                elif name == 'LSTM':
                    # LSTM은 특성 중요도 계산 생략
                    continue
                    
                elif name in ['Random Forest', 'XGBoost', 'LightGBM']:
                    # 트리 기반 모델의 특성 중요도
                    feature_names = self.X_train.columns
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                    else:
                        continue
                else:
                    continue
                
                # 특성 중요도 데이터프레임 생성
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                })
                
                # 중요도 기준 정렬
                importance_df = importance_df.sort_values(
                    'Importance', 
                    ascending=False
                ).reset_index(drop=True)
                
                # 상위 15개 특성만 선택
                importance_df = importance_df.head(15)
                
                # 특성 중요도 시각화
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
                
                # 특성 중요도 테이블
                st.write(f"#### {name} 모델의 특성 중요도 상세")
                st.dataframe(
                    importance_df.style.background_gradient(
                        cmap='YlOrRd',
                        subset=['Importance']
                    )
                )
                
                # 특성 간 상관관계 분석
                if name == 'Linear Regression':
                    st.write("#### 주요 특성 간 상관관계")
                    top_features = importance_df['Feature'].head(10).tolist()
                    
                    # 다항식 특성의 경우 원본 특성만 선택
                    original_features = [f for f in self.X_train.columns if f in top_features]
                    
                    if original_features:
                        corr_matrix = self.X_train[original_features].corr()
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            colorscale='RdBu',
                            zmin=-1,
                            zmax=1
                        ))
                        
                        fig.update_layout(
                            title='주요 특성 간 상관관계',
                            height=600,
                            width=800
                        )
                        
                        st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"특성 중요도 분석 중 오류 발생: {str(e)}")

    def plot_roc_curves(self):
        """모델별 ROC 곡선 시각화"""
        try:
            st.write("### 모델별 ROC 곡선 비교")
            
            fig = go.Figure()
            auc_scores = {}
            
            # Random classifier 기준선
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
                        # 선형 회귀의 경우 예측값을 이진 분류로 변환
                        y_pred = model.predict(self.X_test_reg)
                        y_true = (self.y_test_reg > 0).astype(int)
                        y_score = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
                        
                    elif name == 'LSTM':
                        # LSTM의 경우 예측 확률 직접 사용
                        y_true = self.y_test_seq
                        y_score = model.predict(self.X_test_seq).ravel()
                        
                    else:
                        # 다른 분류 모델들
                        y_true = self.y_test
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
            
            st.plotly_chart(fig)
            
            # AUC 점수 테이블
            if auc_scores:
                st.write("### AUC 점수 비교")
                auc_df = pd.DataFrame(
                    auc_scores.items(),
                    columns=['모델', 'AUC 점수']
                ).sort_values('AUC 점수', ascending=False)
                
                st.dataframe(
                    auc_df.style.background_gradient(
                        cmap='YlOrRd',
                        subset=['AUC 점수']
                    ).format({'AUC 점수': '{:.4f}'})
                )
            
        except Exception as e:
            st.error(f"ROC 곡선 생성 중 오류 발생: {str(e)}")
            st.info("일부 모델에서 ROC 곡선을 생성할 수 없습니다.")

class ModelSignalAnalyzer:
    def __init__(self, models, data, predictions):
        self.models = models
        self.data = data
        self.predictions = predictions
        self.performance_metrics = {}
        self.returns = self.data['Close'].pct_change().fillna(0)  # NaN 값을 0으로 처리
        
    def analyze_signals(self):
        """모델별 매매 신호 분석"""
        try:
            signal_metrics = {}
            
            for name, signals in self.predictions.items():
                # 매매 포지션 초기화
                positions = pd.Series(0, index=signals.index)
                positions[signals == 'BUY'] = 1
                positions[signals == 'SELL'] = -1
                
                # 해당 기간의 수익률 계산
                period_returns = self.returns[signals.index]
                strategy_returns = positions * period_returns
                
                # 누적 수익률 계산
                cumulative_returns = (1 + strategy_returns).cumprod()
                
                # 성과 지표 계산
                total_return = cumulative_returns.iloc[-1] - 1
                sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
                max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
                win_rate = len(strategy_returns[strategy_returns > 0]) / len(strategy_returns)
                
                # 매매 신호 통계
                signal_counts = signals.value_counts()
                total_signals = len(signals)
                
                metrics = {
                    '누적 수익률': total_return,
                    '샤프 비율': sharpe_ratio,
                    '최대 낙폭': max_drawdown,
                    '승률': win_rate,
                    '매수 신호 비율': signal_counts.get('BUY', 0) / total_signals,
                    '매도 신호 비율': signal_counts.get('SELL', 0) / total_signals,
                    '관망 신호 비율': signal_counts.get('HOLD', 0) / total_signals,
                    '총 거래 횟수': len(positions[positions != 0])
                }
                
                signal_metrics[name] = metrics
                
                # 수익률 곡선 시각화
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns,
                    mode='lines',
                    name=f'{name} 전략 수익률',
                    line=dict(width=2)
                ))
                
                fig.update_layout(
                    title=f'{name} 모델의 누적 수익률',
                    xaxis_title='날짜',
                    yaxis_title='누적 수익률',
                    height=500
                )
                
                st.plotly_chart(fig)
                
                # 성과 지표 표시
                metrics_df = pd.DataFrame([metrics]).T
                metrics_df.columns = ['값']
                
                # 성과 지표 포맷팅
                formatted_metrics = metrics_df.copy()
                formatted_metrics.loc['누적 수익률', '값'] = f"{metrics['누적 수익률']*100:.2f}%"
                formatted_metrics.loc['샤프 비율', '값'] = f"{metrics['샤프 비율']:.2f}"
                formatted_metrics.loc['최대 낙폭', '값'] = f"{metrics['최대 낙폭']*100:.2f}%"
                formatted_metrics.loc['승률', '값'] = f"{metrics['승률']*100:.2f}%"
                formatted_metrics.loc['매수 신호 비율', '값'] = f"{metrics['매수 신호 비율']*100:.2f}%"
                formatted_metrics.loc['매도 신호 비율', '값'] = f"{metrics['매도 신호 비율']*100:.2f}%"
                formatted_metrics.loc['관망 신호 비율', '값'] = f"{metrics['관망 신호 비율']*100:.2f}%"
                formatted_metrics.loc['총 거래 횟수', '값'] = f"{metrics['총 거래 횟수']}"
                
                st.write(f"### {name} 모델 성과 지표")
                st.dataframe(formatted_metrics)
                
                # 매매 신호 분포 시각화
                fig_signals = go.Figure(data=[
                    go.Pie(
                        labels=['매수', '매도', '관망'],
                        values=[
                            metrics['매수 신호 비율'],
                            metrics['매도 신호 비율'],
                            metrics['관망 신호 비율']
                        ],
                        hole=.3
                    )
                ])
                
                fig_signals.update_layout(
                    title=f'{name} 모델의 매매 신호 분포',
                    height=400
                )
                
                st.plotly_chart(fig_signals)
            
            return signal_metrics
            
        except Exception as e:
            st.error(f"신호 분석 중 오류 발생: {str(e)}")
            return None

    def plot_model_comparison(self):
        """모델 성능 비교 시각화"""
        try:
            # 모델별 성능 지표 계산
            for name in self.predictions.keys():
                # predictions가 Series인 경우 처리
                if isinstance(self.predictions[name], pd.Series):
                    signals = self.predictions[name]
                else:
                    # predictions가 다른 형태인 경우 Series로 변환
                    signals = pd.Series(self.predictions[name])
                
                metrics = {}
                
                # 매매 포지션 초기화
                positions = pd.Series(0, index=signals.index)
                positions.loc[signals == 'BUY'] = 1
                positions.loc[signals == 'SELL'] = -1
                
                # 해당 기간의 수익률 계산
                period_returns = self.returns[signals.index]
                strategy_returns = positions * period_returns
                strategy_returns = strategy_returns.fillna(0)  # NaN 값 처리
                
                # 누적 수익률 계산
                cumulative_returns = (1 + strategy_returns).cumprod()
                
                # 안전한 성과 지표 계산
                try:
                    # 누적 수익률
                    if len(cumulative_returns) > 0:
                        metrics['누적 수익률'] = float(cumulative_returns.iloc[-1] - 1)
                    else:
                        metrics['누적 수익률'] = 0.0
                    
                    # 평균 수익률
                    metrics['평균 수익률'] = float(strategy_returns.mean()) if len(strategy_returns) > 0 else 0.0
                    
                    # 샤프 비율 계산
                    returns_std = float(strategy_returns.std())
                    returns_mean = float(strategy_returns.mean())
                    
                    if returns_std > 0 and not np.isnan(returns_std) and not np.isnan(returns_mean):
                        metrics['샤프 비율'] = float(np.sqrt(252) * returns_mean / returns_std)
                    else:
                        metrics['샤프 비율'] = 0.0
                    
                    # 승률 계산
                    positive_returns = strategy_returns[strategy_returns > 0]
                    total_trades = len(strategy_returns[strategy_returns != 0])
                    
                    if total_trades > 0:
                        metrics['승률'] = float(len(positive_returns) / total_trades)
                    else:
                        metrics['승률'] = 0.0
                        
                    # 최대 손실/수익
                    metrics['최대 손실'] = float(strategy_returns.min()) if len(strategy_returns) > 0 else 0.0
                    metrics['최대 수익'] = float(strategy_returns.max()) if len(strategy_returns) > 0 else 0.0
                    
                except Exception as calc_error:
                    st.warning(f"{name} 모델의 성과 지표 계산 중 오류 발생: {str(calc_error)}")
                    metrics = {
                        '누적 수익률': 0.0,
                        '평균 수익률': 0.0,
                        '샤프 비율': 0.0,
                        '승률': 0.0,
                        '최대 손실': 0.0,
                        '최대 수익': 0.0
                    }
                
                # NaN 체크 및 처리
                for key in metrics:
                    if np.isnan(metrics[key]) or np.isinf(metrics[key]):
                        metrics[key] = 0.0
                
                self.performance_metrics[name] = metrics
            
            # 성능 지표 데이터프레임 생성
            metrics_df = pd.DataFrame.from_dict(self.performance_metrics, orient='index')
            
            # NaN 값 처리
            metrics_df = metrics_df.fillna(0.0)
            
            # 1. 누적 수익률 비교
            fig1 = go.Figure()
            for name in metrics_df.index:
                fig1.add_trace(
                    go.Bar(
                        name=name,
                        x=['누적 수익률'],
                        y=[metrics_df.loc[name, '누적 수익률'] * 100],
                        text=[f"{metrics_df.loc[name, '누적 수익률']*100:.2f}%"],
                        textposition='auto'
                    )
                )
            
            fig1.update_layout(
                title='모델별 누적 수익률 비교',
                yaxis_title='수익률 (%)',
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig1)
            
            # 2. 성능 지표 포맷팅 및 표시
            formatted_metrics = metrics_df.copy()
            formatted_metrics['누적 수익률'] = formatted_metrics['누적 수익률'].apply(lambda x: f"{x*100:.2f}%")
            formatted_metrics['평균 수익률'] = formatted_metrics['평균 수익률'].apply(lambda x: f"{x*100:.2f}%")
            formatted_metrics['승률'] = formatted_metrics['승률'].apply(lambda x: f"{x*100:.2f}%")
            formatted_metrics['샤프 비율'] = formatted_metrics['샤프 비율'].apply(lambda x: f"{x:.2f}")
            formatted_metrics['최대 손실'] = formatted_metrics['최대 손실'].apply(lambda x: f"{x*100:.2f}%")
            formatted_metrics['최대 수익'] = formatted_metrics['최대 수익'].apply(lambda x: f"{x*100:.2f}%")
            
            st.write("### 모델별 성능 지표 비교")
            st.dataframe(formatted_metrics)
            
            # 3. 승률과 샤프 비율 비교
            fig2 = go.Figure()
            for name in metrics_df.index:
                fig2.add_trace(
                    go.Scatter(
                        x=[metrics_df.loc[name, '승률']],
                        y=[metrics_df.loc[name, '샤프 비율']],
                        mode='markers+text',
                        name=name,
                        text=[name],
                        textposition="top center",
                        marker=dict(size=15)
                    )
                )
            
            fig2.update_layout(
                title='승률 vs 샤프 비율',
                xaxis_title='승률',
                yaxis_title='샤프 비율',
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig2)
            
            # 4. 모델 순위 계산 (0으로 나누기 방지)
            metrics_df['종합 점수'] = (
                metrics_df['누적 수익률'].fillna(0) * 0.4 +
                metrics_df['승률'].fillna(0) * 0.3 +
                (metrics_df['샤프 비율'].fillna(0) / 10) * 0.3
            )
            
            ranks_df = pd.DataFrame({
                '모델': metrics_df.index,
                '종합 점수': metrics_df['종합 점수']
            }).sort_values('종합 점수', ascending=False)
            
            st.write("### 모델 종합 순위")
            st.dataframe(
                ranks_df.style.background_gradient(
                    cmap='YlOrRd',
                    subset=['종합 점수']
                ).format({'종합 점수': '{:.4f}'})
            )
            
            return self.performance_metrics
            
        except Exception as e:
            st.error(f"모델 비교 분석 중 오류 발생: {str(e)}")
            st.write("Error details:", str(e))
            return None

# 메인 분석 부분 수정
if st.sidebar.button("분석 시작"):
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    if stock_data is not None:
        with st.spinner("데이터 분석 중..."):
            try:
                # 기술적 지표 계산
                tech_analyzer = TechnicalAnalyzer(stock_data)
                
                # 확률적 분석
                prob_analyzer = ProbabilisticAnalyzer(tech_analyzer.data)
                
                # 모델 학습
                if prob_analyzer.train_all_models():
                    # 모델 성능 비교 분석
                    st.subheader("📊 모델 성능 비교 분석")
                    prob_analyzer.compare_models()
                    
                    # 특성 중요도 분석
                    st.subheader("📈 특성 중요도 분석")
                    prob_analyzer.plot_feature_importance()
                    
                    # ROC 곡선 분석
                    st.subheader("📉 ROC 곡선 분석")
                    prob_analyzer.plot_roc_curves()
                    
                    # 모델별 매매 신호 분석
                    st.subheader("🤖 모델별 매매 신호 분석")
                    model_analyzer = ModelSignalAnalyzer(prob_analyzer.models, tech_analyzer.data, prob_analyzer.predictions)
                    model_analyzer.analyze_signals()
                    
                    # 모델 비교 결과 표시
                    fig, matrix_df = model_analyzer.plot_model_comparison()
                    
                    # 매트릭스 표시
                    st.write("### 모델별 매매 신호 확률")
                    st.dataframe(matrix_df.style.apply(lambda x: ['background-color: #e6ffe6' if v == 'BUY'
                                                                else 'background-color: #ffe6e6' if v == 'SELL'
                                                                else 'background-color: #f2f2f2'
                                                                for v in x], subset=['Current Signal']))
                    
                    # 히트맵 표시
                    st.plotly_chart(fig)
                    
                    # 모델 앙상블 기반 최종 추천
                    current_signals = matrix_df['Current Signal'].value_counts()
                    st.write("### 모델 앙상블 기반 최종 추천")
                    
                    total_models = len(matrix_df)
                    buy_strength = current_signals.get('BUY', 0) / total_models * 100
                    sell_strength = current_signals.get('SELL', 0) / total_models * 100
                    hold_strength = current_signals.get('HOLD', 0) / total_models * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("매수 신호 강도", f"{buy_strength:.1f}%")
                    with col2:
                        st.metric("매도 신호 강도", f"{sell_strength:.1f}%")
                    with col3:
                        st.metric("관망 신호 강도", f"{hold_strength:.1f}%")
                    
                    # 가중치 기반 최종 추천
                    weighted_signals = {}
                    for _, row in matrix_df.iterrows():
                        accuracy = float(row['Accuracy'].rstrip('%')) / 100
                        signal = row['Current Signal']
                        weighted_signals[signal] = weighted_signals.get(signal, 0) + accuracy
                    
                    max_signal = max(weighted_signals.items(), key=lambda x: x[1])
                    total_weight = sum(weighted_signals.values())
                    confidence = (max_signal[1] / total_weight) * 100
                    
                    signal_color = {
                        'BUY': 'green',
                        'SELL': 'red',
                        'HOLD': 'blue'
                    }
                    
                    st.markdown(f"### 가중치 기반 최종 추천: "
                              f"<span style='color: {signal_color[max_signal[0]]}'>{max_signal[0]}</span> "
                              f"(신뢰도: {confidence:.1f}%)", unsafe_allow_html=True)
                    
                    # 선형 회귀 분석 결과 시각화
                    prob_analyzer.plot_regression_analysis()
                    
            except Exception as e:
                st.error(f"분석 중 오류 발생: {str(e)}")
    else:
        st.error(
            "데이터를 불러오는데 실패했습니다. "
            "티커 심볼을 확인해주세요."
        ) 

