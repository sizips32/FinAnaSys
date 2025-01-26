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
import sys
from sklearn.metrics import precision_score, recall_score, f1_score

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
    signals = pd.DataFrame(index=data.index[-len(pred_mean):])  # 수정된 부분
    
    # 기술적 지표 기반 신호
    signals['RSI_signal'] = np.where(data['RSI'].iloc[-len(pred_mean):] < 30, 1, 
                                   np.where(data['RSI'].iloc[-len(pred_mean):] > 70, -1, 0))
    
    signals['MACD_signal'] = np.where(data['MACD'].iloc[-len(pred_mean):] > data['Signal'].iloc[-len(pred_mean):], 1, 
                                     np.where(data['MACD'].iloc[-len(pred_mean):] < data['Signal'].iloc[-len(pred_mean):], -1, 0))
    
    # 볼린저 밴드 기반 신호
    signals['BB_signal'] = np.where(data['Close'].iloc[-len(pred_mean):] < data['BB_lower'].iloc[-len(pred_mean):], 1,
                                   np.where(data['Close'].iloc[-len(pred_mean):] > data['BB_upper'].iloc[-len(pred_mean):], -1, 0))
    
    # 확률적 예측 기반 신호
    current_price = data['Close'].iloc[-len(pred_mean):]
    confidence_interval = 1.96 * pred_std
    
    signals['Pred_signal'] = np.where(pred_mean - confidence_interval > current_price, 1,
                                     np.where(pred_mean + confidence_interval < current_price, -1, 0))
    
    # 종합 신호
    signals['Final_signal'] = signals.mean(axis=1)
    
    return signals

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
    def __init__(self, data, test_size=0.2):
        self.data = data
        self.test_size = test_size
        self.models = {}
        self.predictions = {}
        self.prepare_data()
        
    def prepare_data(self):
        """데이터 전처리 및 레이블 생성"""
        # 기술적 지표 중 사용할 특성 선택
        feature_columns = [col for col in self.data.columns 
                         if col not in ['Date', 'Target', 'Label', 'Close']]
        
        # 다음날 수익률 계산 (레이블)
        self.data['Target'] = self.data['Close'].shift(-1) / self.data['Close'] - 1
        self.data['Label'] = np.where(self.data['Target'] > 0, 1, 0)
        
        # NaN 제거
        self.data.dropna(inplace=True)
        
        # 특성 스케일링
        self.scaler = MinMaxScaler()
        self.features_scaled = self.scaler.fit_transform(self.data[feature_columns])
        
        # 학습/테스트 분할
        split_idx = int(len(self.features_scaled) * (1 - self.test_size))
        self.X_train = self.features_scaled[:split_idx]
        self.X_test = self.features_scaled[split_idx:]
        self.y_train = self.data['Label'].iloc[:split_idx]
        self.y_test = self.data['Label'].iloc[split_idx:]
    
    def train_models(self):
        """다양한 모델 학습"""
        try:
            # 기본 모델 정의 (RandomForest만 사용)
            models = {
                'RandomForest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            }
            
            # 추가 모델 사용 가능 여부 확인 및 추가
            try:
                from xgboost import XGBClassifier
                models['XGBoost'] = XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            except ImportError:
                st.warning("XGBoost를 사용할 수 없습니다. RandomForest만 사용됩니다.")
            
            try:
                from lightgbm import LGBMClassifier
                models['LightGBM'] = LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            except ImportError:
                st.warning("LightGBM을 사용할 수 없습니다.")
            
            # 각 모델 학습 및 예측
            for name, model in models.items():
                st.info(f"{name} 모델 학습 중...")
                model.fit(self.X_train, self.y_train)
                self.models[name] = model
                
                # 예측 확률 계산
                if hasattr(model, 'predict_proba'):
                    self.predictions[name] = model.predict_proba(self.X_test)[:, 1]
                else:
                    self.predictions[name] = model.predict(self.X_test)
                
                st.success(f"{name} 모델 학습 완료")
            
            return True
            
        except Exception as e:
            st.error(f"모델 학습 중 오류 발생: {str(e)}")
            st.info("""
            오류 해결을 위한 제안:
            1. 데이터의 형식과 크기 확인
            2. 메모리 사용량 확인
            3. 필요한 라이브러리 설치 여부 확인
            """)
            return False
    
    def calculate_ensemble_probability(self):
        """앙상블 예측 확률 계산"""
        if not self.predictions:
            st.error("먼저 모델을 학습시켜주세요.")
            return None, None
        
        all_predictions = np.array(list(self.predictions.values()))
        mean_prob = np.mean(all_predictions, axis=0)
        std_prob = np.std(all_predictions, axis=0)
        
        return mean_prob, std_prob
    
    def get_trading_signals(self, threshold=0.6):
        """트레이딩 신호 생성"""
        mean_prob, std_prob = self.calculate_ensemble_probability()
        if mean_prob is None:
            return None
        
        signals = pd.DataFrame(index=self.data.index[-len(mean_prob):])
        signals['Probability'] = mean_prob
        signals['Uncertainty'] = std_prob
        signals['Signal'] = np.where(mean_prob > threshold, 1,
                                   np.where(mean_prob < (1-threshold), -1, 0))
        
        return signals
    
    def get_model_metrics(self):
        """모델별 성능 지표 계산"""
        metrics = {}
        for name, predictions in self.predictions.items():
            metrics[name] = {
                '정확도': accuracy_score(self.y_test, predictions > 0.5),
                '정밀도': precision_score(self.y_test, predictions > 0.5),
                '재현율': recall_score(self.y_test, predictions > 0.5),
                'F1 Score': f1_score(self.y_test, predictions > 0.5)
            }
        return pd.DataFrame(metrics).T

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
                if prob_analyzer.train_models():
                    # 트레이딩 신호 생성
                    signals = prob_analyzer.get_trading_signals()
                    
                    if signals is not None:
                        # 결과 시각화
                        st.subheader("📊 투자 의사결정 대시보드")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            final_prob = signals['Probability'].iloc[-1]
                            signal_text = "매수" if final_prob > 0.6 else "매도" if final_prob < 0.4 else "관망"
                            st.metric("투자 신호", signal_text)
                            st.metric("매수 확률", f"{final_prob:.1%}")
                        
                        with col2:
                            uncertainty = signals['Uncertainty'].iloc[-1]
                            st.metric("불확실성", f"{uncertainty:.1%}")
                        
                        with col3:
                            avg_prob = signals['Probability'].mean()
                            st.metric("평균 매수 확률", f"{avg_prob:.1%}")
                        
                        # 모델 성능 지표 표시
                        st.subheader("🤖 모델 성능 분석")
                        metrics_df = prob_analyzer.get_model_metrics()
                        st.dataframe(metrics_df.style.format("{:.2%}"))
                        
                        # 기술적 지표 시각화
                        st.subheader("📈 주요 기술적 지표")
                        latest_data = tech_analyzer.data.iloc[-1]
                        tech_metrics = {
                            'RSI': f"{latest_data['RSI']:.2f}",
                            'MACD': f"{latest_data['MACD']:.2f}",
                            'BB Width': f"{latest_data['BB_Width']:.2f}",
                            'ATR': f"{latest_data['ATR']:.2f}"
                        }
                        st.write(tech_metrics)

            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
                st.info("""
                오류 해결을 위한 제안:
                1. 충분한 데이터가 있는지 확인
                2. 날짜 범위 조정
                3. 다른 주식 심볼 시도
                """)
    else:
        st.error(
            "데이터를 불러오는데 실패했습니다. "
            "티커 심볼을 확인해주세요."
        ) 

