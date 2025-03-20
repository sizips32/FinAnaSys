import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go

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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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
        # 종목 코드에서 공백 제거
        ticker = ticker.strip()
        
        if not ticker:
            st.error("종목 코드를 입력해주세요.")
            return None
        
        # 미국 주요 주식 목록
        US_STOCKS = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'META': 'Meta Platforms Inc.',
            'TSLA': 'Tesla Inc.',
            'NVDA': 'NVIDIA Corporation'
        }
        
        # 미국 주식인지 확인
        is_us_stock = ticker.upper() in US_STOCKS or '.' in ticker
        
        if is_us_stock:
            try:
                # 미국 주식 데이터 가져오기
                df = fdr.DataReader(ticker, start, end)
                if df is None or df.empty:
                    st.error(f"{ticker}에 대한 주가 데이터가 없습니다.")
                    if ticker.upper() in US_STOCKS:
                        st.info(f"'{US_STOCKS[ticker.upper()]}' ({ticker.upper()})의 데이터를 찾을 수 없습니다.")
                    else:
                        st.info("미국 주식 심볼을 확인하고 다시 시도해주세요.")
                    return None
                
                stock_name = US_STOCKS.get(ticker.upper(), ticker.upper())
                st.success(f"미국 주식 '{stock_name}' ({ticker.upper()}) 데이터를 성공적으로 불러왔습니다.")
                
            except Exception as us_error:
                st.error(f"미국 주식 데이터를 가져오는데 실패했습니다: {str(us_error)}")
                st.info("주식 심볼을 확인하고 다시 시도해주세요.")
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
            
            # 이동평균 계산
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_60'] = df['Close'].rolling(window=60).mean()
            
            # RSI 계산
            delta = df['Close'].diff()
            gain = delta.copy()
            loss = delta.copy()
            for i in range(len(delta)):
                if delta[i] >= 0:
                    gain[i] = delta[i]
                    loss[i] = 0
                else:
                    gain[i] = 0
                    loss[i] = abs(delta[i])
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD 계산
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # 추가 지표 계산
            df['ROC'] = df['Close'].pct_change(periods=12) * 100
            df['MOM'] = df['Close'].diff(periods=10)
            df['ATR'] = self.calculate_atr(df)
            df['OBV'] = self.calculate_obv(df)
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Price_Change'] = df['Close'].pct_change()
            df['Volatility'] = df['Price_Change'].rolling(window=20).std()
            
            # 결측치 처리
            df.fillna(method='bfill', inplace=True)
            df.fillna(method='ffill', inplace=True)
            
            self.data = df
            self.features = [
                'SMA_5', 'SMA_20', 'SMA_60', 'RSI', 'MACD', 'Signal',
                'ROC', 'MOM', 'ATR', 'OBV', 'Volume_MA',
                'Price_Change', 'Volatility'
            ]
            
        except Exception as e:
            st.error(f"기술적 지표 계산 중 오류 발생: {str(e)}")
    
    def calculate_atr(self, df):
        try:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            return true_range.rolling(14).mean()
        except Exception as e:
            st.error(f"ATR 계산 중 오류 발생: {str(e)}")
            return pd.Series(0, index=df.index)
    
    def calculate_obv(self, df):
        try:
            obv = np.zeros(len(df))
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv[i] = obv[i-1] + df['Volume'].iloc[i]
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv[i] = obv[i-1] - df['Volume'].iloc[i]
                else:
                    obv[i] = obv[i-1]
            return pd.Series(obv, index=df.index)
        except Exception as e:
            st.error(f"OBV 계산 중 오류 발생: {str(e)}")
            return pd.Series(0, index=df.index)

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
        try:
            performance_metrics = {}
            for name, model in self.models.items():
                metrics = {}
                
                if name == 'Linear Regression':
                    y_pred = model.predict(self.X_test_reg)
                    metrics['R2'] = r2_score(self.y_test_reg, y_pred)
                    metrics['MAE'] = mean_absolute_error(self.y_test_reg, y_pred)
                    metrics['RMSE'] = np.sqrt(mean_squared_error(self.y_test_reg, y_pred))
                    
                    # 방향성 정확도 계산
                    correct_direction = 0
                    total_predictions = len(y_pred)
                    y_test_reg_values = self.y_test_reg.values  # numpy array로 변환
                    
                    for i in range(total_predictions):
                        if (y_pred[i] > 0 and y_test_reg_values[i] > 0) or \
                           (y_pred[i] < 0 and y_test_reg_values[i] < 0):
                            correct_direction += 1
                    
                    metrics['방향성 정확도'] = (correct_direction / total_predictions) * 100
                
                elif name == 'LSTM':
                    y_pred = (model.predict(self.X_test_seq) > 0.5).astype(int)
                    metrics['정확도'] = accuracy_score(self.y_test_seq, y_pred)
                    metrics['정밀도'] = precision_score(self.y_test_seq, y_pred)
                    metrics['재현율'] = recall_score(self.y_test_seq, y_pred)
                    metrics['F1 점수'] = f1_score(self.y_test_seq, y_pred)
                
                else:
                    y_pred = model.predict(self.X_test_scaled)
                    metrics['정확도'] = accuracy_score(self.y_test, y_pred)
                    metrics['정밀도'] = precision_score(self.y_test, y_pred)
                    metrics['재현율'] = recall_score(self.y_test, y_pred)
                    metrics['F1 점수'] = f1_score(self.y_test, y_pred)
                
                performance_metrics[name] = metrics
            
            # 결과 시각화
            st.write("### 모델별 성능 지표")
            for name, metrics in performance_metrics.items():
                st.write(f"\n#### {name}")
                metrics_df = pd.DataFrame([metrics]).T
                metrics_df.columns = ['값']
                st.dataframe(
                    metrics_df.style
                    .format('{:.4f}')
                    .background_gradient(cmap='YlOrRd')
                )
            
            return performance_metrics
            
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
        try:
            if 'Linear Regression' not in self.models:
                st.warning("선형 회귀 모델이 없습니다.")
                return
                
            st.write("### 선형 회귀 분석 결과")
            
            try:
                # DataFrame을 1차원 배열로 변환
                if isinstance(self.y_test_reg, pd.DataFrame):
                    y_test_reg = self.y_test_reg.iloc[:, 0].values
                elif isinstance(self.y_test_reg, pd.Series):
                    y_test_reg = self.y_test_reg.values
                else:
                    y_test_reg = np.array(self.y_test_reg).ravel()
                
                # X_test_reg가 이미 numpy array이므로 그대로 사용
                X_test_reg = self.X_test_reg
                
                # 예측값 계산
                try:
                    y_pred = self.models['Linear Regression'].predict(X_test_reg)
                    
                    # 차원 일치 확인 및 조정
                    if len(y_pred.shape) > 1:
                        y_pred = y_pred.ravel()
                    if len(y_test_reg.shape) > 1:
                        y_test_reg = y_test_reg.ravel()
                    
                    # NaN 값 필터링
                    valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_test_reg)
                    y_pred = y_pred[valid_mask]
                    y_test_reg = y_test_reg[valid_mask]
                    
                    if len(y_pred) == 0 or len(y_test_reg) == 0:
                        st.warning("회귀 분석을 위한 유효한 데이터가 없습니다.")
                        return
                    
                    # 데이터 범위 계산
                    min_val = float(min(np.min(y_test_reg), np.min(y_pred)))
                    max_val = float(max(np.max(y_test_reg), np.max(y_pred)))
                    
                    # 산점도 생성
                    fig = go.Figure()
                    
                    # 예측값 vs 실제값 산점도
                    fig.add_trace(go.Scatter(
                        x=y_test_reg,
                        y=y_pred,
                        mode='markers',
                        name='예측값',
                        marker=dict(
                            color='blue',
                            size=8,
                            opacity=0.6
                        )
                    ))
                    
                    # 이상적인 예측선 추가
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='이상적인 예측',
                        line=dict(
                            color='red',
                            dash='dash'
                        )
                    ))
                    
                    # 레이아웃 설정
                    fig.update_layout(
                        title='실제값 vs 예측값 비교',
                        xaxis_title='실제 수익률',
                        yaxis_title='예측 수익률',
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
                    
                    # 그래프 표시
                    st.plotly_chart(fig)
                    
                    # 회귀 분석 메트릭스 계산
                    try:
                        metrics = {
                            'R2 Score': r2_score(y_test_reg, y_pred),
                            'MAE': mean_absolute_error(y_test_reg, y_pred),
                            'MSE': mean_squared_error(y_test_reg, y_pred),
                            'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred))
                        }
                        
                        # 메트릭스 표시
                        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['값'])
                        st.write("### 회귀 분석 성능 지표")
                        st.dataframe(
                            metrics_df.style
                            .format({'값': '{:.4f}'})
                            .background_gradient(cmap='YlOrRd')
                        )
                    except Exception as metric_error:
                        st.error(f"성능 지표 계산 중 오류 발생: {str(metric_error)}")
                        
                except Exception as pred_error:
                    st.error(f"예측 수행 중 오류 발생: {str(pred_error)}")
                    
            except Exception as data_error:
                st.error(f"데이터 전처리 중 오류 발생: {str(data_error)}")
                
        except Exception as e:
            st.error(f"회귀 분석 시각화 중 오류 발생: {str(e)}")
            st.write("디버그 정보:")
            st.write(f"y_test_reg 타입: {type(self.y_test_reg)}")
            st.write(f"y_test_reg 형태: {self.y_test_reg.shape if hasattr(self.y_test_reg, 'shape') else 'shape 없음'}")
            st.write(f"X_test_reg 타입: {type(self.X_test_reg)}")
            st.write(f"X_test_reg 형태: {self.X_test_reg.shape if hasattr(self.X_test_reg, 'shape') else 'shape 없음'}")
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
                    'Accuracy': f"{metrics['승률']:.2f}%",
                    'Cumulative Return': f"{metrics['누적 수익률']*100:.2f}%",
                    'Sharpe Ratio': f"{metrics['샤프 비율']:.2f}"
                }
                comparison_data.append(row_data)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            if comparison_df.empty:
                return None, comparison_df
            
            # 시각화
            fig = go.Figure()
            colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
            
            # 각 신호별 막대 그래프 생성
            for signal in ['BUY', 'SELL', 'HOLD']:
                mask = comparison_df['Current Signal'] == signal
                if mask.any():
                    fig.add_trace(go.Bar(
                        name=signal,
                        x=comparison_df[mask]['Model'],
                        y=[float(acc.strip('%')) for acc in comparison_df[mask]['Accuracy']],
                        marker_color=colors[signal]
                    ))
            
            # 레이아웃 설정
            fig.update_layout(
                title='모델별 성능 비교',
                xaxis_title='모델',
                yaxis_title='승률 (%)',
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
        self.returns = pd.Series(self.data['Close'].pct_change().fillna(0))
    
    def analyze_signals(self):
        try:
            metrics = {}
            for name, signals in self.predictions.items():
                if signals is None or signals.empty:
                    continue
                
                positions = []
                returns_list = []
                
                for idx, signal in signals.items():
                    if signal == 'BUY':
                        positions.append(1)
                    elif signal == 'SELL':
                        positions.append(-1)
                    else:
                        positions.append(0)
                    
                    if idx in self.returns.index:
                        returns_list.append(self.returns[idx])
                    else:
                        returns_list.append(0)
                
                positions = pd.Series(positions, index=signals.index)
                period_returns = pd.Series(returns_list, index=signals.index)
                
                strategy_returns = positions * period_returns
                strategy_returns = strategy_returns.fillna(0)
                cumulative_returns = (1 + strategy_returns).cumprod()
                
                total_trades = sum(1 for x in positions if x != 0)
                win_trades = sum(1 for r in strategy_returns if r > 0)
                
                metrics_dict = {
                    '승률': float(win_trades / total_trades * 100) if total_trades > 0 else 0.0,
                    '누적 수익률': float(cumulative_returns.iloc[-1] - 1) if len(cumulative_returns) > 0 else 0.0,
                    '샤프 비율': self.calculate_sharpe_ratio(strategy_returns)
                }
                
                metrics[name] = metrics_dict
            
            self.performance_metrics = metrics  # performance_metrics 업데이트
            self.plot_performance_comparison()
            return metrics
            
        except Exception as e:
            st.error(f"신호 분석 중 오류 발생: {str(e)}")
            return None
    
    def calculate_sharpe_ratio(self, returns):
        try:
            if len(returns) == 0 or returns.std() == 0:
                return 0.0
            
            risk_free_rate = 0.02  # 연간 2%
            excess_returns = returns - risk_free_rate/252
            return float(np.sqrt(252) * excess_returns.mean() / returns.std())
            
        except Exception as e:
            st.error(f"샤프 비율 계산 중 오류 발생: {str(e)}")
            return 0.0
    
    def calculate_max_drawdown(self, cumulative_returns):
        try:
            if len(cumulative_returns) == 0:
                return 0.0
            
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            return float(drawdowns.min())
            
        except Exception as e:
            st.error(f"최대 낙폭 계산 중 오류 발생: {str(e)}")
            return 0.0
    
    def plot_performance_comparison(self):
        try:
            if not self.performance_metrics:
                return
            
            # 누적 수익률 비교
            returns_data = []
            for name, metrics in self.performance_metrics.items():
                returns_data.append({
                    '모델': name,
                    '누적 수익률': metrics['누적 수익률'],
                    '승률': metrics['승률'],
                    '샤프 비율': metrics['샤프 비율']
                })
            
            returns_df = pd.DataFrame(returns_data)
            
            # 성능 지표 시각화
            fig = go.Figure()
            metrics_to_plot = ['누적 수익률', '승률', '샤프 비율']
            
            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=returns_df['모델'],
                    y=returns_df[metric],
                    text=[f"{val:.2f}%" if metric != '샤프 비율' else f"{val:.2f}" 
                          for val in returns_df[metric]],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title='모델별 성능 비교',
                xaxis_title='모델',
                yaxis_title='성능 지표',
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig)
            
            # 상세 성능 지표 표시
            st.write("### 상세 성능 지표")
            for name, metrics in self.performance_metrics.items():
                st.write(f"\n#### {name} 모델")
                metrics_df = pd.DataFrame.from_dict(
                    {k: [v] for k, v in metrics.items()},
                    orient='columns'
                )
                st.dataframe(
                    metrics_df.style
                    .format({
                        '누적 수익률': '{:.2f}%',
                        '평균 수익률': '{:.2f}%',
                        '승률': '{:.2f}%',
                        '샤프 비율': '{:.2f}',
                        '최대 낙폭': '{:.2f}%'
                    })
                    .background_gradient(cmap='YlOrRd')
                )
            
        except Exception as e:
            st.error(f"성능 비교 시각화 중 오류 발생: {str(e)}")

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
                    'Accuracy': f"{metrics['승률']:.2f}%",
                    'Cumulative Return': f"{metrics['누적 수익률']*100:.2f}%",
                    'Sharpe Ratio': f"{metrics['샤프 비율']:.2f}"
                }
                comparison_data.append(row_data)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            if comparison_df.empty:
                return None, comparison_df
            
            # 시각화
            fig = go.Figure()
            colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
            
            # 각 신호별 막대 그래프 생성
            for signal in ['BUY', 'SELL', 'HOLD']:
                mask = comparison_df['Current Signal'] == signal
                if mask.any():
                    fig.add_trace(go.Bar(
                        name=signal,
                        x=comparison_df[mask]['Model'],
                        y=[float(acc.strip('%')) for acc in comparison_df[mask]['Accuracy']],
                        marker_color=colors[signal]
                    ))
            
            # 레이아웃 설정
            fig.update_layout(
                title='모델별 성능 비교',
                xaxis_title='모델',
                yaxis_title='승률 (%)',
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
            
            return fig, comparison_df
            
        except Exception as e:
            st.error(f"모델 비교 시각화 중 오류 발생: {str(e)}")
            return None, pd.DataFrame()

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
                                        st.dataframe(matrix_df.style.apply(lambda x: ['background-color: #e6ffe6' if v == 'BUY'
                                                                                    else 'background-color: #ffe6e6' if v == 'SELL'
                                                                                    else 'background-color: #f2f2f2'
                                                                                    for v in x], subset=['Current Signal']))
                                        st.plotly_chart(fig)
                                        
                                        # 앙상블 기반 추천 계산
                                        try:
                                            current_signals = matrix_df['Current Signal'].value_counts()
                                            total_models = len(matrix_df)
                                            
                                            if total_models > 0:
                                                st.write("### 모델 앙상블 기반 최종 추천")
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
                                                
                                                # 가중치 기반 신호 계산
                                                try:
                                                    weighted_signals = {}
                                                    for _, row in matrix_df.iterrows():
                                                        accuracy = float(row['Accuracy'].rstrip('%')) / 100
                                                        signal = row['Current Signal']
                                                        weighted_signals[signal] = weighted_signals.get(signal, 0) + accuracy
                                                    
                                                    if weighted_signals:
                                                        max_signal = max(weighted_signals.items(), key=lambda x: x[1])
                                                        total_weight = sum(weighted_signals.values())
                                                        
                                                        if total_weight > 0:
                                                            confidence = (max_signal[1] / total_weight) * 100
                                                            signal_color = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'blue'}
                                                            st.markdown(f"### 가중치 기반 최종 추천: <span style='color: {signal_color[max_signal[0]]}'>{max_signal[0]}</span> (신뢰도: {confidence:.1f}%)", unsafe_allow_html=True)
                                                except Exception as e:
                                                    st.error(f"가중치 기반 신호 계산 중 오류 발생: {str(e)}")
                                            else:
                                                st.warning("모델 결과가 없어 앙상블 분석을 수행할 수 없습니다.")
                                        except Exception as e:
                                            st.error(f"앙상블 기반 추천 계산 중 오류 발생: {str(e)}")
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
