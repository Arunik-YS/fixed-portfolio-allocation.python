import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 1. 모바일 친화적 페이지 설정 (좌우 여백을 통제하여 모바일 핏 맞춤)
st.set_page_config(page_title="Portfolio Optimizer", layout="centered")

# 2. 데이터 수집 함수 (캐싱을 통해 모바일 로딩 속도 최적화)
@st.cache_data(show_spinner=False)
def fetch_data(tickers, period="3mo"):
    # yfinance를 통해 수정종가(Adj Close) 데이터 다운로드
    data = yf.download(tickers, period=period)['Adj Close']
    return data

# 3. 포트폴리오 성과 계산 (수익률, 표준편차, 샤프지수)
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    # 영업일 기준 252일을 곱해 연환산(Annualized) 수치 도출
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (returns - risk_free_rate) / std_dev
    return returns, std_dev, sharpe_ratio

# 4. 목적함수 (샤프지수를 최대화하기 위해 음의 샤프지수를 최소화)
def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

# 5. 최적화 실행 함수 (SLSQP 알고리즘)
def maximize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    
    # 제약조건: 모든 종목 비중의 합은 1 (100%)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # 각 종목의 비중 범위: 0.0 ~ 1.0 (공매도 금지)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # 초기값: 모든 종목에 동일한 비중으로 분산 투자
    init_guess = num_assets * [1. / num_assets,]
    
    result = minimize(negative_sharpe, init_guess, args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# ==========================================
# 모바일 UI / UX 렌더링 파트
# ==========================================
st.title("📈 최적 배분 계산기")

# 모바일 화면을 고려해 Expander(접기/펴기) 활용하여 공간 절약
with st.expander("⚙️ 분석 설정", expanded=True):
    # 입력이 편하도록 기본값으로 모니터링하시는 ETF 세팅
    default_tickers = ["TQQQ", "KORU", "SGOV"]
    user_input = st.text_input("티커 입력 (쉼표로 구분)", value=", ".join(default_tickers))
    
    # 국고채 3년물 등 무위험수익률 (기본값 3.0%)
    risk_free_input = st.number_input("무위험수익률 (%)", value=3.0, step=0.1) / 100

tickers = [ticker.strip().upper() for ticker in user_input.split(",") if ticker.strip()]

if st.button("최적 비중 계산하기", use_container_width=True): # 버튼을 화면 너비에 꽉 차게 (모바일 터치 최적화)
    if len(tickers) < 2 or len(tickers) > 5:
        st.warning("티커는 최소 2개에서 최대 5개까지 입력해주세요.")
    else:
        with st.spinner("데이터를 분석 중입니다..."):
            # 데이터 로드 및 전처리
            data = fetch_data(tickers)
            
            # 일간 로그수익률 계산
            log_returns = np.log(data / data.shift(1)).dropna()
            
            # 평균 수익률 및 공분산 행렬 계산
            mean_returns = log_returns.mean()
            cov_matrix = log_returns.cov()
            
            # 최적화 수행
            opt_result = maximize_sharpe_ratio(mean_returns, cov_matrix, risk_free_input)
            
            # 결과 도출
            opt_weights = opt_result.x
            opt_ret, opt_std, opt_sharpe = portfolio_performance(opt_weights, mean_returns, cov_matrix, risk_free_input)
            
            # UI 출력 (모바일 가독성을 위해 상하 배치)
            st.success("✅ 최적화 완료!")
            
            st.subheader("🏆 최대 샤프지수 포트폴리오")
            st.metric(label="예상 연환산 수익률", value=f"{opt_ret * 100:.2f}%")
            st.metric(label="예상 연환산 변동성", value=f"{opt_std * 100:.2f}%")
            st.metric(label="샤프지수", value=f"{opt_sharpe:.2f}")
            
            st.divider()
            
            st.subheader("⚖️ 종목별 최적 투자 비중")
            # 비중 결과를 DataFrame으로 변환 후 시각화
            weight_df = pd.DataFrame({"비중 (%)": np.round(opt_weights * 100, 2)}, index=tickers)
            st.dataframe(weight_df, use_container_width=True)