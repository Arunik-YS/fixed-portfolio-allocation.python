import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

st.set_page_config(page_title="장기 황금 비중 진단기", layout="centered")
st.markdown("<style>.stNumberInput, .stTextInput { margin-bottom: -15px; }</style>", unsafe_allow_html=True)

# 자산군 고정 (VOO 대신 단가 부담이 적은 SPLG 반영)
TARGET_TICKERS = ["SPLG", "TLT", "IAU"]
RISK_FREE_RATE = 0.03 # 무위험수익률 3%

# 1. 10년치 장기 데이터 수집 (티커 리스트를 매개변수로 받아 캐시 꼬임 방지)
@st.cache_data(ttl="7d", show_spinner=False)
def fetch_long_term_data(tickers):
    df = yf.download(list(tickers), period="10y", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        return df['Close'].dropna()
    else:
        return df[['Close']].dropna()

# 2. 최근 5일 데이터로 현재가 가져오기 (10분만 캐싱하여 실시간성 확보 및 안전망 구축)
@st.cache_data(ttl="10m", show_spinner=False)
def get_current_prices(tickers):
    prices = {}
    df = yf.download(list(tickers), period="5d", progress=False)
    for ticker in tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                prices[ticker] = float(df['Close'][ticker].iloc[-1])
            else:
                prices[ticker] = float(df['Close'].iloc[-1])
        except Exception:
            # 혹시 데이터가 누락되더라도 앱이 멈추지 않도록 0.0으로 예외 처리
            prices[ticker] = 0.0
    return prices

# 3. 포트폴리오 연산 로직
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std_dev, (returns - RISK_FREE_RATE) / std_dev

def negative_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

# --- 황금 비중 백그라운드 계산 (캐시 키로 활용하기 위해 튜플로 전달) ---
tickers_tuple = tuple(TARGET_TICKERS)

with st.spinner("과거 10년간의 데이터를 분석하여 최적의 황금 비중을 계산 중입니다..."):
    history_10y = fetch_long_term_data(tickers_tuple)
    log_returns = np.log(history_10y / history_10y.shift(1)).dropna()
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    
    num_assets = len(TARGET_TICKERS)
    bounds = tuple((0.15, 1.0) for _ in range(num_assets)) # 최소 15% 편입 조건 유지
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    opt_result = minimize(negative_sharpe, num_assets * [1./num_assets], 
                          args=(mean_returns, cov_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = {TARGET_TICKERS[i]: opt_result.x[i] for i in range(num_assets)}
    current_prices = get_current_prices(tickers_tuple)

# ==========================================
# UI 렌더링
# ==========================================
st.title("🛡️ 장기 자산 배분 진단기")
st.caption("SPLG(주식), TLT(장기채), IAU(금) 조합의 10년 백테스트 기반 리밸런싱")

# 1. 황금 비중 안내
st.subheader("🎯 10년 최적화 타겟 비중")
st.info("시장의 단기 노이즈를 무시하고 평생 유지해야 할 전략적 목표 비중입니다.")
col_w1, col_w2, col_w3 = st.columns(3)
col_w1.metric("SPLG (S&P 500)", f"{optimal_weights['SPLG']*100:.1f}%")
col_w2.metric("TLT (미 장기채)", f"{optimal_weights['TLT']*100:.1f}%")
col_w3.metric("IAU (금)", f"{optimal_weights['IAU']*100:.1f}%")

st.divider()

# 2. 내 계좌 입력
st.subheader("💼 현재 내 계좌 상태 입력")
shares_input = {}
with st.container(border=True):
    cols = st.columns(3)
    for i, ticker in enumerate(TARGET_TICKERS):
        with cols[i]:
            shares_input[ticker] = st.number_input(f"{ticker} 보유 주수", min_value=0, step=1, key=f"s_{ticker}")
            # 가격 데이터가 정상적으로 존재할 때만 출력하도록 방어 코드 추가
            price_display = current_prices.get(ticker, 0.0)
            st.caption(f"현재가: ${price_display:.2f}")

add_cash = st.number_input("💵 리밸런싱에 투입할 추가 현금 ($)", min_value=0.0, step=100.0)

# 3. 진단 및 리밸런싱 실행
if st.button("내 포트폴리오 진단하기", use_container_width=True, type="primary"):
    current_values = {t: shares_input[t] * current_prices.get(t, 0.0) for t in TARGET_TICKERS}
    total_eval = sum(current_values.values())
    total_budget = total_eval + add_cash
    
    if total_budget == 0:
        st.warning("보유 주수나 추가 현금을 입력해주세요.")
    else:
        st.subheader("📊 리밸런싱 처방전")
        
        results = []
        needs_rebalancing = False
        
        for t in TARGET_TICKERS:
            curr_price = current_prices.get(t, 0.0)
            curr_weight = current_values[t] / total_budget if total_budget > 0 else 0
            target_weight = optimal_weights[t]
            weight_diff = curr_weight - target_weight
            
            target_value = total_budget * target_weight
            # 현재가가 0인 예외 상황 방어
            target_shares = target_value / curr_price if curr_price > 0 else 0
            share_diff = target_shares - shares_input[t]
            
            is_out_of_band = abs(weight_diff) >= 0.05
            if is_out_of_band:
                needs_rebalancing = True
                
            if share_diff > 0.5:
                action = f"🟢 {int(share_diff)}주 매수"
            elif share_diff < -0.5:
                action = f"🔴 {int(abs(share_diff))}주 매도"
            else:
                action = "⚪ 유지"

            results.append({
                "종목": t,
                "현재 비중": f"{curr_weight*100:.1f}%",
                "목표 비중": f"{target_weight*100:.1f}%",
                "비중 오차": f"{weight_diff*100:+.1f}%",
                "진단": "⚠️ 이탈" if is_out_of_band else "✅ 정상",
                "액션 플랜": action
            })

        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        
        if needs_rebalancing:
            st.error("🚨 5% 이상 비중이 틀어진 자산이 있습니다. 위 액션 플랜에 따라 즉시 매매를 진행하세요.")
        else:
            st.success("🎉 모든 자산이 5% 오차범위 내에 있습니다. 이번 달은 매매 없이 유지(Hold)를 권장합니다.")
