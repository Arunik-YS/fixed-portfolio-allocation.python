import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

st.set_page_config(page_title="장기 황금 비중 진단기", layout="centered")
st.markdown("<style>.stNumberInput, .stTextInput { margin-bottom: -15px; }</style>", unsafe_allow_html=True)

TARGET_TICKERS = ["SPLG", "TLT", "IAU"]
RISK_FREE_RATE = 0.03

@st.cache_data(ttl="7d", show_spinner=False)
def fetch_long_term_data(tickers):
    df = yf.download(list(tickers), period="10y", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        return df['Close'].dropna()
    else:
        return df[['Close']].dropna()

@st.cache_data(ttl="10m", show_spinner=False)
def get_current_prices(tickers):
    prices = {}
    df = yf.download(list(tickers), period="5d", progress=False)
    for ticker in tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                valid_data = df['Close'][ticker].dropna()
            else:
                valid_data = df['Close'].dropna()
            if not valid_data.empty:
                prices[ticker] = float(valid_data.iloc[-1])
            else:
                prices[ticker] = 0.0
        except Exception:
            prices[ticker] = 0.0
    return prices

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std_dev, (returns - RISK_FREE_RATE) / std_dev

def negative_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

# --- 데이터 연산 코어 ---
tickers_tuple = tuple(TARGET_TICKERS)

with st.spinner("과거 10년간의 데이터를 분석하여 최적의 황금 비중을 계산 중입니다..."):
    history_10y = fetch_long_term_data(tickers_tuple)
    
    # 1. 일반 일간 수익률 계산 (누적 수익률 시뮬레이션용)
    daily_returns = history_10y.pct_change().dropna()
    
    # 2. 로그 수익률 계산 (최적화 알고리즘용)
    log_returns = np.log(history_10y / history_10y.shift(1)).dropna()
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    
    num_assets = len(TARGET_TICKERS)
    bounds = tuple((0.15, 1.0) for _ in range(num_assets))
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
col_w1, col_w2, col_w3 = st.columns(3)
col_w1.metric("SPLG (S&P 500)", f"{optimal_weights['SPLG']*100:.1f}%")
col_w2.metric("TLT (미 장기채)", f"{optimal_weights['TLT']*100:.1f}%")
col_w3.metric("IAU (금)", f"{optimal_weights['IAU']*100:.1f}%")

# [신뢰도 향상 추가 요소 1] 시각적 백테스트 차트 구현
with st.expander("📊 과거 10년 시뮬레이션 및 백테스트 검증", expanded=False):
    st.markdown("#### 💵 10,000달러 투자 시 자산 성장 추이")
    
    # 가중치를 반영한 포트폴리오의 일간 수익률 계산
    portfolio_daily_ret = (
        daily_returns['SPLG'] * optimal_weights['SPLG'] +
        daily_returns['TLT'] * optimal_weights['TLT'] +
        daily_returns['IAU'] * optimal_weights['IAU']
    )
    
    # 누적 수익률 곡선 생성 (초기값 10,000달러 기준)
    cum_portfolio = (1 + portfolio_daily_ret).cumprod() * 10000
    cum_splg = (1 + daily_returns['SPLG']).cumprod() * 10000
    
    # 차트용 데이터프레임 통합
    chart_data = pd.DataFrame({
        "황금비중 포트폴리오": cum_portfolio,
        "S&P 500 (주식 올인)": cum_splg
    }, index=history_10y.index[1:])
    
    # Streamlit 모바일 반응형 라인 차트 출력
    st.line_chart(chart_data)
    
    # [신뢰도 향상 추가 요소 2] 통계 지표 비교 표
    st.markdown("#### 📊 전략별 핵심 통계 비교")
    ret_p, std_p, sharpe_p = portfolio_performance(opt_result.x, mean_returns, cov_matrix)
    
    # MDD(최대낙폭) 단순 계산
    mdd_portfolio = ((cum_portfolio / cum_portfolio.cummax()) - 1).min() * 100
    mdd_splg = ((cum_splg / cum_splg.cummax()) - 1).min() * 100
    
    stats_df = pd.DataFrame({
        "지표": ["예상 연수익률", "연 변동성(위험)", "샤프 지수(가성비)", "최대 낙폭(MDD)"],
        "황금비중 포트폴리오": [f"{ret_p*100:.2f}%", f"{std_p*100:.2f}%", f"{sharpe_p:.2f}", f"{mdd_portfolio:.1f}%"],
        "S&P 500 올인": [f"{(mean_returns['SPLG']*252)*100:.2f}%", f"{(np.sqrt(cov_matrix.loc['SPLG','SPLG']*252))*100:.2f}%", f"{(mean_returns['SPLG']*252 - RISK_FREE_RATE)/(np.sqrt(cov_matrix.loc['SPLG','SPLG']*252)):.2f}", f"{mdd_splg:.1f}%"]
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

st.divider()

# 2. 내 계좌 입력
st.subheader("💼 현재 내 계좌 상태 입력")
shares_input = {}
with st.container(border=True):
    cols = st.columns(3)
    for i, ticker in enumerate(TARGET_TICKERS):
        with cols[i]:
            shares_input[ticker] = st.number_input(f"{ticker} 보유 주수", min_value=0, step=1, key=f"s_{ticker}")
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
            target_shares = target_value / curr_price if curr_price > 0 else 0
            share_diff = target_shares - shares_input[t]
            
            is_out_of_band = abs(weight_diff) >= 0.05
            if is_out_of_band:
                needs_rebalancing = True
                
            rounded_diff = round(share_diff)
            
            if rounded_diff >= 1:
                action = f"🟢 {rounded_diff}주 매수"
            elif rounded_diff <= -1:
                action = f"🔴 {abs(rounded_diff)}주 매도"
            else:
                if is_out_of_band and curr_price > 0:
                    action = "⚪ 금액 부족 (1주 미만)"
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
