import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from datetime import datetime, timedelta
import warnings

# 設定頁面資訊
st.set_page_config(
    page_title="CFO 5.1 終極大亂鬥",
    page_icon="⚔️",
    layout="wide"
)

warnings.filterwarnings('ignore')

# ============================================
# 1. 側邊欄參數設定 (UI 介面)
# ============================================
st.sidebar.header("⚙️ 參數設定")

# 投資標的
TARGET_TICKER = st.sidebar.text_input("🎯 投資標的 (Ticker)", value="QLD")
st.sidebar.caption("支援如: BTC-USD, NVDA, TQQQ, QLD")

# 日期設定
default_start = datetime(2020, 1, 1)
default_end = datetime(2024, 5, 20)
START_DATE = st.sidebar.date_input("📅 回測開始日期", default_start)
END_DATE = st.sidebar.date_input("📅 回測結束日期", default_end)

# 資金設定
INITIAL_CAPITAL = st.sidebar.number_input("💰 初始本金", value=100000, step=10000)
MONTHLY_CONTRIBUTION = st.sidebar.number_input("💵 每月投入", value=1000, step=100)

# API Key
FRED_API_KEY = st.sidebar.text_input("🔑 FRED API Key", value="9382c202c6133484efb2c1cb571495af", type="password")

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 開始回測", type="primary")

# ============================================
# 2. 核心引擎
# ============================================

class CFO_Battle_Engine:
    def __init__(self, ticker, api_key, initial_capital, monthly_contribution):
        self.ticker = ticker.strip()
        self.api_key = api_key
        self.initial_capital = initial_capital
        self.monthly_contribution = monthly_contribution
        self.data = None
        self.dataset = None
        self.strategies = [
            'DCA', 
            'Pure_MA200', 
            'CFO_9.0_CashMaster', 
            'CFO_5.1_MacroKelly'
        ]
        self.cash = {s: initial_capital for s in self.strategies}
        self.holdings = {s: 0.0 for s in self.strategies}
        self.interest_earned = {s: 0.0 for s in self.strategies}
        self.total_invested = initial_capital

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_kelly_simple(self, price_series, current_date, rf_rate):
        window = 60
        # 取得直到今天的過去數據
        past_data = price_series.loc[:current_date].tail(window+1)
        if len(past_data) < window: return 0.0
        
        returns = past_data.pct_change().dropna()
        if len(returns) < 30: return 0.0

        mu = returns.mean() * 365
        var = returns.var() * 365
        hurdle = rf_rate

        if var == 0: return 0.0
        f = (mu - hurdle) / var
        # Half-Kelly + Cap at 1.0
        return max(0.0, min(f * 0.5, 1.0))

    def run_backtest(self, dataset, start_date, end_date):
        # 篩選回測期間
        mask_date = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        df = dataset.loc[(dataset.index >= mask_date) & (dataset.index <= end_date_dt)].copy()
        
        if df.empty:
            return pd.DataFrame()

        # 計算 RSI
        df['RSI'] = self.calculate_rsi(df['PRICE'])

        monthly_dates = df.resample('MS').first().index
        history = []
        
        for d in df.index:
            price = df.loc[d, 'PRICE']
            
            # 1. 現金生息
            rf = df.loc[d, 'RISK_FREE_RATE']
            daily_yield = (rf + 0.02) / 365 
            
            for s in self.strategies:
                if self.cash[s] > 0:
                    interest = self.cash[s] * max(0, daily_yield)
                    self.cash[s] += interest
                    self.interest_earned[s] += interest

            # 2. 每月入金
            if d in monthly_dates:
                self.total_invested += self.monthly_contribution
                for s in self.strategies:
                    self.cash[s] += self.monthly_contribution

            # 3. 取得指標數據
            ma200 = df.loc[d, 'MA200']
            rsi = df.loc[d, 'RSI']
            vix = df.loc[d, 'VIX']
            liq = df.loc[d, 'FED_LIQUIDITY']
            liq_ma = df.loc[d, 'LIQ_MA50']
            gsr = df.loc[d, 'GOLD_SILVER_RATIO']
            gsr_ma = df.loc[d, 'GSR_MA200']
            gcr = df.loc[d, 'GOLD_COPPER_RATIO']

            if np.isnan(ma200) or np.isnan(rsi): continue

            # --- 策略權重計算 ---
            target_weights = {s: 0.0 for s in self.strategies}
            is_bull = price > ma200
            
            # A. DCA
            target_weights['DCA'] = 1.0
            
            # B. Pure MA200
            target_weights['Pure_MA200'] = 1.0 if is_bull else 0.0
            
            # 基礎 Kelly 值
            base_kelly = self.calculate_kelly_simple(dataset['PRICE'], d, rf)

            # C. CFO 9.0 (CashMaster)
            k9 = base_kelly
            if is_bull and k9 < 0.3: k9 = 0.3
            if not is_bull: k9 = 0.0
            target_weights['CFO_9.0_CashMaster'] = k9

            # D. CFO 5.1 (Macro-Kelly)
            macro_score = 0
            if liq > liq_ma: macro_score += 1
            if gsr < gsr_ma: macro_score += 1
            if gcr < 550: macro_score += 1
            if rsi < 45: macro_score += 1
            if rsi > 75: macro_score -= 1

            multiplier = max(0.2, min((macro_score + 1.5) / 3.0, 1.5))
            k51 = base_kelly * multiplier
            
            if vix > 32 or rsi > 85: k51 *= 0.2
            
            target_weights['CFO_5.1_MacroKelly'] = min(1.0, k51)

            # 4. 執行再平衡
            for s in self.strategies:
                total_val = self.cash[s] + self.holdings[s] * price
                target_pos_val = total_val * target_weights[s]
                curr_pos_val = self.holdings[s] * price
                diff = target_pos_val - curr_pos_val

                if abs(diff) > total_val * 0.01:
                    if diff > 0: # Buy
                        cost = min(self.cash[s], diff)
                        self.holdings[s] += cost / price
                        self.cash[s] -= cost
                    else: # Sell
                        val_to_sell = abs(diff)
                        self.holdings[s] -= val_to_sell / price
                        self.cash[s] += val_to_sell

            row = {'Date': d}
            for s in self.strategies:
                row[s] = self.cash[s] + self.holdings[s] * price
            history.append(row)

        return pd.DataFrame(history).set_index('Date')

# 使用 Streamlit Cache 機制，避免每次都要重新下載數據
@st.cache_data(ttl=3600)
def get_market_data(ticker, start_date, end_date, api_key):
    download_start = (pd.to_datetime(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # 1. 下載 Yahoo 數據
    yf_tickers = [ticker, 'GC=F', 'SI=F', 'HG=F']
    try:
        df = yf.download(yf_tickers, start=download_start, end=end_str, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            try: 
                df_close = df.xs('Close', axis=1, level=0)
            except:
                df_close = df.iloc[:, :len(yf_tickers)]
                df_close.columns = yf_tickers
        else:
            df_close = df[['Close']] if 'Close' in df.columns else df

        if len(yf_tickers) == 1:
            data = pd.DataFrame(df_close); data.columns = [ticker]
        else:
            data = df_close

        mapper = {'GC=F': 'GOLD', 'SI=F': 'SILVER', 'HG=F': 'COPPER', ticker: 'PRICE'}
        data = data.rename(columns=mapper)
        data = data.ffill().bfill()
        data.index = data.index.tz_localize(None)
    except Exception as e:
        return None, f"Yahoo 下載失敗: {str(e)}"

    # 2. 下載 FRED 數據
    try:
        fred_syms = {
            'VIXCLS': 'VIX',
            'DGS10': 'TNX',
            'DTB3': 'RISK_FREE_RATE',
            'WALCL': 'FED_LIQUIDITY'
        }
        fred_df = web.DataReader(list(fred_syms.keys()), 'fred', download_start, end_str, api_key=api_key)
        fred_df = fred_df.rename(columns=fred_syms)
        fred_df['RISK_FREE_RATE'] = fred_df['RISK_FREE_RATE'] / 100
        fred_df = fred_df.ffill().bfill()
        fred_df.index = fred_df.index.tz_localize(None)
    except Exception as e:
        # Fallback 假資料
        fred_df = pd.DataFrame(index=data.index)
        for col in ['VIX', 'TNX', 'RISK_FREE_RATE', 'FED_LIQUIDITY']:
            fred_df[col] = 0
        fred_df['RISK_FREE_RATE'] = 0.04
        st.warning(f"⚠️ FRED 數據下載失敗 ({str(e)})，將使用模擬數據運行。")

    # 3. 合併與計算
    full_df = data.join(fred_df, how='left').ffill().bfill()
    full_df['GOLD_SILVER_RATIO'] = full_df['GOLD'] / full_df['SILVER']
    full_df['GSR_MA200'] = full_df['GOLD_SILVER_RATIO'].rolling(200).mean()
    full_df['GOLD_COPPER_RATIO'] = full_df['GOLD'] / full_df['COPPER']
    full_df['LIQ_MA50'] = full_df['FED_LIQUIDITY'].rolling(50).mean()
    full_df['MA200'] = full_df['PRICE'].rolling(200).mean()
    
    return full_df, None

# ============================================
# 3. 主程式邏輯
# ============================================

st.title("⚔️ 終極大亂鬥: CFO 5.1 (混合數據源穩健版)")
st.markdown("""
本系統結合 **Yahoo Finance** 與 **FRED 總經數據**，進行多策略回測。
核心策略包含 `DCA`, `Pure MA200`, `CFO 9.0 CashMaster`, 以及 `CFO 5.1 MacroKelly`。
""")

if run_btn:
    with st.spinner('📥 正在抓取數據並進行模擬戰鬥...'):
        # 1. 獲取數據
        dataset, error_msg = get_market_data(TARGET_TICKER, START_DATE, END_DATE, FRED_API_KEY)
        
        if error_msg:
            st.error(error_msg)
        elif dataset is None or dataset.empty:
            st.error("❌ 無法獲取數據，請檢查標的代碼或日期。")
        else:
            st.success(f"✅ 數據下載成功 (包含 {len(dataset)} 筆交易日資料)")
            
            # 2. 初始化引擎與回測
            eng = CFO_Battle_Engine(TARGET_TICKER, FRED_API_KEY, INITIAL_CAPITAL, MONTHLY_CONTRIBUTION)
            eng.dataset = dataset # 注入數據
            res = eng.run_backtest(dataset, START_DATE, END_DATE)

            if not res.empty:
                # 3. 繪圖
                st.subheader("📈 策略淨值走勢圖 (Log Scale)")
                fig, ax = plt.subplots(figsize=(12, 6))
                plt.style.use('dark_background')
                colors = ['gray', 'cyan', 'yellow', '#FF00FF']
                
                for i, s in enumerate(eng.strategies):
                    lw = 2.5 if 'CFO' in s else 1
                    ax.plot(res.index, res[s], label=s, color=colors[i], linewidth=lw)
                
                ax.set_title(f'Strategy Battle: {TARGET_TICKER}', fontsize=14, color='white')
                ax.legend()
                ax.set_yscale('log')
                ax.grid(True, alpha=0.2)
                st.pyplot(fig)

                # 4. 統計報表
                st.subheader("🏆 最終戰績結算")
                
                final_day = res.index[-1]
                last_price = dataset.loc[final_day, 'PRICE']
                days = (res.index[-1] - res.index[0]).days
                years = max(days / 365.0, 0.1)

                stats_data = []
                final_vals = res.iloc[-1].sort_values(ascending=False)

                for strat, val in final_vals.items():
                    if strat == 'Date': continue
                    
                    # CAGR 計算
                    ret = val / eng.total_invested
                    cagr = (ret ** (1/years)) - 1
                    
                    # 持倉佔比
                    pos_val = eng.holdings[strat] * last_price
                    cash_val = eng.cash[strat]
                    total = pos_val + cash_val
                    ratio = (pos_val / total) * 100 if total > 0 else 0
                    
                    intr = eng.interest_earned[strat]
                    
                    stats_data.append({
                        "Strategy": strat,
                        "Final Balance": f"${val:,.0f}",
                        "CAGR": f"{cagr*100:.1f}%",
                        "Crypto %": f"{ratio:.0f}%",
                        "Cash %": f"{100-ratio:.0f}%",
                        "Interest Earned": f"${intr:,.0f}"
                    })

                st.write(f"💰 **總投入本金**: ${eng.total_invested:,.0f}")
                
                # 顯示漂亮的 DataFrame 表格
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats, use_container_width=True)

            else:
                st.warning("❌ 回測結果為空，請檢查日期範圍是否包含交易日。")