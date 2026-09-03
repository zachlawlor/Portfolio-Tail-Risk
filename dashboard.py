#--------------------------------------------------------
# Interactive Portfolio Tail-Risk Analytics Dashboard Using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import norm, t


plt.style.use("seaborn-v0_8")

#--------------------------------------------------------
# Page configuration

st.set_page_config(
    page_title = "Portfolio Tail-Risk Analytics",
    page_icon = "❖",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# ----------------------------------------------------------
# Minimal CSS to change selected tab colour to dark grey

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f0f0;
        border-radius: 0.3rem;
        padding: 0.4rem 1rem;
        font-weight: 600;
        color: #333;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4a4a4a !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html = True)

st.title("❖ Portfolio Tail-Risk Analytics")
st.caption("Performance · Risk Models · Backtesting · Stress Testing · Correlation")

# ----------------------------------------
# Load Data

@st.cache_data
def load_data():
    prices = pd.read_csv("data/01_asset_prices.csv", index_col = 0, parse_dates = True)
    log_returns = pd.read_csv("data/01_log_returns.csv", index_col = 0, parse_dates = True)
    port1_returns = pd.read_csv("data/01_port1_log_returns.csv", index_col = 0, parse_dates = True)["Equal Weight"]
    all_port_returns = pd.read_csv("data/01_all_portfolio_log_returns.csv", index_col = 0, parse_dates = True)
    
    historical_metrics = pd.read_csv("data/02_historical_metrics.csv")
    normal_metrics = pd.read_csv("data/03_normal_metrics.csv")
    student_t_metrics = pd.read_csv("data/04_student_t_metrics.csv")
    stress_results = pd.read_csv("data/07_stress_test_results.csv")
    stress_pivot = pd.read_csv("data/07_stress_loss_pivot.csv", index_col = 0)
    
    return {
        "prices": prices,
        "log_returns": log_returns,
        "port1_returns": port1_returns,
        "all_port_returns": all_port_returns,
        "historical_metrics": historical_metrics,
        "normal_metrics": normal_metrics,
        "student_t_metrics": student_t_metrics,
        "stress_results": stress_results,
        "stress_pivot": stress_pivot
    }

try:
    data = load_data()
except FileNotFoundError as e:
    st.error(f"Data files not found. Please run Phases 01-07 first.\n\nError: {e}")
    st.stop()

# ------------------------------
# Portfolio Types

portfolio_types = {
    "Equal Weight": {"VOO": 0.20, "EFA": 0.20, "TLT": 0.20, "AGG": 0.20, "GLD": 0.20},
    "Classic 60/40": {"VOO": 0.30, "EFA": 0.30, "TLT": 0.20, "AGG": 0.20, "GLD": 0.00},
    "Defensive": {"VOO": 0.05, "EFA": 0.05, "TLT": 0.30, "AGG": 0.30, "GLD": 0.30},
    "Aggressive Growth": {"VOO": 0.40, "EFA": 0.40, "TLT": 0.10, "AGG": 0.10, "GLD": 0.00},
    "Conservative Income": {"VOO": 0.10, "EFA": 0.10, "TLT": 0.35, "AGG": 0.35, "GLD": 0.10}
}

# ------------------------------
# Sidebar

with st.sidebar:
    st.header("⚙️ Configuration")
    st.divider()
    
    selected_portfolio = st.selectbox(
        "Select Portfolio Strategy",
        list(portfolio_types.keys()),
        index = 0
    )
    weights = portfolio_types[selected_portfolio]
    
    st.divider()
    st.subheader("⚖ Asset Weights")
    for asset, weight in weights.items():
        st.write(f"{asset}: **{weight:.0%}**")
    
    st.divider()
    
    value_options = {     # Dropdown for portfolio value
        "€10,000": 10_000,
        "€20,000": 20_000,
        "€50,000": 50_000,
        "€100,000": 100_000,
        "€1,000,000": 1_000_000
    }
    selected_value_label = st.selectbox(
        "💰 Initial Portfolio Value",
        list(value_options.keys()),
        index = 3   # default €100,000
    )
    portfolio_value = value_options[selected_value_label]

# ------------------------------
# Helper: Calculate Metrics

def calculate_metrics(returns_series):
    cumulative = np.exp(returns_series.cumsum())
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak   # negative values
    
    annual_drift = returns_series.mean() * 252
    annual_vol = returns_series.std() * np.sqrt(252)
    cagr = cumulative.iloc[-1] ** (252 / len(returns_series)) - 1 if len(returns_series) > 0 else 0
    sharpe = (annual_drift - 0.03) / annual_vol if annual_vol != 0 else 0
    max_dd = drawdown.min()               
    total_return = cumulative.iloc[-1] - 1
    kurtosis = returns_series.kurtosis() + 3
    
    var_95 = np.percentile(returns_series, 5)   
    var_99 = np.percentile(returns_series, 1)   
    es_95 = returns_series[returns_series <= var_95].mean()  
    es_99 = returns_series[returns_series <= var_99].mean()  
    
    return {
        "cumulative": cumulative,
        "drawdown": drawdown,
        "annual_drift": annual_drift,
        "annual_vol": annual_vol,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_return,
        "kurtosis": kurtosis,
        "var_95": var_95,
        "var_99": var_99,
        "es_95": es_95,
        "es_99": es_99
    }

# ---------------------------------------
# Compute All Portfolio Metrics

all_returns = data["all_port_returns"]
all_portfolios_metrics = {}
for name in all_returns.columns:
    all_portfolios_metrics[name] = calculate_metrics(all_returns[name])

selected_metrics = calculate_metrics(data["all_port_returns"][selected_portfolio])

# ------------------------------
# Key Metrics Row (two rows of four)

st.divider()
st.subheader("▥ Key Performance Indicators")

row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
row1_col1.metric("📈 Annual Drift", f"{selected_metrics['annual_drift']:.2%}")
row1_col2.metric("💰 CAGR", f"{selected_metrics['cagr']:.2%}")
row1_col3.metric("📉 Volatility", f"{selected_metrics['annual_vol']:.2%}")
row1_col4.metric("⭐ Sharpe", f"{selected_metrics['sharpe']:.2f}")

row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
row2_col1.metric("🔻 Max Drawdown (Loss)", f"{abs(selected_metrics['max_dd']):.2%}")
row2_col2.metric("🎯 Kurtosis", f"{selected_metrics['kurtosis']:.2f}")
row2_col3.metric("💵 Total Return", f"{selected_metrics['total_return']:.2%}")
row2_col4.metric("🏦 Final Value", f"€{selected_metrics['cumulative'].iloc[-1] * portfolio_value:,.0f}")

st.divider()

# ------------------------------
# Tabs

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " 📈 Performance  ",
    " 🛡️ Risk Models  ", 
    " 🧪 Backtesting  ",
    " ⚡ Stress Tests ",
    " 🔗 Correlation  "
])

# ------------------------------------------------
# TAB 1: PERFORMANCE

with tab1:
    st.subheader("Selected Portfolio Growth")
    fig, ax = plt.subplots(figsize = (12, 5))
    ax.plot(selected_metrics["cumulative"].index, 
            selected_metrics["cumulative"] * portfolio_value, 
            color = "blue", linewidth = 2.5, label = selected_portfolio)
    ax.axhline(y = portfolio_value, color = 'grey', linestyle = '--', alpha = 0.6)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (€)')
    ax.legend(loc = 'upper left')
    ax.grid(True, alpha = 0.3)
    st.pyplot(fig)

    st.subheader("Overall Portfolio Growth (All Strategies)")
    fig, ax = plt.subplots(figsize = (12, 5))
    colors = ["blue", "green", "orange", "red", "purple"]
    for i, name in enumerate(all_returns.columns):
        cum = all_portfolios_metrics[name]["cumulative"]
        ax.plot(cum.index, cum * portfolio_value, color = colors[i], linewidth = 1.8, label = name)
    ax.axhline(y = portfolio_value, color = 'grey', linestyle = '--', alpha = 0.6)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (€)')
    ax.legend(loc = 'upper left')
    ax.grid(True, alpha = 0.3)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Drawdown from Peak")
        fig, ax = plt.subplots(figsize = (9, 4))
        ax.fill_between(selected_metrics["drawdown"].index, 0, 
                        -selected_metrics["drawdown"] * 100,
                        color = 'red', alpha = 0.4)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha = 0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("Log Return Distribution")
        fig, ax = plt.subplots(figsize = (9, 4))
        ax.hist(data["all_port_returns"][selected_portfolio] * 100, bins = 60, 
                color = 'blue', alpha = 0.7, edgecolor = 'white', density = True)
        ax.axvline(selected_metrics["var_95"] * 100, color = 'red', linestyle = '--', 
                   linewidth = 2, label = f'95% VaR Loss: {abs(selected_metrics["var_95"]):.2%}')
        ax.axvline(selected_metrics["var_99"] * 100, color = 'orange', linestyle = '--', 
                   linewidth = 2, label = f'99% VaR Loss: {abs(selected_metrics["var_99"]):.2%}')
        ax.axvline(selected_metrics["annual_drift"]/252 * 100, color = 'green', 
                   linestyle = '-', linewidth = 1.5, label = 'Mean')
        ax.set_xlabel('Daily Log Return (%)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha = 0.3)
        st.pyplot(fig)

    st.divider()
    st.subheader("📋 Portfolio Metrics Comparison")
    compare_df = pd.DataFrame({
        "Portfolio": list(all_portfolios_metrics.keys()),
        "Annual Drift": [all_portfolios_metrics[n]['annual_drift'] for n in all_portfolios_metrics],
        "Volatility": [all_portfolios_metrics[n]['annual_vol'] for n in all_portfolios_metrics],
        "Sharpe": [all_portfolios_metrics[n]['sharpe'] for n in all_portfolios_metrics],
        "Max DD (Loss)": [abs(all_portfolios_metrics[n]['max_dd']) for n in all_portfolios_metrics],
        "Total Return": [all_portfolios_metrics[n]['total_return'] for n in all_portfolios_metrics],
        "Kurtosis": [all_portfolios_metrics[n]['kurtosis'] for n in all_portfolios_metrics],
        "99% VaR (Loss)": [abs(all_portfolios_metrics[n]['var_99']) for n in all_portfolios_metrics],
        "99% ES (Loss)": [abs(all_portfolios_metrics[n]['es_99']) for n in all_portfolios_metrics]
    })
    st.dataframe(compare_df.style.format({
        "Annual Drift": "{:.2%}",
        "Volatility": "{:.2%}",
        "Sharpe": "{:.2f}",
        "Max DD (Loss)": "{:.2%}",
        "Total Return": "{:.2%}",
        "Kurtosis": "{:.2f}",
        "99% VaR (Loss)": "{:.2%}",
        "99% ES (Loss)": "{:.2%}"
    }), use_container_width = True, hide_index = True)

    st.info("💡 **Universal Insight:** Higher returns do not necessarily indicate a better portfolio. Comparing CAGR, volatility and maximum drawdown shows how effectively each strategy balances long-term growth against downside risk.")

# ---------------------------------------------------------------
# TAB 2: RISK MODELS

with tab2:
    st.subheader("Dynamic VaR & Expected Shortfall")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("95% VaR (Loss)", f"{abs(selected_metrics['var_95']):.2%}", 
                  delta = f"€{abs(selected_metrics['var_95'] * portfolio_value):,.0f}", delta_color = "off")
        st.metric("99% VaR (Loss)", f"{abs(selected_metrics['var_99']):.2%}", 
                  delta = f"€{abs(selected_metrics['var_99'] * portfolio_value):,.0f}", delta_color = "off")
    with col2:
        st.metric("95% ES (Loss)", f"{abs(selected_metrics['es_95']):.2%}", 
                  delta = f"€{abs(selected_metrics['es_95'] * portfolio_value):,.0f}", delta_color = "off")
        st.metric("99% ES (Loss)", f"{abs(selected_metrics['es_99']):.2%}", 
                  delta = f"€{abs(selected_metrics['es_99'] * portfolio_value):,.0f}", delta_color = "off")
    
    st.divider()
    st.subheader("Model Comparison")
    
    rets = data["all_port_returns"][selected_portfolio]
    mu = rets.mean()
    sigma = rets.std()
    
    t_fit_failed = False
    try:
        df_t, loc_t, scale_t = t.fit(rets)
        t_var_95 = t.ppf(0.05, df_t, loc_t, scale_t)
        t_var_99 = t.ppf(0.01, df_t, loc_t, scale_t)
        t_alpha_95 = t.ppf(0.05, df_t)
        t_alpha_99 = t.ppf(0.01, df_t)
        t_es_95 = loc_t - scale_t * (t.pdf(t_alpha_95, df_t) / 0.05) * ((df_t + t_alpha_95**2) / (df_t - 1))
        t_es_99 = loc_t - scale_t * (t.pdf(t_alpha_99, df_t) / 0.01) * ((df_t + t_alpha_99**2) / (df_t - 1))
    except Exception:
        t_fit_failed = True
        df_t = None
        t_var_95 = t_var_99 = t_es_95 = t_es_99 = None

    norm_var_95 = norm.ppf(0.05, mu, sigma)
    norm_var_99 = norm.ppf(0.01, mu, sigma)
    norm_es_95 = mu - sigma * norm.pdf(norm.ppf(0.05)) / 0.05
    norm_es_99 = mu - sigma * norm.pdf(norm.ppf(0.01)) / 0.01
    
    hist_var = [abs(selected_metrics['var_95'])*100, abs(selected_metrics['var_99'])*100]
    hist_es = [abs(selected_metrics['es_95'])*100, abs(selected_metrics['es_99'])*100]
    norm_var = [abs(norm_var_95)*100, abs(norm_var_99)*100]
    norm_es = [abs(norm_es_95)*100, abs(norm_es_99)*100]
    if not t_fit_failed:
        t_var = [abs(t_var_95)*100, abs(t_var_99)*100]
        t_es = [abs(t_es_95)*100, abs(t_es_99)*100]
    else:
        t_var = None
        t_es = None
    
    fig, axes = plt.subplots(1, 2, figsize = (14, 5))
    width = 0.25
    x = np.arange(2)
    labels = ['95%', '99%']
    
    axes[0].bar(x - width, hist_var, width, label = 'Historical', color = 'blue', alpha = 0.8)
    axes[0].bar(x, norm_var, width, label = 'Normal', color = 'green', alpha = 0.8)
    if t_var:
        axes[0].bar(x + width, t_var, width, label = 'Student-t', color = 'orange', alpha = 0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('Loss (%)')
    axes[0].set_title('Value at Risk (VaR)')
    axes[0].legend()
    axes[0].grid(True, alpha = 0.3)
    
    axes[1].bar(x - width, hist_es, width, label = 'Historical', color = 'blue', alpha = 0.8)
    axes[1].bar(x, norm_es, width, label = 'Normal', color = 'green', alpha = 0.8)
    if t_es:
        axes[1].bar(x + width, t_es, width, label = 'Student-t', color = 'orange', alpha = 0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Loss (%)')
    axes[1].set_title('Expected Shortfall (ES)')
    axes[1].legend()
    axes[1].grid(True, alpha = 0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.divider()
    st.subheader("99% VaR Across All Portfolios")
    fig, ax = plt.subplots(figsize = (10, 5))
    portfolios = list(all_portfolios_metrics.keys())
    var_99_all = [abs(all_portfolios_metrics[n]['var_99']) * 100 for n in portfolios]
    bars = ax.bar(portfolios, var_99_all, color = ['blue', 'green', 'orange', 'red', 'purple'], alpha = 0.7)
    ax.axhline(y = 0, color = 'grey', linewidth = 0.8)
    ax.set_ylabel('99% VaR Loss (%)')
    ax.set_title('99% Historical VaR Comparison Across All Strategies')
    ax.grid(True, alpha = 0.3)
    for bar, val in zip(bars, var_99_all):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}%', ha = 'center', fontsize = 9)
    st.pyplot(fig)
    
    st.info("💡 **Universal Insight:** : Different assumptions about the distribution of returns can produce materially different VaR and ES estimates. The Student-t model accounts for heavier tails, potentially providing a more conservative assessment of extreme losses.")
# -------------------------------------------------------
# TAB 3: BACKTESTING

with tab3:
    st.subheader("Expanding Window Backtest (2021-2025)")
    st.caption("99% VaR Breaches — Expected ~1% of days (~2.5 per year).")
    
    try:
        backtest_df = pd.read_csv("data/06_backtest_results.csv")
        all_backtest = pd.read_csv("data/06_backtest_all_portfolios.csv")
        
        fig, ax = plt.subplots(figsize = (10, 5))
        years = backtest_df["Year"]
        ax.bar(years - 0.2, backtest_df["Hist 99"], width = 0.2, label = 'Historical', color = 'blue', alpha = 0.7)
        ax.bar(years, backtest_df["Norm 99"], width = 0.2, label = 'Normal', color = 'red', alpha = 0.7)
        ax.bar(years + 0.2, backtest_df["T 99"], width = 0.2, label = 'Student-t', color = 'green', alpha = 0.7)
        expected_values = backtest_df["Expected 99%"]
        ax.plot(years, expected_values, color = 'grey', linestyle = '--', linewidth = 1.5, label = 'Expected')
        ax.set_xlabel('Test Year')
        ax.set_ylabel('Number of Breaches')
        ax.set_title('99% VaR Breaches by Model')
        ax.legend()
        ax.grid(True, alpha = 0.3)
        st.pyplot(fig)
        
        df_2022 = all_backtest[all_backtest["Year"] == 2022]
        fig, ax = plt.subplots(figsize = (10, 5))
        x_pos = np.arange(len(df_2022["Portfolio"]))
        width = 0.25
        ax.bar(x_pos - width, df_2022["Historical 99"], width, label = 'Historical', color = 'blue', alpha = 0.7)
        ax.bar(x_pos, df_2022["Normal 99"], width, label = 'Normal', color = 'red', alpha = 0.7)
        ax.bar(x_pos + width, df_2022["Student-t 99"], width, label = 'Student-t', color = 'green', alpha = 0.7)
        ax.axhline(y = 2.5, color = 'grey', linestyle = '--', linewidth = 1.5, label = 'Expected (~2.5)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_2022["Portfolio"])
        ax.set_ylabel('Breaches (99% VaR)')
        ax.set_title('2022 Crisis: Breaches by Portfolio')
        ax.legend()
        ax.grid(True, alpha = 0.3)
        st.pyplot(fig)
        
        st.subheader("Backtest Data")
        st.dataframe(backtest_df.round(4), use_container_width = True)
        
    except FileNotFoundError:
        st.warning("Backtest data not found. Run Phase 06 first.")
    
    st.info("💡 **Universal Insight:** A risk model is only useful if it performs well out-of-sample. Comparing observed VaR breaches with the expected frequency helps identify models that underestimate or overestimate actual market risk.")
# ----------------------------------------------------------------------------
# TAB 4: STRESS TESTS

with tab4:
    st.subheader("Stress Test Scenario Losses")
    
    stress_results = data["stress_results"]
    stress_df = stress_results[stress_results["Portfolio"] == selected_portfolio].copy()
    worst = stress_df.loc[stress_df["Loss %"].idxmax()]
    
    fig, ax = plt.subplots(figsize = (9, 5))
    stress_sorted = stress_df.sort_values("Loss %", ascending = False)
    colors = ['red' if x > 0 else 'green' for x in stress_sorted["Loss %"]]
    bars = ax.barh(stress_sorted["Scenario"], stress_sorted["Loss %"] * 100, color = colors, alpha = 0.7)
    ax.set_xlabel('Loss (%)')
    ax.set_title('Loss by Scenario')
    ax.grid(True, alpha = 0.3)
    for bar, loss in zip(bars, stress_sorted["Loss %"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{loss:.1%}', va = 'center', fontsize = 9)
    st.pyplot(fig)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.metric("⚠️ Worst Scenario", worst["Scenario"], 
                  delta = f"{worst['Loss %']:.1%} (€{worst['Loss (€)']:,.0f})",
                  delta_color = "off")
    with col2:
        st.write("")
    
    st.subheader("Loss Percentage Across All Portfolios by Scenario")  
    fig_line, ax_line = plt.subplots(figsize = (10, 5))
    plot_data = stress_results.pivot(index = "Scenario", columns = "Portfolio", values = "Loss %") * 100
    colors = ["blue", "green", "orange", "red", "purple"]
    for i, portfolio in enumerate(plot_data.columns):
        ax_line.plot(plot_data.index, plot_data[portfolio],
                     marker = 'o', linewidth = 2.5, markersize = 8,
                     color = colors[i], label = portfolio)
    ax_line.axhline(y = 0, color = "black", linewidth = 0.8, linestyle = "-")
    ax_line.set_ylabel("Loss (%)")
    ax_line.set_xlabel("Scenario")
    ax_line.set_title("Stress Test: Loss Percentage Across All Portfolios")
    ax_line.legend(loc = "best")
    ax_line.grid(axis = "y", alpha = 0.3)
    plt.tight_layout()
    st.pyplot(fig_line)
    
    st.divider()
    st.subheader("Loss Summary (All Portfolios)")
    st.dataframe(stress_results.style.format({"Loss %": "{:.1%}", "Loss (€)": "€{:,.0f}"}), 
                 use_container_width = True, hide_index = True)
    
    st.divider()
    st.subheader("Heatmap (All Portfolios)")
    try:
        stress_pivot = pd.read_csv("data/07_stress_loss_pivot.csv", index_col = 0)
        fig, ax = plt.subplots(figsize = (10, 5))
        sns.heatmap(stress_pivot * 100, annot = True, fmt = ".1f", cmap = "Reds", 
                    ax = ax, cbar_kws = {'label': 'Loss (%)'}, vmin = 0, vmax = 40)
        ax.set_title("Stress Test Losses (%) by Portfolio and Scenario")
        st.pyplot(fig)
    except FileNotFoundError:
        st.warning("Stress test pivot data not found. Run Phase 07 first.")
    
    st.info("💡 **Universal Insight:** Stress testing exposes vulnerabilities that may not appear in daily VaR. Extreme scenarios, such as simultaneous equity and bond losses, show how portfolio diversification can break down during adverse market conditions.")
# --------------------------------------------------
# TAB 5: CORRELATION

with tab5:
    st.subheader("Correlation Breakdown (Pre-2022 vs 2022)")
    
    log_returns = data["log_returns"]
    full_corr = log_returns.corr()
    pre_2022 = log_returns[log_returns.index.year < 2022].corr()
    crisis_2022 = log_returns[log_returns.index.year == 2022].corr()
    
    fig, axes = plt.subplots(1, 3, figsize = (18, 5))
    sns.heatmap(full_corr, annot = True, fmt = ".2f", cmap = "coolwarm", center = 0,
                linewidths = 0.5, ax = axes[0], cbar_kws = {'label': 'Correlation'})
    axes[0].set_title("Full Period (2016-2025)", fontsize = 14, fontweight = "bold")
    sns.heatmap(pre_2022, annot = True, fmt = ".2f", cmap = "coolwarm", center = 0,
                linewidths = 0.5, ax = axes[1], cbar_kws = {'label': 'Correlation'})
    axes[1].set_title("Pre-2022 (Normal)", fontsize = 14, fontweight = "bold")
    sns.heatmap(crisis_2022, annot = True, fmt = ".2f", cmap = "coolwarm", center = 0,
                linewidths = 0.5, ax = axes[2], cbar_kws = {'label': 'Correlation'})
    axes[2].set_title("2022 (Crisis)", fontsize = 14, fontweight = "bold")
    plt.tight_layout()
    st.pyplot(fig)
    
    if "VOO" in log_returns.columns and "TLT" in log_returns.columns:
        voo_tlt_pre = pre_2022.loc["VOO", "TLT"]
        voo_tlt_crisis = crisis_2022.loc["VOO", "TLT"]
    
    st.info("💡 **Universal Insight:** Diversification depends on assets not moving together. Rising correlations during stressed markets can reduce diversification benefits and increase portfolio losses precisely when protection is most needed.")

st.divider()
st.caption("❖ Built with Streamlit | Data from Phases 01-07")