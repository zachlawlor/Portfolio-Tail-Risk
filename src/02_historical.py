# Phase 02: Historical VaR & Expected Shortfall

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok = True) # Ensuring these folders exist 
os.makedirs("data", exist_ok = True)

port1_log_returns = pd.read_csv("data/01_port1_log_returns.csv", index_col = 0, parse_dates = True)["Equal Weight"] # Loading in Equal Weight log returns
all_portfolio_log_returns = pd.read_csv("data/01_all_portfolio_log_returns.csv", index_col = 0, parse_dates = True) # Loading in all portfolio log returns

portfolio_value = 100_000  # Using €100k as initial investment for simplicity
risk_free_rate = 0.03  # Matches Phase 01

var_95 = np.percentile(port1_log_returns, 5) # Var & ES for Equal weight portfolio
var_99 = np.percentile(port1_log_returns, 1)

es_95 = port1_log_returns[port1_log_returns <= var_95].mean()
es_99 = port1_log_returns[port1_log_returns <= var_99].mean()

comparison_results = [] 

for name in all_portfolio_log_returns.columns: # Repeating above but for all portfolios
    rets = all_portfolio_log_returns[name]
    
    all_var_95 = np.percentile(rets, 5)
    all_var_99 = np.percentile(rets, 1)
    all_es_95 = rets[rets <= all_var_95].mean()
    all_es_99 = rets[rets <= all_var_99].mean()
    
    comparison_results.append({
        "Portfolio": name,
        "95% VaR (Loss)": abs(all_var_95),   # positive loss magnitude
        "95% ES (Loss)": abs(all_es_95),
        "99% VaR (Loss)": abs(all_var_99),
        "99% ES (Loss)": abs(all_es_99),
        "VaR (€) 99%": abs(all_var_99 * portfolio_value),
        "ES (€) 99%": abs(all_es_99 * portfolio_value)
    })

df_compare = pd.DataFrame(comparison_results) # Data Framing the results

print("=" * 60) # Print Section
print("Phase 02: Historical Var & Expected Shortfall")
print("=" * 60)
print(f"\nPortfolio: Equal Weight")
print(f"Notional Value: €{portfolio_value:,}")
print(f"Period: 2016-01-01 to 2025-12-31\n")
print("-" * 60)

print(f"\n95% VaR (Loss):               {abs(var_95):>8.2%}")
print(f"99% VaR (Loss):               {abs(var_99):>8.2%}")
print(f"95% ES (Loss):                {abs(es_95):>8.2%}")
print(f"99% ES (Loss):                {abs(es_99):>8.2%}\n")

print("-" * 60 + "\n")
print(df_compare.round(4).to_string(index = False),"\n")
print("-" * 60)

plt.style.use("seaborn-v0_8")  # Plot style

plt.figure(figsize = (10, 6)) # Plotting the ES & Var on a histogram 
plt.hist(port1_log_returns * 100, bins = 60, color = "blue", alpha = 0.7, edgecolor = "black", linewidth = 0.3)
# Lines placed at actual (negative) return thresholds but labels show positive loss
plt.axvline(var_95 * 100, color = "red", linestyle = "--", linewidth = 2, label = f"95% VaR Loss: {abs(var_95):.2%}")
plt.axvline(var_99 * 100, color = "darkred", linestyle = "--", linewidth = 2, label = f"99% VaR Loss: {abs(var_99):.2%}")
plt.axvline(es_95 * 100, color = "orange", linestyle = ":", linewidth = 2, label = f"95% ES Loss: {abs(es_95):.2%}")
plt.axvline(es_99 * 100, color = "purple", linestyle = ":", linewidth = 2, label = f"99% ES Loss: {abs(es_99):.2%}")
plt.xlabel("Daily Log Return (%)", fontweight = "bold")
plt.ylabel("Frequency", fontweight = "bold")
plt.title("Equal Weight Portfolio: Daily Log Returns with VaR / ES", fontsize = 14, fontweight = "bold")
plt.legend()
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/02_historical_var_distribution.png", dpi = 150, bbox_inches = "tight")
plt.show()

results_df = pd.DataFrame({
    "Portfolio": ["Equal Weight"],
    "95% VaR": [var_95],
    "99% VaR": [var_99],
    "95% ES": [es_95],
    "99% ES": [es_99]
})

results_df.to_csv("data/02_historical_metrics.csv", index = False) # Saving both the results of Equal Weight historical Var & ES + all portfolio 
df_compare.to_csv("data/02_historical_comparison.csv", index = False)  # Comparison saved with positive losses