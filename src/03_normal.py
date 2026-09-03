# Phase 03: Normal VaR & Expected Shortfall

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

os.makedirs("outputs", exist_ok = True) # Ensures folders exist 
os.makedirs("data", exist_ok = True)

port1_log_returns = pd.read_csv("data/01_port1_log_returns.csv", index_col = 0, parse_dates = True)["Equal Weight"] # Importing data - same as before
all_portfolio_log_returns = pd.read_csv("data/01_all_portfolio_log_returns.csv", index_col = 0, parse_dates = True)

portfolio_value = 100_000
risk_free_rate = 0.03

mean_return = port1_log_returns.mean() # Paramters for normal distribution
std_return = port1_log_returns.std()

var_95_normal = norm.ppf(0.05, mean_return, std_return) # Now calculating Var & ES if log returns were normally distributed (negative values)
var_95_normal_dollars = abs(var_95_normal * portfolio_value)   # Positive loss in euros
var_99_normal = norm.ppf(0.01, mean_return, std_return)
var_99_normal_dollars = abs(var_99_normal * portfolio_value)
var_999_normal = norm.ppf(0.001, mean_return, std_return) 
var_999_normal_dollars = abs(var_999_normal * portfolio_value)

alpha_95 = 0.05
z_95 = norm.ppf(alpha_95)
es_95_normal = mean_return - std_return * norm.pdf(z_95) / alpha_95   # negative
es_95_normal_dollars = abs(es_95_normal * portfolio_value)

alpha_99 = 0.01
z_99 = norm.ppf(alpha_99)
es_99_normal = mean_return - std_return * norm.pdf(z_99) / alpha_99
es_99_normal_dollars = abs(es_99_normal * portfolio_value)

alpha_999 = 0.001
z_999 = norm.ppf(alpha_999)
es_999_normal = mean_return - std_return * norm.pdf(z_999) / alpha_999
es_999_normal_dollars = abs(es_999_normal * portfolio_value)

comparison_results = []
for name in all_portfolio_log_returns.columns: # Same again now but for all portfolio types
    rets = all_portfolio_log_returns[name]
    mu = rets.mean()
    sigma = rets.std()
    
    norm_var_95 = norm.ppf(0.05, mu, sigma)
    norm_var_99 = norm.ppf(0.01, mu, sigma)
    
    z_95 = norm.ppf(0.05)
    z_99 = norm.ppf(0.01)
    norm_es_95 = mu - sigma * norm.pdf(z_95) / 0.05
    norm_es_99 = mu - sigma * norm.pdf(z_99) / 0.01
    
    comparison_results.append({
        "Portfolio": name,
        "95% VaR (Loss)": abs(norm_var_95),   # positive loss
        "95% ES (Loss)": abs(norm_es_95),
        "99% VaR (Loss)": abs(norm_var_99),
        "99% ES (Loss)": abs(norm_es_99),
        "VaR (€) 99%": abs(norm_var_99 * portfolio_value),
        "ES (€) 99%": abs(norm_es_99 * portfolio_value)
    })

df_compare = pd.DataFrame(comparison_results)

historical_metrics = pd.read_csv("data/02_historical_metrics.csv") # Loading in historical metrics for comparison with normal dist
hist_var_95 = historical_metrics["95% VaR"].iloc[0]   # negative
hist_var_99 = historical_metrics["99% VaR"].iloc[0]
hist_es_95 = historical_metrics["95% ES"].iloc[0]
hist_es_99 = historical_metrics["99% ES"].iloc[0]
hist_var_999 = np.percentile(port1_log_returns, 0.1)   # negative
hist_es_999 = port1_log_returns[port1_log_returns <= hist_var_999].mean()  # negative

print("=" * 60) # Print Section 
print("Phase 03: Normal VaR & Expected Shortfall")
print("=" * 60)
print(f"\nPortfolio: Equal Weight")
print(f"Notional Value: €{portfolio_value:,}")
print(f"Period: 2016-01-01 to 2025-12-31\n")
print("-" * 60)

print(f"\n95% Normal VaR (Loss):             {abs(var_95_normal):>8.2%}")
print(f"99% Normal VaR (Loss):             {abs(var_99_normal):>8.2%}")
print(f"95% Normal ES (Loss):              {abs(es_95_normal):>8.2%}")
print(f"99% Normal ES (Loss):              {abs(es_99_normal):>8.2%}")
print("\n" + "-" * 60)

print(f"\n{'Metric':<20} {'Normal':>12} {'Historical':>12} {'Difference':>12}\n") # Prints a table with a comparision column 
print(f"{'95% VaR (Loss)':<20} {abs(var_95_normal):>12.2%} {abs(hist_var_95):>12.2%} {(abs(var_95_normal) - abs(hist_var_95)):>12.2%}")
print(f"{'99% VaR (Loss)':<20} {abs(var_99_normal):>12.2%} {abs(hist_var_99):>12.2%} {(abs(var_99_normal) - abs(hist_var_99)):>12.2%}")
print(f"{'95% ES (Loss)':<20} {abs(es_95_normal):>12.2%} {abs(hist_es_95):>12.2%} {(abs(es_95_normal) - abs(hist_es_95)):>12.2%}")
print(f"{'99% ES (Loss)':<20} {abs(es_99_normal):>12.2%} {abs(hist_es_99):>12.2%} {(abs(es_99_normal) - abs(hist_es_99)):>12.2%}")

print("\n" + "-" * 60 + "\n")
print(df_compare.round(4).to_string(index = False))
print("\n" + "-" * 60)

x = np.linspace(port1_log_returns.min() * 100, port1_log_returns.max() * 100, 100)
y_norm = norm.pdf(x / 100, mean_return, std_return) * 100

plt.style.use("seaborn-v0_8")  # Plot style

plt.figure(figsize = (10, 6)) # Plot 1: Comparing Var of Historical to normal 
plt.hist(port1_log_returns * 100, bins = 60, density = True, color = "blue", alpha = 0.7, edgecolor = "black", linewidth = 0.3)
plt.fill_between(x, 0, y_norm, color = "red", alpha = 0.15, label = "Normal Density")
plt.plot(x, y_norm, color = "red", linewidth = 2, label = "Normal Fit")
plt.axvline(hist_var_95 * 100, color = "darkblue", linestyle = "-", linewidth = 2, label = f"Historical 95% VaR Loss: {abs(hist_var_95):.2%}")
plt.axvline(hist_var_99 * 100, color = "blue", linestyle = "-", linewidth = 2, label = f"Historical 99% VaR Loss: {abs(hist_var_99):.2%}")
plt.axvline(var_95_normal * 100, color = "orange", linestyle = "--", linewidth = 2, label = f"Normal 95% VaR Loss: {abs(var_95_normal):.2%}")
plt.axvline(var_99_normal * 100, color = "red", linestyle = "--", linewidth = 2, label = f"Normal 99% VaR Loss: {abs(var_99_normal):.2%}")
plt.xlabel("Daily Log Return (%)", fontweight = "bold")
plt.ylabel("Density", fontweight = "bold")
plt.title("Equal Weight Portfolio: Historical vs Normal Distribution Value-at-Risk", fontsize = 14, fontweight = "bold")
plt.legend()
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/03_normal_distribution_var.png", dpi = 150, bbox_inches = "tight")
plt.show()

plt.figure(figsize = (10, 6)) # Plot 2: Comparing ES of Historical to normal 
plt.hist(port1_log_returns * 100, bins = 60, density = True, color = "blue", alpha = 0.7, edgecolor = "black", linewidth = 0.3)
plt.fill_between(x, 0, y_norm, color = "green", alpha = 0.15, label = "Normal Density")
plt.plot(x, y_norm, color = "green", linewidth = 2, label = "Normal Fit")
plt.axvline(hist_es_95 * 100, color = "darkred", linestyle = "-", linewidth = 2, label = f"Historical 95% ES Loss: {abs(hist_es_95):.2%}")
plt.axvline(hist_es_99 * 100, color = "red", linestyle = "-", linewidth = 2, label = f"Historical 99% ES Loss: {abs(hist_es_99):.2%}")
plt.axvline(es_95_normal * 100, color = "orange", linestyle = "--", linewidth = 2, label = f"Normal 95% ES Loss: {abs(es_95_normal):.2%}")
plt.axvline(es_99_normal * 100, color = "gold", linestyle = "--", linewidth = 2, label = f"Normal 99% ES Loss: {abs(es_99_normal):.2%}")
plt.xlabel("Daily Log Return (%)", fontweight = "bold")
plt.ylabel("Density", fontweight = "bold")
plt.title("Equal Weight Portfolio: Historical vs Normal Expected Shortfall", fontsize = 14, fontweight = "bold")
plt.legend()
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/03_normal_distribution_es.png", dpi = 150, bbox_inches = "tight")
plt.show()

plt.figure(figsize = (10, 6)) # Plot 3: Bar chart showing the difference between normal & historical VaR & ES
labels = ["95% VaR", "99% VaR","99.9% VaR", "95% ES", "99% ES", "99.9% ES"]
historical_values = [abs(hist_var_95) * 100, abs(hist_var_99) * 100, abs(hist_var_999) * 100, abs(hist_es_95) * 100, abs(hist_es_99) * 100, abs(hist_es_999) * 100]
normal_values = [abs(var_95_normal) * 100, abs(var_99_normal) * 100, abs(var_999_normal) * 100, abs(es_95_normal) * 100, abs(es_99_normal) * 100, abs(es_999_normal) * 100]
x_pos = np.arange(len(labels))
width = 0.35
plt.bar(x_pos - width/2, historical_values, width, label = "Historical", color = "blue", alpha = 0.7)
plt.bar(x_pos + width/2, normal_values, width, label = "Normal", color = "red", alpha = 0.7)
plt.axhline(y = 0, color = "black", linewidth = 0.8)
plt.xticks(x_pos, labels, fontweight = "bold")
plt.ylabel("Loss (%)", fontweight = "bold")
plt.title("Equal Weight: Historical vs Normal VaR / ES", fontsize = 14, fontweight = "bold")
plt.legend()
plt.grid(axis = "y", alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/03_normal_comparison.png", dpi = 150, bbox_inches = "tight")
plt.show()

results_df = pd.DataFrame({
    "Portfolio": ["Equal Weight"],
    "95% VaR": [var_95_normal],
    "99% VaR": [var_99_normal],
    "95% ES": [es_95_normal],
    "99% ES": [es_99_normal]
})
results_df.to_csv("data/03_normal_metrics.csv", index = False) # Saving results 
df_compare.to_csv("data/03_normal_comparison.csv", index = False)   