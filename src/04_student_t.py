# Phase 04: Student-t VaR & Expected Shortfall

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import t
import os

os.makedirs("outputs", exist_ok = True) # Ensures folders exist
os.makedirs("data", exist_ok = True)

port1_log_returns = pd.read_csv("data/01_port1_log_returns.csv", index_col = 0, parse_dates = True)["Equal Weight"] # Import data
all_portfolio_log_returns = pd.read_csv("data/01_all_portfolio_log_returns.csv", index_col = 0, parse_dates = True)

portfolio_value = 100_000
risk_free_rate = 0.03

df, loc, scale = t.fit(port1_log_returns) # Fitting parameters of student-t distribution to data 

var_95_t = t.ppf(0.05, df, loc, scale) # Var using the inverse CDF (negative)
var_95_t_dollars = abs(var_95_t * portfolio_value)   # positive loss in euros
var_99_t = t.ppf(0.01, df, loc, scale)
var_99_t_dollars = abs(var_99_t * portfolio_value)

alpha_95 = 0.05
t_alpha_95 = t.ppf(alpha_95, df)
es_95_t = loc - scale * (t.pdf(t_alpha_95, df) / alpha_95) * ((df + t_alpha_95**2) / (df - 1)) # Expected shortfall formula (negative)
es_95_t_dollars = abs(es_95_t * portfolio_value)

alpha_99 = 0.01
t_alpha_99 = t.ppf(alpha_99, df)
es_99_t = loc - scale * (t.pdf(t_alpha_99, df) / alpha_99) * ((df + t_alpha_99**2) / (df - 1))
es_99_t_dollars = abs(es_99_t * portfolio_value)


comparison_results = []
for name in all_portfolio_log_returns.columns: # Repeating for all porfolio types

    rets = all_portfolio_log_returns[name]
    df_i, loc_i, scale_i = t.fit(rets)

    t_var_95 = t.ppf(0.05, df_i, loc_i, scale_i)
    t_var_99 = t.ppf(0.01, df_i, loc_i, scale_i)
    
    t_alpha_95 = t.ppf(0.05, df_i)
    t_alpha_99 = t.ppf(0.01, df_i)
    t_es_95 = loc_i - scale_i * (t.pdf(t_alpha_95, df_i) / 0.05) * ((df_i + t_alpha_95**2) / (df_i - 1))
    t_es_99 = loc_i - scale_i * (t.pdf(t_alpha_99, df_i) / 0.01) * ((df_i + t_alpha_99**2) / (df_i - 1))
    
    comparison_results.append({
        "Portfolio": name,
        "DF": df_i,
        "95% VaR (Loss)": abs(t_var_95),   # positive loss magnitude
        "95% ES (Loss)": abs(t_es_95),
        "99% VaR (Loss)": abs(t_var_99),
        "99% ES (Loss)": abs(t_es_99),
        "VaR (€) 99%": abs(t_var_99 * portfolio_value),
        "ES (€) 99%": abs(t_es_99 * portfolio_value)
    })

df_compare = pd.DataFrame(comparison_results) # Data frame

historical_metrics = pd.read_csv("data/02_historical_metrics.csv") # For comparison
hist_var_95 = historical_metrics["95% VaR"].iloc[0]   
hist_var_99 = historical_metrics["99% VaR"].iloc[0]
hist_es_95 = historical_metrics["95% ES"].iloc[0]
hist_es_99 = historical_metrics["99% ES"].iloc[0]

normal_metrics = pd.read_csv("data/03_normal_metrics.csv")
norm_var_95 = normal_metrics["95% VaR"].iloc[0]
norm_var_99 = normal_metrics["99% VaR"].iloc[0]
norm_es_95 = normal_metrics["95% ES"].iloc[0]
norm_es_99 = normal_metrics["99% ES"].iloc[0]

print("=" * 60) # Print section
print("Phase 04: Student-t VaR & Expected Shortfall")
print("=" * 60)
print(f"\nPortfolio: Equal Weight")
print(f"Notional Value: €{portfolio_value:,}")
print(f"Period: 2016-01-01 to 2025-12-31\n")
print("-" * 60)

print(f"\nStudent-t Distribution Parameters\n")
print(f"   Degrees of Freedom:            {df:>8.2f}")
print(f"   Location (mu):                 {loc:>8.4%}")
print(f"   Scale (sigma):                 {scale:<8.4%}")

print(f"\n95% Student-t VaR (Loss):         {abs(var_95_t):>8.2%}")
print(f"99% Student-t VaR (Loss):         {abs(var_99_t):>8.2%}")
print(f"95% Student-t ES (Loss):          {abs(es_95_t):>8.2%}")
print(f"99% Student-t ES (Loss):          {abs(es_99_t):>8.2%}\n")
print("-" * 60)

print(f"\n{'Metric':<20} {'Historical':>12} {'Normal':>12} {'Student-t':>12}\n")
print(f"{'95% VaR (Loss)':<20} {abs(hist_var_95):>12.2%} {abs(norm_var_95):>12.2%} {abs(var_95_t):>12.2%}")
print(f"{'99% VaR (Loss)':<20} {abs(hist_var_99):>12.2%} {abs(norm_var_99):>12.2%} {abs(var_99_t):>12.2%}")
print(f"{'95% ES (Loss)':<20} {abs(hist_es_95):>12.2%} {abs(norm_es_95):>12.2%} {abs(es_95_t):>12.2%}")
print(f"{'99% ES (Loss)':<20} {abs(hist_es_99):>12.2%} {abs(norm_es_99):>12.2%} {abs(es_99_t):>12.2%}")
print("\n" + "-" * 60 + "\n")

print(df_compare.round(4).to_string(index = False))
print("\n" + "-" * 60)

x = np.linspace(port1_log_returns.min() * 100, port1_log_returns.max() * 100, 100) 
y_t = t.pdf(x / 100, df, loc, scale) * 100

plt.style.use("seaborn-v0_8")  # Plot style

plt.figure(figsize = (10, 6)) # Plot 1: Showing Student-t distribution Var vs Historical
plt.hist(port1_log_returns * 100, bins = 60, density = True, color = "blue", alpha = 0.7, edgecolor = "black", linewidth = 0.3)
plt.fill_between(x, 0, y_t, color = "green", alpha = 0.15, label = "Student-t Density")
plt.plot(x, y_t, color = "green", linewidth = 2, label = f"Student-t Fit (df={df:.1f})")
plt.axvline(hist_var_95 * 100, color = "darkblue", linestyle = "-", linewidth = 2, label = f"Historical 95% VaR Loss: {abs(hist_var_95):.2%}")
plt.axvline(hist_var_99 * 100, color = "blue", linestyle = "-", linewidth = 2, label = f"Historical 99% VaR Loss: {abs(hist_var_99):.2%}")
plt.axvline(var_95_t * 100, color = "orange", linestyle = "--", linewidth = 2, label = f"Student-t 95% VaR Loss: {abs(var_95_t):.2%}")
plt.axvline(var_99_t * 100, color = "red", linestyle = "--", linewidth = 2, label = f"Student-t 99% VaR Loss: {abs(var_99_t):.2%}")
plt.xlabel("Daily Log Return (%)", fontweight = "bold")
plt.ylabel("Density", fontweight = "bold")
plt.title("Equal Weight Portfolio: Historical vs Student-t Distribution VaR", fontsize = 14, fontweight = "bold")
plt.legend()
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/04_student_t_distribution_var.png", dpi = 150, bbox_inches = "tight")
plt.show()

plt.figure(figsize = (10, 6)) # Plot 2: Showing Student-t distribution ES vs Historical 
plt.hist(port1_log_returns * 100, bins = 60, density = True, color = "blue", alpha = 0.7, edgecolor = "black", linewidth = 0.3)
plt.fill_between(x, 0, y_t, color = "purple", alpha = 0.15, label = "Student-t Density")
plt.plot(x, y_t, color = "purple", linewidth = 2.5, label = f"Student-t Fit (df={df:.1f})")
plt.axvline(hist_es_95 * 100, color = "darkblue", linestyle = "-", linewidth = 2, label = f"Historical 95% ES Loss: {abs(hist_es_95):.2%}")
plt.axvline(hist_es_99 * 100, color = "blue", linestyle = "-", linewidth = 2, label = f"Historical 99% ES Loss: {abs(hist_es_99):.2%}")
plt.axvline(es_95_t * 100, color = "orange", linestyle = "--", linewidth = 2, label = f"Student-t 95% ES Loss: {abs(es_95_t):.2%}")
plt.axvline(es_99_t * 100, color = "red", linestyle = "--", linewidth = 2, label = f"Student-t 99% ES Loss: {abs(es_99_t):.2%}")
plt.xlabel("Daily Log Return (%)", fontweight = "bold")
plt.ylabel("Density", fontweight = "bold")
plt.title("Equal Weight Portfolio: Historical vs Student-t Expected Shortfall", fontsize = 14, fontweight = "bold")
plt.legend()
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/04_student_t_distribution_es.png", dpi = 150, bbox_inches = "tight")
plt.show()

plt.figure(figsize = (10, 6)) # Plot 3: Bar Chart comparing all 3 models so far
labels = ["95% VaR", "99% VaR", "95% ES", "99% ES"]
historical_values = [abs(hist_var_95) * 100, abs(hist_var_99) * 100, abs(hist_es_95) * 100, abs(hist_es_99) * 100]
normal_values = [abs(norm_var_95) * 100, abs(norm_var_99) * 100, abs(norm_es_95) * 100, abs(norm_es_99) * 100]
t_values = [abs(var_95_t) * 100, abs(var_99_t) * 100, abs(es_95_t) * 100, abs(es_99_t) * 100]
x_pos = np.arange(len(labels))
width = 0.25
plt.bar(x_pos - width, historical_values, width, label = "Historical", color = "blue", alpha = 0.7)
plt.bar(x_pos, normal_values, width, label = "Normal", color = "red", alpha = 0.7)
plt.bar(x_pos + width, t_values, width, label = "Student-t", color = "green", alpha = 0.7)
plt.axhline(y = 0, color = "black", linewidth = 0.8)
plt.xticks(x_pos, labels, fontweight = "bold")
plt.ylabel("Loss (%)", fontweight = "bold")
plt.title("Equal Weight: Historical vs Normal vs Student-t VaR / ES", fontsize = 14, fontweight = "bold")
plt.legend()
plt.grid(axis = "y", alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/04_three_model_comparison.png", dpi = 150, bbox_inches = "tight")
plt.show()

results_df = pd.DataFrame({
    "Portfolio": ["Equal Weight"],
    "DF": [df],
    "95% VaR": [var_95_t],
    "99% VaR": [var_99_t],
    "95% ES": [es_95_t],
    "99% ES": [es_99_t]
})

results_df.to_csv("data/04_student_t_metrics.csv", index = False) # Saving results (negative raw values)
df_compare.to_csv("data/04_student_t_comparison.csv", index = False)  # positive losses in comparison