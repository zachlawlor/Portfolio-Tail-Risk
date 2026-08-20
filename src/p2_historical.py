# Phase 2: Historical VaR & Expected Shortfall

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok = True) # Ensures output folder exists

portfolio_returns = pd.read_csv("data/portfolio_returns.csv", index_col = 0, parse_dates = True)
portfolio_returns = portfolio_returns["PORT"] # Loading in portfolio returns data

portfolio_value = 100_000 # Setting up initial investment

var_95 = np.percentile(portfolio_returns, 5) # 95% Value at Risk
var_95_dollars = var_95 * portfolio_value

var_99 = np.percentile(portfolio_returns, 1) # 99% Value at Risk
var_99_dollars = var_99 * portfolio_value

print(f"\n95% Historical VaR (One Day):") # Print 95% VaR
print(f"   • Loss of {var_95_dollars:,.0f}€ or worse on 5% of days")
print(f"   • That's a {var_95:.2%} daily loss")

print(f"\n99% Historical VaR (One Day):") # Print 99% VaR
print(f"   • Loss of {var_99_dollars:,.0f}€ or worse on 1% of days")
print(f"   • That's a {var_99:.2%} daily loss")

worst_days_95 = portfolio_returns[portfolio_returns <= var_95] # Filtering to only include worst 5% of returns
es_95 = worst_days_95.mean() # Average of the worst 5% 
es_95_dollars = es_95 * portfolio_value # 95% Expected Shortfall

worst_days_99 = portfolio_returns[portfolio_returns <= var_99] # Filtering to only include worst 1% of returns
es_99 = worst_days_99.mean() # Average of the worst 1%
es_99_dollars = es_99 * portfolio_value # 99% Expected Shortfall

print(f"\n95% Expected Shortfall (One Day):") # Print Expected Shortfall
print(f"   • On the worst 5% of days, the average loss is {es_95_dollars:,.0f}€")
print(f"   • That's a {es_95:.2%} average daily loss")

print(f"\n99% Expected Shortfall (One Day):") # Print Expected Shortfall
print(f"   • On the worst 1% of days, the average loss is {es_99_dollars:,.0f}€")
print(f"   • That's a {es_99:.2%} average daily loss")

print(f"\nFor a €{portfolio_value:,} portfolio: ") # Print out summary 
print(f"95% of days: You won't lose more than €{abs(var_95_dollars):,.0f}")
print(f"Worst 5% of days: Average loss is €{abs(es_95_dollars):,.0f}")
print(f"99% of days: You won't lose more than €{abs(var_99_dollars):,.0f}")
print(f"Worst 1% of days: Average loss is €{abs(es_99_dollars):,.0f}\n")

plt.figure(figsize = (10, 6)) # Plot 1: Distribution of daily returns with VaR/ES
plt.hist(portfolio_returns * 100, bins = 50, color = "blue", alpha = 0.7, edgecolor = "black")
plt.axvline(var_95 * 100, color = "red", linestyle = "--", linewidth = 2, label = f"95% VaR: {var_95:.2%}")
plt.axvline(var_99 * 100, color = "darkred", linestyle = "--", linewidth = 2, label = f"99% VaR: {var_99:.2%}")
plt.axvline(es_95 * 100, color = "orange", linestyle = ":", linewidth = 2, label = f"95% ES: {es_95:.2%}")
plt.axvline(es_99 * 100, color = "purple", linestyle = ":", linewidth = 2, label = f"99% ES: {es_99:.2%}")
plt.xlabel("Daily Return (%)", fontweight = "bold")
plt.ylabel("Frequency", fontweight = "bold")
plt.title("Distribution of Daily Portfolio Returns with VaR / ES", fontsize = 14, fontweight = "bold")
plt.legend()
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/historical_distribution.png", dpi = 150, bbox_inches = "tight")
plt.show()

results = {"Historical 95% VaR" : var_95, "Historical 99% VaR" : var_99, "Historical 95% Expected Shortfall" : es_95, "Historical 99% Expected Shortfall" : es_99} 

results_df = pd.DataFrame([results])
results_df.to_csv("data/historical_metrics.csv", index = False) # Saving results
