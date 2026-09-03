# Phase 01: Data Loader & Basic Statistics 

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 

assets = { 
    "US Equities": "VOO",
    "International Equities": "EFA",
    "Long Term Bonds": "TLT",   
    "Aggregate Bonds": "AGG",     
    "Gold": "GLD"
} # Assets chosen for project - mix of equity, bonds & commodities 

portfolios = {
    "Equal Weight":  {"VOO": 0.20, "EFA": 0.20, "TLT": 0.20, "AGG": 0.20, "GLD": 0.20}, # Will be using equal weight for main analysis - Port 1
    "Classic 60/40": {"VOO": 0.30, "EFA": 0.30, "TLT": 0.20, "AGG": 0.20, "GLD": 0.00}, # Port 2
    "Defensive": {"VOO": 0.05, "EFA": 0.05, "TLT": 0.30, "AGG": 0.30, "GLD": 0.30}, # Port 3
    "Aggressive Growth": {"VOO": 0.40, "EFA": 0.40, "TLT": 0.10, "AGG": 0.10, "GLD": 0.00}, # Port 4
    "Conservative Income": {"VOO": 0.10, "EFA": 0.10, "TLT": 0.35, "AGG": 0.35, "GLD": 0.10}# Port 5
} # Created 5 different portfolio types for analysis - note all sum up to equal 1

data = yf.download(list(assets.values()), start = "2016-01-01", end = "2025-12-31", auto_adjust = True) # Time period = 10 years

prices = data["Close"].dropna() # Calculating daily log returns as they are time additive 
log_returns = np.log(prices / prices.shift(1)).dropna()

portfolio_log_returns = {}
for name, weights in portfolios.items(): # Looping through all 5 portfolio types & multiplying the assets by the correct weights for each
    port_ret = 0
    for asset in weights:
        port_ret += (np.exp(log_returns[asset]) * weights[asset])  
    portfolio_log_returns[name] = np.log(port_ret)

port1_log_returns = portfolio_log_returns["Equal Weight"] # Extract the main one (Equal Weight) for analysis - Port 1

annual_drift = port1_log_returns.mean() * 252 # Annual drift 
annual_volatility = port1_log_returns.std() * np.sqrt(252) # Volatility
risk_free_rate = 0.03 # Chosen as risk-free rate for project
sharpe_ratio = float((annual_drift - risk_free_rate) / annual_volatility) # Sharpe Ratio
cumulative = np.exp(port1_log_returns.cumsum()) # Converts daily log returns to cumulative returns - time additive property
peak = cumulative.expanding().max() # Calculates peaks
drawdown = (cumulative - peak) / peak # Measures how far portfolio falls from previous peak
max_drawdown = drawdown.min() # Min as max is negative
total_return = cumulative.iloc[-1] - 1 # Total return
annual_growth = cumulative.iloc[-1] ** (252 / len(port1_log_returns)) - 1 # Compounded annual growth rate - discrete ( yearly )
skewness = port1_log_returns.skew() # Skewness
kurtosis = port1_log_returns.kurtosis() + 3 # Kurtosis - normal distribution has kurtosis 3 

comparison_data = {} 
for name, weights in portfolios.items(): # Repeats same as above but loops through all 5 portfolios 
    port_ret = 0
    for asset in weights:
        port_ret += np.exp(log_returns[asset]) * weights[asset]  
    port_ret = np.log(port_ret)
    
    comp_annual_drift = port_ret.mean() * 252
    comp_vol = port_ret.std() * np.sqrt(252)
    comp_sharpe = (comp_annual_drift - risk_free_rate) / comp_vol
    cum_growth = np.exp(port_ret.cumsum())
    peak = cum_growth.expanding().max()
    comp_max_dd = (cum_growth - peak) / peak
    comp_total_return = cum_growth.iloc[-1] - 1
    comp_cagr = cum_growth.iloc[-1] ** (252 / len(port_ret)) - 1
    comp_skewness = port_ret.skew()
    comp_kurtosis = port_ret.kurtosis() + 3
    
    comparison_data[name] = {
        "Annual Drift": comp_annual_drift,
        "Volatility": comp_vol,
        "Sharpe": comp_sharpe,
        "Max DD": abs(comp_max_dd.min()),
        "Total Return": comp_total_return,
        "CAGR": comp_cagr,
        "Skewness": comp_skewness,
        "Kurtosis": comp_kurtosis
    }

df_compare = pd.DataFrame(comparison_data).T # Creating a dataframe & flipping it for readability

print("=" * 60) # This section is to print out everything this code does
print("Phase 01: Data Loader & Basic Statistics")
print("=" * 60)
print(f"\nPeriod of Analysis: 2016-01-01 to 2025-12-31")
print(f"Assets: {', '.join([f'{k} ({v})' for k, v in assets.items()])}")
print(f"Portfolio Types: {', '.join(portfolios.keys())}")
print(f"Main Portfolio for Analysis: Equal Weight\n")
print("-" * 60)

print(f"\nAnnualized Drift:                  {annual_drift:>15.2%}") # Print analysis
print(f"Annualised Compounded Growth Rate: {annual_growth:>15.2%}")
print(f"Annualized Volatility:             {annual_volatility:>15.2%}")
print(f"Sharpe Ratio:                      {sharpe_ratio:>15.2f}")
print(f"Maximum Drawdown:                  {abs(max_drawdown):>15.2%}")
print(f"Total Return:                      {total_return:>15.2%}")
print(f"Skewness:                          {skewness:>15.2f}")
print(f"Kurtosis:                          {kurtosis:>15.2f}\n")
print("-" * 60)

print("\n",df_compare.round(4).to_string(),"\n") # All comparision data
print("-" * 60)

plt.style.use("seaborn-v0_8")  # Plot style
os.makedirs("data", exist_ok = True) # Ensures data folder exists
os.makedirs("outputs", exist_ok = True) # Ensures output folder exists

plt.figure(figsize = (10, 6)) # Plot 1 showing portfolio growth over time of Equal Weight Portfolio
plt.plot(cumulative.index, cumulative, color = "blue", linewidth = 2.5, label = "Portfolio Value")
plt.fill_between(cumulative.index, 1, cumulative, alpha = 0.3, color = "blue")
plt.axhline(y = 1, color = "gray", linestyle = "--", alpha = 0.7, linewidth = 1)
plt.title("Equal Weight Portfolio Growth Over Time", fontsize = 14, fontweight = "bold")
plt.xlabel("Date", fontweight = "bold")
plt.ylabel("Growth of €1", fontweight = "bold")
plt.legend(loc = "upper left")
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/01_portfolio_growth.png", dpi = 150, bbox_inches = "tight")
plt.show()

plt.figure(figsize = (10, 6)) # Plot 2 showing portfolio drawdown of equal weight portfolio
plt.fill_between(drawdown.index, 0, abs(drawdown), color = "red", alpha = 0.4, label = "Drawdown")
plt.plot(drawdown.index, abs(drawdown), color = "red", linewidth = 1, alpha = 0.8)
plt.axhline(y = 0, color = "black", linewidth = 0.8)
plt.title("Equal Weight Portfolio Drawdown from Peak", fontsize = 14, fontweight = "bold")
plt.xlabel("Date", fontweight = "bold")
plt.ylabel("Drawdown", fontweight = "bold")
plt.legend(loc = "upper left")
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/01_portfolio_drawdown.png", dpi = 150, bbox_inches = "tight")
plt.show()

plt.figure(figsize = (10, 6)) # Plot 3 showing daily returns distribution of equal weight portfolio
plt.hist(port1_log_returns, bins = 50, color = "green", alpha = 0.7, edgecolor = "black", linewidth = 0.5)
plt.title("Distribution of Daily Equal Weight Portfolio Log Returns", fontsize = 14, fontweight = "bold")
plt.xlabel("Daily Log Return", fontweight = "bold")
plt.ylabel("Frequency", fontweight = "bold")
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/01_log_returns_distribution.png", dpi = 150, bbox_inches = "tight")
plt.show()

plt.figure(figsize=(10, 6)) # Plot 4 showing all the portfolios types growth over time
for name, weights in portfolios.items():
    port_ret = 0
    for asset in weights:
        port_ret += np.exp(log_returns[asset]) * weights[asset]  
    port_ret = np.log(port_ret)
    all_cumulative = np.exp(port_ret.cumsum()) 
    plt.plot(all_cumulative.index, all_cumulative, linewidth = 1.5, label = name)
plt.axhline(y = 1, color = "gray", linestyle = "--", alpha = 0.5)
plt.title("Portfolio Growth Comparison (All Strategies)", fontsize = 14, fontweight = "bold")
plt.xlabel("Date", fontweight = "bold")
plt.ylabel("Growth of €1", fontweight = "bold")
plt.legend(loc = "upper left")
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/01_portfolios_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

prices.to_csv("data/01_asset_prices.csv") # Creating csv files to store data
log_returns.to_csv("data/01_log_returns.csv")  
port1_log_returns.to_csv("data/01_port1_log_returns.csv", header = ["Equal Weight"])
pd.DataFrame(portfolio_log_returns).to_csv("data/01_all_portfolio_log_returns.csv") 