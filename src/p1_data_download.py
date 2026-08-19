# Phase 1: Data Download & Analysis

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import os 

os.makedirs("data", exist_ok = True) # Ensures data folder exists
os.makedirs("outputs", exist_ok = True) # Ensures output folder exists

assets = {"S&P 500" : "VOO", "Euro Stoxx 50" : "FEZ", "US Treasury" : "TLT", "Gold" : "GLD"} # Assets chosen for project
weights = {"VOO" : 0.5, "FEZ" : 0.2, "TLT" : 0.2, "GLD" : 0.1} # Asset weights -> Must sum to 1

data = yf.download(list(assets.values()), start = "2021-01-01", end = "2026-01-01", auto_adjust = True) # Downloading 5 years worth of data

prices = data["Close"] # Calculating daily returns
returns = prices.pct_change()
returns = returns.dropna()

portfolio_returns = 0 # Calculating Portfolio daily return
for asset in weights:
    portfolio_returns += returns[asset] * weights[asset]

annual_return = portfolio_returns.mean() * 252 # Annual mean return
annual_volatility = portfolio_returns.std() * np.sqrt(252) # Annual volatility

treasury = yf.download("^TNX", start = "2021-01-01", end = "2026-01-01")["Close"].dropna() # Downloading 10 year treasury yield to compute risk free rate
risk_free_rate = treasury.mean().iloc[0] / 100 # Calculating risk-free rate
sharpe_ratio = float((annual_return - risk_free_rate) / annual_volatility) # Sharpe Ratio

cumulative = (1 + portfolio_returns).cumprod() # Cumulative return
peak = cumulative.expanding().max() 
drawdown = (cumulative - peak) / peak
max_drawdown = drawdown.min() # Computes max drawdown -> biggest downward movement

total_return = cumulative.iloc[-1] - 1
annual_growth = cumulative.iloc[-1] ** (252 / len(portfolio_returns)) - 1 # Compouned annual growth rate


print(f"\nAnnualized Return:                 {annual_return:>15.2%}") # Print analysis
print(f"Annualised Compounded Growth Rate: {annual_growth:>15.2%}")
print(f"Annualized Volatility:             {annual_volatility:>15.2%}")
print(f"Sharpe Ratio:                      {sharpe_ratio:>15.2f}")
print(f"Maximum Drawdown:                  {max_drawdown:>15.2%}")
print(f"Total Return:                      {total_return:>15.2%}\n")

plt.figure(figsize = (10, 6)) # Plot 1 showing portfolio growth over time
plt.plot(cumulative.index, cumulative, color = "blue", linewidth = 2.5, label = "Portfolio Value")
plt.fill_between(cumulative.index, 1, cumulative, alpha = 0.3, color = "blue")
plt.axhline(y = 1, color = "gray", linestyle = "--", alpha = 0.7, linewidth = 1)
plt.title("Portfolio Growth Over Time", fontsize = 14, fontweight = "bold")
plt.xlabel("Date", fontweight = "bold")
plt.ylabel("Growth of €1", fontweight = "bold")
plt.legend(loc = "upper left")
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/portfolio_growth.png", dpi = 150, bbox_inches = "tight")
plt.show()

plt.figure(figsize = (10, 6)) # Plot 2 showing portfolio drawdown
plt.fill_between(drawdown.index, 0, drawdown, color = "red", alpha = 0.4, label = "Drawdown")
plt.plot(drawdown.index, drawdown, color = "red", linewidth = 1, alpha = 0.8)
plt.axhline(y = 0, color = "black", linewidth = 0.8)
plt.title("Portfolio Drawdown from Peak", fontsize = 14, fontweight = "bold")
plt.xlabel("Date", fontweight = "bold")
plt.ylabel("Drawdown", fontweight = "bold")
plt.legend(loc = "lower left")
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/portfolio_drawdown.png", dpi = 150, bbox_inches = "tight")
plt.show()

plt.figure(figsize = (10, 6)) # Plot 3 showing daily returns distribution
plt.hist(portfolio_returns, bins = 50, color = "green", alpha = 0.7, edgecolor = "black", linewidth = 0.5)
plt.title("Distribution of Daily Portfolio Returns", fontsize = 14, fontweight = "bold")
plt.xlabel("Daily Return", fontweight = "bold")
plt.ylabel("Frequency", fontweight = "bold")
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/returns_distribution.png", dpi = 150, bbox_inches = "tight")
plt.show()

prices.to_csv("data/asset_prices.csv") # Creating csv files to store data
returns.to_csv("data/asset_returns.csv")
portfolio_returns.to_csv("data/portfolio_returns.csv", header = ["PORT"])