# Phase 06: Expanding Window Backtesting 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm, t, chi2
import os

os.makedirs("outputs", exist_ok = True)
os.makedirs("data", exist_ok = True)

port1_log_returns = pd.read_csv("data/01_port1_log_returns.csv", index_col = 0, parse_dates = True)["Equal Weight"]
all_portfolio_log_returns = pd.read_csv("data/01_all_portfolio_log_returns.csv", index_col = 0, parse_dates = True)

portfolio_value = 100_000
test_years = [2021, 2022, 2023, 2024, 2025] # These are the years we will test using data up to this year e.g 2022 backtest trains on 2016-2021 data

def calculate_var(returns_series, method = "historical", alpha = 0.01):  # This functions calclates var for a specified training period
    if method == "historical": 
        return np.percentile(returns_series, alpha * 100)
    elif method == "normal":
        mu = returns_series.mean()
        sigma = returns_series.std()
        return norm.ppf(alpha, mu, sigma)
    elif method == "student_t":
        df, loc, scale = t.fit(returns_series)
        return t.ppf(alpha, df, loc, scale)
    else:
        raise ValueError("Method must be 'historical', 'normal', or 'student_t'")

def kupiec_test(breaches, total_days, confidence=0.99):# This is a likelihood ratio test - we accept if p value > 0.05 generally for 95% VaR

    p = 1 - confidence
    x = breaches
    T = total_days
    if x == 0 or x == T:
        return np.nan
    p_hat = x / T
    LR = -2 * np.log( ((1-p)**(T-x) * p**x) / ((1-p_hat)**(T-x) * p_hat**x) ) # LR Test 
    p_value = 1 - chi2.cdf(LR, df=1)
    return p_value

backtest_results = [] # Store the results of the backtest 

for year in test_years:
    train_data = port1_log_returns[port1_log_returns.index.year < year] # Reminder: this is only using the equal weight portfolio at first 
    test_data = port1_log_returns[port1_log_returns.index.year == year]
    
    if len(train_data) < 100 or len(test_data) == 0: # This is to ensure there is enough data to train on accurately 
        continue
    
    hist_var_99 = calculate_var(train_data, "historical", 0.01) # 99% VaR
    norm_var_99 = calculate_var(train_data, "normal", 0.01)
    t_var_99 = calculate_var(train_data, "student_t", 0.01)
    
    hist_breaches_99 = (test_data < hist_var_99).sum() # Counts the number of times that the VaR ( 99% ) is exceeded / breached using all the models
    norm_breaches_99 = (test_data < norm_var_99).sum()
    t_breaches_99 = (test_data < t_var_99).sum()

    hist_var_95 = calculate_var(train_data, "historical", 0.05) # 95% VaR
    norm_var_95 = calculate_var(train_data, "normal", 0.05)
    t_var_95 = calculate_var(train_data, "student_t", 0.05)
    
    hist_breaches_95 = (test_data < hist_var_95).sum() # Counts the number of times that the VaR ( 95% ) is exceeded / breached using all the models
    norm_breaches_95 = (test_data < norm_var_95).sum()
    t_breaches_95 = (test_data < t_var_95).sum()
    
    expected_99 = len(test_data) * 0.01 # Expected number of breaches under both 95% and 99% VaR
    expected_95 = len(test_data) * 0.05
    
    p_hist_99 = kupiec_test(hist_breaches_99, len(test_data), 0.99) # Kupiec p-values being calculated by function
    p_norm_99 = kupiec_test(norm_breaches_99, len(test_data), 0.99)
    p_t_99 = kupiec_test(t_breaches_99, len(test_data), 0.99)
    p_hist_95 = kupiec_test(hist_breaches_95, len(test_data), 0.95)
    p_norm_95 = kupiec_test(norm_breaches_95, len(test_data), 0.95)
    p_t_95 = kupiec_test(t_breaches_95, len(test_data), 0.95)
    
    backtest_results.append({ # Creates a dictionary to store results of the backtest for each year & loops
        "Year": year,
        "Days": len(test_data),
        "Expected 99%": expected_99,
        "Expected 95%": expected_95,
        "Hist 99": hist_breaches_99,
        "Norm 99": norm_breaches_99,
        "T 99": t_breaches_99,
        "Hist 95": hist_breaches_95,
        "Norm 95": norm_breaches_95,
        "T 95": t_breaches_95,
        "Hist 99 p": p_hist_99,
        "Norm 99 p": p_norm_99,
        "T 99 p": p_t_99,
        "Hist 95 p": p_hist_95,
        "Norm 95 p": p_norm_95,
        "T 95 p": p_t_95
    })

df_backtest = pd.DataFrame(backtest_results)

all_backtest_results = [] # repeats the same as above but has an additional loop to go through all 5 portfolio types 

for name in all_portfolio_log_returns.columns:
    rets = all_portfolio_log_returns[name]
    
    for year in test_years:
        train_data = rets[rets.index.year < year]
        test_data = rets[rets.index.year == year]
        
        if len(train_data) < 100 or len(test_data) == 0:
            continue
        
        hist_var_99 = calculate_var(train_data, "historical", 0.01)
        norm_var_99 = calculate_var(train_data, "normal", 0.01)
        t_var_99 = calculate_var(train_data, "student_t", 0.01)
        
        all_backtest_results.append({
            "Portfolio": name,
            "Year": year,
            "Days": len(test_data),
            "Historical 99": (test_data < hist_var_99).sum(),
            "Normal 99": (test_data < norm_var_99).sum(),
            "Student-t 99": (test_data < t_var_99).sum()
        })

df_all_backtest = pd.DataFrame(all_backtest_results)

print("=" * 60) # Print Section
print("Phase 06: Expanding Window Backtesting")
print("=" * 60)
print("\nTesting Period: 2021-2025")
print("Training: All data before each test year")
print(f"Portfolio: Equal Weight\n")
print("-" * 60 + "\n")

for _, row in df_backtest.iterrows():
    year = int(row["Year"])
    days = int(row["Days"])
    print(f"\n{year} ({days} days)\n")
    print(f"99% VaR:")
    print(f"Expected: {row['Expected 99%']:.1f} | Historical: {row['Hist 99']:.0f} (p={row['Hist 99 p']:.3f}), Normal: {row['Norm 99']:.0f} (p={row['Norm 99 p']:.3f}), Student-t: {row['T 99']:.0f} (p={row['T 99 p']:.3f})")
    print(f"\n95% VaR:")
    print(f"Expected: {row['Expected 95%']:.1f} | Historical: {row['Hist 95']:.0f} (p={row['Hist 95 p']:.3f}), Normal: {row['Norm 95']:.0f} (p={row['Norm 95 p']:.3f}), Student-t: {row['T 95']:.0f} (p={row['T 95 p']:.3f})")

print("\n" + "-" * 60 + "\n")
pivot_hist = df_all_backtest.pivot(index = "Portfolio", columns = "Year", values = "Historical 99")
pivot_norm = df_all_backtest.pivot(index = "Portfolio",columns = "Year", values = "Normal 99")
pivot_t = df_all_backtest.pivot(index = "Portfolio",columns = "Year", values = "Student-t 99")

print("Historical 99% Breaches\n")
print(pivot_hist.to_string())
print("\nNormal 99% Breaches\n")
print(pivot_norm.to_string())
print("\nStudent-t 99% Breaches\n")
print(pivot_t.to_string())
print("\n" + "-" * 60)

plt.style.use("seaborn-v0_8")  # Plot style

plt.figure(figsize = (12, 6)) # Plot 1: Breaches Over time - equal weight portfolio 99% VaR breaches
years = df_backtest["Year"]
x_pos = np.arange(len(years))
width = 0.15
plt.bar(x_pos - 0.75*width, df_backtest["Hist 99"], width, label = "Historical 99%", color = "blue", alpha = 0.7, edgecolor = "black")
plt.bar(x_pos - 0.25*width, df_backtest["Norm 99"], width, label = "Normal 99%", color = "red", alpha = 0.7, edgecolor = "black")
plt.bar(x_pos + 0.25*width, df_backtest["T 99"], width, label = "Student-t 99%", color = "green", alpha = 0.7, edgecolor = "black")
plt.bar(x_pos + 0.75*width, df_backtest["Expected 99%"], width, label = "Expected 99%", color = "black", alpha = 0.5, edgecolor = "black")
plt.xticks(x_pos, years, fontweight = "bold")
plt.xlabel("Test Year", fontweight = "bold")
plt.ylabel("Number of Breaches", fontweight = "bold")
plt.title("Equal Weight: 99% VaR Breaches by Model (2021-2025)", fontsize = 14, fontweight = "bold")
plt.legend()
plt.grid(axis = "y", alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/06_backtest_breaches_99.png", dpi = 150, bbox_inches = "tight")
plt.show()

plt.figure(figsize = (10, 6)) # Plot 2: Normal breaches 99% - traffic light colours
breach_rates = df_backtest["Norm 99"] / df_backtest["Days"] * 100
colors = []
for rate in breach_rates:
    if rate <= 1.5:
        colors.append("green")
    elif rate <= 2.5:
        colors.append("orange")
    else:
        colors.append("red")
bars = plt.bar(years, breach_rates, color = colors, alpha = 0.7, edgecolor = "black", linewidth = 1)
plt.axhline(y = 0, color = "green", linestyle = "--", linewidth = 1.5, label = "Green Zone (≤1.5%)")
plt.axhline(y = 1.5, color = "orange", linestyle = "--", linewidth = 1.5, label = "Orange Zone (1.5-2.5%)")
plt.axhline(y = 2.5, color = "red", linestyle = "--", linewidth = 1.5, label = "Red Zone (>=2.5%)")
for bar, rate in zip(bars, breach_rates):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{rate:.1f}%', ha = 'center', va = 'bottom', fontsize = 10, fontweight = 'bold')
plt.xlabel("Test Year", fontweight = "bold")
plt.ylabel("Breach Rate (%)", fontweight = "bold")
plt.title("Normal Model (99% VaR): Traffic Light Backtest (Equal Weight)", fontsize = 14, fontweight = "bold")
plt.legend()
plt.grid(axis = "y", alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/06_backtest_traffic_light.png", dpi = 150, bbox_inches = "tight")
plt.show()

plt.figure(figsize = (10, 6)) # Plot 3: 2022 Crisis comparison 
df_2022 = df_all_backtest[df_all_backtest["Year"] == 2022]
portfolios = df_2022["Portfolio"]
x_pos = np.arange(len(portfolios))
width = 0.25
plt.bar(x_pos - width, df_2022["Historical 99"], width, label = "Historical", color = "blue", alpha = 0.7, edgecolor = "black")
plt.bar(x_pos, df_2022["Normal 99"], width, label = "Normal", color = "red", alpha = 0.7, edgecolor = "black")
plt.bar(x_pos + width, df_2022["Student-t 99"], width, label = "Student-t", color = "green", alpha = 0.7, edgecolor = "black")
plt.axhline(y = 2.5, color = "black", linestyle = "--", linewidth = 1.5, label = "Expected Breaches (~2.5)")
plt.xticks(x_pos, portfolios, rotation = 15, fontweight = "bold")
plt.xlabel("Portfolio", fontweight = "bold")
plt.ylabel("Number of Breaches (99% VaR)", fontweight = "bold")
plt.title("2022 Crisis: 99% VaR Breaches by Portfolio", fontsize = 14, fontweight = "bold")
plt.legend()
plt.grid(axis = "y", alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/06_backtest_2022_crisis.png", dpi = 150, bbox_inches = "tight")
plt.show()

plt.figure(figsize = (10, 6)) # Plot 4: 95% vs 99% VaR breach rate analysis for normal distribution 
x_pos = np.arange(len(years))
width = 0.35
breach_99 = df_backtest["Norm 99"] / df_backtest["Days"] * 100
breach_95 = df_backtest["Norm 95"] / df_backtest["Days"] * 100
plt.bar(x_pos - width/2, breach_95, width, label = "95% VaR", color = "orange", alpha = 0.7, edgecolor = "black")
plt.bar(x_pos + width/2, breach_99, width, label = "99% VaR", color = "red", alpha = 0.7, edgecolor = "black")
plt.axhline(y = 1.0, color = "green", linestyle = "--", linewidth = 1.5, label = "Expected 99% (1%)")
plt.axhline(y = 5.0, color = "blue", linestyle = "--", linewidth = 1.5, label = "Expected 95% (5%)")
plt.xticks(x_pos, years, fontweight = "bold")
plt.xlabel("Test Year", fontweight = "bold")
plt.ylabel("Breach Rate (%)", fontweight = "bold")
plt.title("Normal Model: 95% vs 99% VaR Breach Rates (Equal Weight)", fontsize = 14, fontweight = "bold")
plt.legend()
plt.grid(axis = "y", alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/06_backtest_95_vs_99.png", dpi = 150, bbox_inches = "tight")
plt.show()

df_backtest.to_csv("data/06_backtest_results.csv", index = False) # Save results
df_all_backtest.to_csv("data/06_backtest_all_portfolios.csv", index = False)