# Phase 05: Monte Carlo Bootstrap (Daily)

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok = True) # Ensures folders exists
os.makedirs("data", exist_ok = True)

port1_log_returns = pd.read_csv("data/01_port1_log_returns.csv", index_col = 0, parse_dates = True)["Equal Weight"] # Data import
all_portfolio_log_returns = pd.read_csv("data/01_all_portfolio_log_returns.csv", index_col = 0, parse_dates = True)

portfolio_value = 100_000
n_simulations = 10_000 # This is the number of times we are looping over the simulation
n_days = 1  # Daily horizon 

def bootstrap_var_es(returns_series, n_sims = 10000, days = 1): # Defines a function that takes the number of days per sim & sims

    simulated_returns = []
    
    for _ in range(n_sims):
        sampled_returns = np.random.choice(returns_series, size = days, replace = True) # Randomly chooses n = days number of daily log returns
        total_return = sampled_returns.sum()  # For days = 1, this is just a single daily return
        simulated_returns.append(total_return)
    
    simulated_returns = np.array(simulated_returns)
    
    var_95 = np.percentile(simulated_returns, 5) # Bootstrap analytics
    var_99 = np.percentile(simulated_returns, 1)
    es_95 = simulated_returns[simulated_returns <= var_95].mean()
    es_99 = simulated_returns[simulated_returns <= var_99].mean()
    
    return {
        "var_95": var_95,
        "var_99": var_99,
        "es_95": es_95,
        "es_99": es_99,
        "simulated_returns": simulated_returns
    }

equal_weight_results = bootstrap_var_es(port1_log_returns, n_simulations, n_days) # Applying function
bootstrap_var_95 = equal_weight_results["var_95"] # Storing results
bootstrap_var_99 = equal_weight_results["var_99"]
bootstrap_es_95 = equal_weight_results["es_95"]
bootstrap_es_99 = equal_weight_results["es_99"]
simulated_returns = equal_weight_results["simulated_returns"]

comparison_results = [] # Repeating for all portfolio types
for name in all_portfolio_log_returns.columns:
    rets = all_portfolio_log_returns[name]
    result = bootstrap_var_es(rets, n_simulations, n_days)
    
    comparison_results.append({
        "Portfolio": name,
        "95% VaR": result["var_95"],
        "95% ES": result["es_95"],
        "99% VaR": result["var_99"],
        "99% ES": result["es_99"],
        "VaR (€) 99%": abs(result["var_99"] * portfolio_value),
        "ES (€) 99%": abs(result["es_99"] * portfolio_value)
    })

df_compare = pd.DataFrame(comparison_results) # Data Frame of results

historical_metrics = pd.read_csv("data/02_historical_metrics.csv") # Importing all previous findings for comparison 
hist_var_95 = historical_metrics["95% VaR"].iloc[0]
hist_var_99 = historical_metrics["99% VaR"].iloc[0]
hist_es_95 = historical_metrics["95% ES"].iloc[0]
hist_es_99 = historical_metrics["99% ES"].iloc[0]

normal_metrics = pd.read_csv("data/03_normal_metrics.csv")
norm_var_95 = normal_metrics["95% VaR"].iloc[0]
norm_var_99 = normal_metrics["99% VaR"].iloc[0]
norm_es_95 = normal_metrics["95% ES"].iloc[0]
norm_es_99 = normal_metrics["99% ES"].iloc[0]

student_t_metrics = pd.read_csv("data/04_student_t_metrics.csv")
t_var_95 = student_t_metrics["95% VaR"].iloc[0]
t_var_99 = student_t_metrics["99% VaR"].iloc[0]
t_es_95 = student_t_metrics["95% ES"].iloc[0]
t_es_99 = student_t_metrics["99% ES"].iloc[0]

print("=" * 60) # Print Section
print("Phase 05: Monte Carlo Bootstrap (Daily)")
print("=" * 60)
print(f"\nPortfolio: Equal Weight")
print(f"Notional Value: €{portfolio_value:,}")
print(f"Simulations: {n_simulations:,}")
print(f"Days per Simulation: {n_days}")
print(f"Period: 2016-01-01 to 2025-12-31\n")
print("-" * 60)

print(f"\n95% Bootstrap VaR:                  {bootstrap_var_95:>8.2%}")
print(f"99% Bootstrap VaR:                  {bootstrap_var_99:>8.2%}")
print(f"95% Bootstrap ES:                   {bootstrap_es_95:>8.2%}")
print(f"99% Bootstrap ES:                   {bootstrap_es_99:>8.2%}\n")
print("-" * 60)

print(f"\n{'Metric':<20} {'Historical':>12} {'Normal':>12} {'Student-t':>12} {'Bootstrap':>12}\n")
print(f"{'95% VaR':<20} {hist_var_95:>12.2%} {norm_var_95:>12.2%} {t_var_95:>12.2%} {bootstrap_var_95:>12.2%}")
print(f"{'99% VaR':<20} {hist_var_99:>12.2%} {norm_var_99:>12.2%} {t_var_99:>12.2%} {bootstrap_var_99:>12.2%}")
print(f"{'95% ES':<20} {hist_es_95:>12.2%} {norm_es_95:>12.2%} {t_es_95:>12.2%} {bootstrap_es_95:>12.2%}")
print(f"{'99% ES':<20} {hist_es_99:>12.2%} {norm_es_99:>12.2%} {t_es_99:>12.2%} {bootstrap_es_99:>12.2%}")

print("\n" + "-" * 60 + "\n")
print(df_compare.round(4).to_string(index = False))
print("\n" + "-" * 60)

plt.style.use("seaborn-v0_8")  # Plot style

plt.figure(figsize = (10, 6)) # Plot 1: Distribution of simulated daily returns 
plt.hist(simulated_returns * 100, bins = 60, color = "blue", alpha = 0.7, edgecolor = "black", linewidth = 0.3)
plt.axvline(bootstrap_var_95 * 100, color = "orange", linestyle = "--", linewidth = 2, label = f"95% VaR: {bootstrap_var_95:.2%}")
plt.axvline(bootstrap_var_99 * 100, color = "red", linestyle = "--", linewidth = 2, label = f"99% VaR: {bootstrap_var_99:.2%}")
plt.axvline(bootstrap_es_95 * 100, color = "darkorange", linestyle = "-", linewidth = 2, label = f"95% ES: {bootstrap_es_95:.2%}")
plt.axvline(bootstrap_es_99 * 100, color = "darkred", linestyle = "-", linewidth = 2, label = f"99% ES: {bootstrap_es_99:.2%}")
plt.xlabel("Simulated Daily Log Return (%)", fontweight = "bold")
plt.ylabel("Frequency", fontweight = "bold")
plt.title("Equal Weight: Monte Carlo Bootstrap Distribution (10,000 Simulations)", fontsize = 14, fontweight = "bold")
plt.legend()
plt.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/05_bootstrap_distribution.png", dpi = 150, bbox_inches = "tight")
plt.show()

plt.figure(figsize = (10, 6)) # Plot 2: Comparison of VaR
metrics = ["95% VaR", "99% VaR"]
historical_vals = [hist_var_95 * 100, hist_var_99 * 100]
normal_vals = [norm_var_95 * 100, norm_var_99 * 100]
t_vals = [t_var_95 * 100, t_var_99 * 100]
bootstrap_vals = [bootstrap_var_95 * 100, bootstrap_var_99 * 100]
x_pos = np.arange(len(metrics))
width = 0.2
plt.bar(x_pos - 1.5*width, historical_vals, width, label = "Historical", color = "blue", alpha = 0.7, edgecolor = "black")
plt.bar(x_pos - 0.5*width, normal_vals, width, label = "Normal", color = "red", alpha = 0.7, edgecolor = "black")
plt.bar(x_pos + 0.5*width, t_vals, width, label = "Student-t", color = "green", alpha = 0.7, edgecolor = "black")
plt.bar(x_pos + 1.5*width, bootstrap_vals, width, label = "Bootstrap", color = "purple", alpha = 0.7, edgecolor = "black")
plt.axhline(y = 0, color = "black", linewidth = 0.8)
plt.xticks(x_pos, metrics, fontweight = "bold")
plt.ylabel("Loss (%)", fontweight = "bold")
plt.title("Equal Weight: VaR Comparison (All Models)", fontsize = 14, fontweight = "bold")
plt.legend()
plt.grid(axis = "y", alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/05_var_comparison.png", dpi = 150, bbox_inches = "tight")
plt.show()

plt.figure(figsize = (10, 6)) # Plot 3: Comparison of ES
metrics = ["95% ES", "99% ES"]
historical_vals = [hist_es_95 * 100, hist_es_99 * 100]
normal_vals = [norm_es_95 * 100, norm_es_99 * 100]
t_vals = [t_es_95 * 100, t_es_99 * 100]
bootstrap_vals = [bootstrap_es_95 * 100, bootstrap_es_99 * 100]
x_pos = np.arange(len(metrics))
width = 0.2
plt.bar(x_pos - 1.5*width, historical_vals, width, label = "Historical", color = "blue", alpha = 0.7, edgecolor = "black")
plt.bar(x_pos - 0.5*width, normal_vals, width, label = "Normal", color = "red", alpha = 0.7, edgecolor = "black")
plt.bar(x_pos + 0.5*width, t_vals, width, label = "Student-t", color = "green", alpha = 0.7, edgecolor = "black")
plt.bar(x_pos + 1.5*width, bootstrap_vals, width, label = "Bootstrap", color = "purple", alpha = 0.7, edgecolor = "black")
plt.axhline(y = 0, color = "black", linewidth = 0.8)
plt.xticks(x_pos, metrics, fontweight = "bold")
plt.ylabel("Loss (%)", fontweight = "bold")
plt.title("Equal Weight: ES Comparison (All Models)", fontsize = 14, fontweight = "bold")
plt.legend()
plt.grid(axis = "y", alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/05_es_comparison.png", dpi = 150, bbox_inches = "tight")
plt.show()

results_df = pd.DataFrame({
    "Portfolio": ["Equal Weight"],
    "95% VaR": [bootstrap_var_95],
    "99% VaR": [bootstrap_var_99],
    "95% ES": [bootstrap_es_95],
    "99% ES": [bootstrap_es_99]
})

results_df.to_csv("data/05_bootstrap_metrics.csv", index = False)
df_compare.to_csv("data/05_bootstrap_comparison.csv", index = False)