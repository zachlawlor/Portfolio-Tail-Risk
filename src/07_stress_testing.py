# Phase 07: Multi‑Scenario Stress Testing

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os

portfolio_types = {
    "Equal Weight": {"VOO": 0.20, "EFA": 0.20, "TLT": 0.20, "AGG": 0.20, "GLD": 0.20},
    "Classic 60/40": {"VOO": 0.30, "EFA": 0.30, "TLT": 0.20, "AGG": 0.20, "GLD": 0.00},
    "Defensive": {"VOO": 0.05, "EFA": 0.05, "TLT": 0.30, "AGG": 0.30, "GLD": 0.30},
    "Aggressive Growth": {"VOO": 0.40, "EFA": 0.40, "TLT": 0.10, "AGG": 0.10, "GLD": 0.00},
    "Conservative Income": {"VOO": 0.10, "EFA": 0.10, "TLT": 0.35, "AGG": 0.35, "GLD": 0.10}
}

os.makedirs("outputs", exist_ok = True)
os.makedirs("data", exist_ok = True)

prices = pd.read_csv("data/01_asset_prices.csv", index_col = 0, parse_dates = True)
log_returns = pd.read_csv("data/01_log_returns.csv", index_col = 0, parse_dates = True)

stress_scenarios = {
    "2008 Crisis":   {"VOO": -0.40, "EFA": -0.45, "TLT": 0.15, "AGG": 0.05, "GLD": 0.25}, # These are the custom scenarios based off of real life events
    "Geopolitical":  {"VOO": -0.15, "EFA": -0.20, "TLT": 0.05, "AGG": 0.05, "GLD": 0.30},
    "Stagflation":   {"VOO": -0.20, "EFA": -0.25, "TLT": -0.20, "AGG": -0.10, "GLD": 0.35},
    "Solvency II":   {"VOO": -0.40, "EFA": -0.40, "TLT": -0.25, "AGG": -0.10, "GLD": -0.15},
    "Tech Bubble":   {"VOO": -0.45, "EFA": -0.35, "TLT": 0.10, "AGG": 0.05, "GLD": -0.10}
}

last_prices = prices.iloc[-1]
initial_value = 100_000

stress_results = []

for port_name, weights in portfolio_types.items():
    for scenario_name, shocks in stress_scenarios.items():
        loss_pct = 0
        for asset, weight in weights.items():
            loss_pct += weight * - shocks[asset] # Note I added a minus sign to report losses as positive numbers and gains as negative numbers 
        loss = loss_pct * initial_value
        
        stress_results.append({
            "Portfolio": port_name,
            "Scenario": scenario_name,
            "Loss (€)": loss,
            "Loss %": loss_pct
        })

df_results = pd.DataFrame(stress_results)

print("=" * 60) # Print Section
print("Phase 07: Multi-Scenario Stress Testing")
print("=" * 60)
print(f"\nNotional Value: €{initial_value:,}")
print(f"Stress Date: {last_prices.name.strftime('%Y-%m-%d')}")
print(f"Scenarios: {len(stress_scenarios)}")
print(f"Portfolios: {len(portfolio_types)}\n")
print("-" * 60)

print("\nScenarios & Shocks")
for name, shocks in stress_scenarios.items():
    print(f"\n{name}:", ", ".join([f"{k}: {v:+.0%}" for k, v in shocks.items()]))

print("\n" + "-" * 60)
pivot_loss = df_results.pivot(index = "Portfolio", columns = "Scenario", values = "Loss %")
pivot_loss_euros = df_results.pivot(index = "Portfolio", columns = "Scenario", values = "Loss (€)")
print("\nLoss Percentage by Portfolio and Scenario\n")
print(pivot_loss.round(4).to_string())
print("\nLoss in Euros by Portfolio and Scenario\n")
print(pivot_loss_euros.round(0).to_string())
print("\n" + "-" * 60)

print("\nWorst Case Scenario Per Portfolio\n")
for port in portfolio_types.keys():
    port_data = df_results[df_results["Portfolio"] == port]
    worst = port_data.loc[port_data["Loss %"].idxmax()]
    print(f"   {port}: {worst['Scenario']} → Loss of {worst['Loss %']:.2%} (€{worst['Loss (€)']:,.0f})")
print("\n" + "-" * 60)

plt.style.use("seaborn-v0_8") # Plot Style

plt.figure(figsize = (12, 6)) # Plot 1: heatmap using seaborn of losses by scenario 
sns.heatmap(pivot_loss * 100, annot = True, fmt = ".1f", cmap = "Reds",
            linewidths = 0.5, cbar_kws = {'label': 'Loss (%)'})
plt.title("Portfolio Stress Test: Loss Percentage by Scenario", fontsize = 14, fontweight = "bold")
plt.xlabel("Scenario", fontweight = "bold")
plt.ylabel("Portfolio", fontweight = "bold")
plt.tight_layout()
plt.savefig("outputs/07_stress_heatmap.png", dpi = 150, bbox_inches = "tight")
plt.show()

eq_weight = df_results[df_results["Portfolio"] == "Equal Weight"].sort_values("Loss %", ascending = False) # Filtering to only include equal weight portfolio
plt.figure(figsize = (10, 6)) # Plot 2: Bar chart of Equal weight port by scenario 
colors = ['darkred' if x > 0.20 else 'red' if x > 0.10 else 'orange' for x in eq_weight["Loss %"]]
bars = plt.bar(eq_weight["Scenario"], eq_weight["Loss %"] * 100, color = colors, alpha = 0.7, edgecolor = "black")
plt.axhline(0, color = "black", linewidth = 0.8)
plt.ylabel("Loss (%)", fontweight = "bold")
plt.title("Equal Weight Portfolio: Loss by Scenario", fontsize = 14, fontweight = "bold")
plt.xticks(rotation = 0, fontweight = "bold")
plt.grid(axis = "y", alpha = 0.3)
for bar, loss in zip(bars, eq_weight["Loss %"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{loss:.1%}', ha = 'center', va = 'bottom', fontsize = 10, fontweight = 'bold')
plt.tight_layout()
plt.savefig("outputs/07_stress_equal_weight.png", dpi = 150, bbox_inches = "tight")
plt.show()

plt.figure(figsize = (10, 6)) # Plot 3: Showing loss percentage across all portfolios
plot_data = pivot_loss.T * 100  # Transpose so scenarios are rows, portfolios are columns
colors = ["blue", "green", "orange", "red", "purple"]
for i, portfolio in enumerate(plot_data.columns):
    plt.plot(plot_data.index, plot_data[portfolio], 
             marker = 'o', linewidth = 2.5, markersize = 8,
             color = colors[i], label = portfolio)
plt.axhline(y = 0, color = "black", linewidth = 0.8, linestyle = "-")
plt.ylabel("Loss (%)", fontweight = "bold")
plt.xlabel("Scenario", fontweight = "bold")
plt.title("Stress Test: Loss Percentage Across All Portfolios", fontsize = 14, fontweight = "bold")
plt.legend(loc = "best")
plt.grid(axis = "y", alpha = 0.3)
plt.tight_layout()
plt.savefig("outputs/07_stress_line_graph.png", dpi = 150, bbox_inches = "tight")
plt.show()

print("Correlation Heatmaps\n")

full_corr = log_returns.corr() # Finds correlation between log returns 
pre_2022 = log_returns[log_returns.index.year < 2022].corr() # Pre 2022 correlation
crisis_2022 = log_returns[log_returns.index.year == 2022].corr() # 2022 correlation

fig, axes = plt.subplots(1, 3, figsize = (18, 5)) # Plot 4: Heat Map of all 3 correlation maps
sns.heatmap(full_corr, annot = True, fmt = ".2f", cmap = "RdBu_r", center = 0,
            linewidths = 0.5, ax = axes[0], cbar_kws = {'label': 'Correlation'})
axes[0].set_title("Full Period (2016-2025)", fontsize = 14, fontweight = "bold")
sns.heatmap(pre_2022, annot = True, fmt = ".2f", cmap = "RdBu_r", center = 0,
            linewidths = 0.5, ax = axes[1], cbar_kws = {'label': 'Correlation'})
axes[1].set_title("Pre-2022 (Pre Crisis)", fontsize = 14, fontweight = "bold")
sns.heatmap(crisis_2022, annot = True, fmt = ".2f", cmap = "RdBu_r", center = 0,
            linewidths = 0.5, ax = axes[2], cbar_kws = {'label': 'Correlation'})
axes[2].set_title("2022 (Crisis)", fontsize = 14, fontweight = "bold")
plt.tight_layout()
plt.savefig("outputs/07_correlation_heatmaps.png", dpi = 150, bbox_inches = "tight")
plt.show()

voo_tlt_pre = pre_2022.loc["VOO", "TLT"]
voo_tlt_crisis = crisis_2022.loc["VOO", "TLT"]

print(f"VOO-TLT Correlation\n")
print(f"   Pre-2022:  {voo_tlt_pre:.3f}")
print(f"   2022:      {voo_tlt_crisis:.3f}")
print(f"   Change:    {voo_tlt_crisis - voo_tlt_pre:+.3f}")

if voo_tlt_crisis > voo_tlt_pre:
    print("\nCorrelation became more positive (diversification failed in 2022).")
else:
    print("\nCorrelation became more negative (diversification held in 2022).")

print("\n" + "-" * 60)

df_results.to_csv("data/07_stress_test_results.csv", index = False) # Save results 
pivot_loss.to_csv("data/07_stress_loss_pivot.csv")
pivot_loss_euros.to_csv("data/07_stress_loss_euros_pivot.csv")