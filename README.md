# Portfolio Tail Risk Analytics

A quantitative finance & actuarial science project investigating **how different risk models measure portfolio tail risk and how accurately they predict extreme losses**.

The **Equal Weight portfolio is the primary focus** of the analysis, with four alternative portfolio strategies used for comparison.

## Goal

The goal of the project is to **evaluate different approaches to measuring portfolio tail risk and determine which models provide the most reliable estimates of extreme losses**, particularly during periods of market stress.

This is assessed through VaR and Expected Shortfall estimation, simulation, out-of-sample backtesting, stress testing and correlation analysis.

## Hypothesis

> **Parametric models that account for heavier-tailed return distributions, such as the Student-t model, will provide more accurate estimates of extreme portfolio losses than models based on the Normal distribution, particularly during periods of market stress.**

The hypothesis is tested by comparing risk estimates across models and evaluating their out-of-sample performance through backtesting and stress analysis.

## Project Phases

### 01 — Data Loader & Basic Statistics

Market data acquisition, portfolio construction, return calculation and basic performance statistics.

### 02 — Historical VaR & Expected Shortfall

Empirical estimates of portfolio tail risk based on observed historical returns.

### 03 — Normal VaR & Expected Shortfall

Parametric risk estimates under a Normal distribution assumption.

### 04 — Student-t VaR & Expected Shortfall

Heavy-tailed modelling of portfolio returns and extreme losses using the Student-t distribution.

### 05 — Monte Carlo Bootstrap (Daily)

Simulation of portfolio returns through Monte Carlo resampling of observed historical returns.

### 06 — Expanding Window Backtesting

Out-of-sample evaluation of VaR models using an expanding estimation window and Kupiec coverage testing.

### 07 — Multi-Scenario Stress Testing

Hypothetical market stress scenarios and analysis of portfolio behaviour and correlation dynamics.

### Dashboard

An interactive Streamlit dashboard bringing the results from Phases 01–07 together.

The dashboard provides:

* Portfolio performance comparison
* VaR & Expected Shortfall analysis
* Risk-model comparison
* VaR backtesting
* Stress-test analysis
* Correlation analysis

The Equal Weight portfolio is the primary focus, with the alternative strategies available for comparison.

## Data

Daily market data covering **2016–2025**, obtained using `yfinance`.

Processed data and analysis outputs are included in the `data/` directory.

## Getting Started

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd Portfolio-Tail-Risk
pip install -r requirements.txt
```

Launch the dashboard:

```bash
streamlit run dashboard.py
```

The repository contains the required processed data and analysis outputs, so the **dashboard can be launched directly without rerunning the analysis phases**.

To reproduce the full analysis from scratch, run **Phases 01–07 in order** before launching the dashboard.

## Tech Stack

`Python` · `NumPy` · `Pandas` · `SciPy` · `Matplotlib` · `Seaborn` · `Streamlit` · `yfinance`

## Report

A full technical report accompanies the project, covering the detailed methodology, mathematical framework, results, interpretation, limitations and references.

[View the Full Technical Report](docs/Portfolio%20Tail-Risk%20Report.pdf)

## Project Structure

```text
Portfolio-Tail-Risk/
├── data/
├── docs/
│   ├── Portfolio Tail-Risk Report.pdf
├── outputs/
├── src/
│   ├── 01_data_loader.py
│   ├── 02_historical.py
│   ├── 03_normal.py
│   ├── 04_student_t.py
│   ├── 05_monte_carlo.py
│   ├── 06_back_testing.py
│   └── 07_stress_testing.py
├── dashboard.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Author

**Zach Lawlor** — Final Year BSc Financial Mathematics & Actuarial Science — University College Cork

## Contact

Email:      zach6lawlor@gmail.com<br>
LinkedIn:   [zachlawlor](https://www.linkedin.com/in/zachlawlor)<br>
GitHub:     [zachlawlor](https://github.com/zachlawlor)
