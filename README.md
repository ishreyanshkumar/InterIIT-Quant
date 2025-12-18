# Algorithmic Strategy Development on Multi-Feature Time Series (Team 33)
### Inter IIT Tech Meet 14.0 - High-Frequency Trading Challenge

## 📈 Executive Summary
This repository contains the source code for a robust **Reinforcement Learning (RL)** intraday trading strategy developed for the Inter IIT Tech Meet 14.0. The model leverages **Proximal Policy Optimization (PPO)** to navigate high-frequency market data, utilizing a rich state space of technical indicators, Heikin-Ashi structures, and adaptive volatility measures.

The strategy is rigorously evaluated on two distinct tickers, EBX and EBY, demonstrating highly profitable and robust performance profiles.

### 🏆 Performance Highlights
The RL agent demonstrated exceptional risk-adjusted returns and stability in out-of-sample evaluations. [cite_start]According to the performance report, the strategy achieved the following metrics[cite: 18]:

| Metric | EBX (255 Days) | EBY (140 Days) |
| :--- | :--- | :--- |
| **Annualized Return** | **81.61%** | **77.97%** |
| **Calmar Ratio** | **70.96** | **63.91** |
| **Sharpe Ratio** | 7.30 | 4.91 |
| **Max Drawdown** | 1.15% | 1.22% |
| **Win Rate** | ~69% | ~69% |
| **Avg Trades/Day** | 13.77 | 18.85 |

---

## 🧠 Strategy Architecture

### 1. Data Engineering
* **Resampling:** Raw tick/second data is resampled into **2-minute candles** to capture meaningful market structure and reduce noise.
* **Feature Space**:
    * **Trend:** Heikin-Ashi transformations, Johnny Ribbon (Regime detection).
    * **Momentum:** RSI, CCI, CMO, Aroon.
    * **Volatility:** ATR, Standard Deviation, Chop Index.
    * **Time Encoding:** Cyclical sine/cosine features for time-of-day awareness.
    * **Adaptive Filters:** KAMA (Kaufman's Adaptive Moving Average).

### 2. Reinforcement Learning (PPO)
* **Agent:** PPO (Proximal Policy Optimization) using `stable-baselines3`.
* **Policy:** `MlpPolicy` with a dual [256, 256] network architecture.
* **Reward Function:** A custom shaped reward function incorporating:
    * Realized PnL scaling.
    * Penalties for stop-loss hits (`-100`) and end-of-day forced closures (`-10`).
    * Bonuses for "waiting" to avoid over-trading in chop (`0.1`).
* **Training:** Parallelized environments (`SubprocVecEnv`) with `VecNormalize` for stable convergence.

---

## 📂 Repository Structure

```text
Team33_InterIIT/
├── data/
│   ├── EBX/                 # Raw csv files for EBX
│   └── EBY/                 # Raw csv files for EBY
├── Models_EBX/              # Saved PPO models and VecNormalize stats for EBX
├── Models_EBY/              # Saved PPO models and VecNormalize stats for EBY
├── test_results/            # Output logs, equity curves, and performance summaries
├── signals_EBX/             # Generated trade signals (CSV)
├── training_plots/          # Entropy and Variance convergence plots
├── RL.py                    # Main CLI script for Train/Test/Backtest
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Team33/InterIIT-Algo-Trading.git](https://github.com/Team33/InterIIT-Algo-Trading.git)
    cd InterIIT-Algo-Trading
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Setup:**
    * Place raw CSV data files in folders named `EBX` and `EBY` in the root directory.
    * *Note: The `alpha_research` module is required for the `backtest_ebullient` mode. Ensure `alpha_research.py` is in your PYTHON PATH if running full event-driven simulations.*

---

## 🚀 Usage Guide

The `RL.py` script serves as the command-line interface for the entire pipeline.

### 1. Training the Agent
Train a new PPO model for a specific ticker. This process handles data resampling, train/test splitting, and parallel training.

```bash
# Syntax: python RL.py train <TICKER>
python RL.py train EBX
```
*Output: Saves model to `Models_EBX/ppo_trading_model_EBX.zip` and plots training metrics to `training_plots/`.*

### 2. Testing the Model
Run the trained model on out-of-sample data to generate performance metrics, equity curves, and trade logs.

```bash
# Syntax: python RL.py test <TICKER> [optional: specific_day_number]
python RL.py test EBX
```
*Output: Generates detailed reports in `test_results/` and signal files in `signals_EBX/`.*

### 3. Backtesting (Simulation)
Run the event-driven backtest using the generated signals (requires the competition's `alpha_research` module).

```bash
# Syntax: python RL.py backtest_ebullient <TICKER>
python RL.py backtest_ebullient EBX
```

---

## 📊 Visual Analysis

The strategy produces comprehensive visual diagnostics found in `test_trade_plots/` and `training_plots/`:
* **Equity Curves:** Visual confirmation of steady capital growth.
* **Drawdown Charts:** Monitoring of risk depth and duration.
* **Trade Visualization:** Candlestick charts overlayed with Entry/Exit points for every test day.
* **Training Metrics:** Entropy loss and Explained Variance plots to verify convergence.

---

## 📜 Requirements

* Python 3.8+
* `numpy`
* `pandas`
* `gymnasium`
* `stable-baselines3`
* `torch`
* `matplotlib`
* `tqdm`
