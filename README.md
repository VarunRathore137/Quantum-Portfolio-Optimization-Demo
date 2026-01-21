# The Hybrid Quantum Portfolio Optimizer

**An Interactive Quantum-Inspired Portfolio Optimization System Using QAOA**

This project demonstrates how quantum computing principles can be applied to real-world portfolio optimization problems. It uses the Quantum Approximate Optimization Algorithm (QAOA) to select optimal stock portfolios from real market data, helping investors maximize returns while minimizing risk.

---

## 🎯 Overview

The Quantum Portfolio Optimization Demo solves the classic Mean-Variance Portfolio Optimization problem using quantum computing techniques. By leveraging QAOA, this system:

- **Fetches real-time market data** from Yahoo Finance (MSFT, TSLA, AAPL)
- **Formulates the problem as a QUBO** (Quadratic Unconstrained Binary Optimization)
- **Solves using quantum algorithms** via Qiskit's QAOA implementation
- **Allocates shares intelligently** based on risk-adjusted returns
- **Provides interactive visualization** of results and recommendations

### Key Features

✅ **Interactive Jupyter Notebook** - Step-by-step walkthrough with explanations  
✅ **Real Market Data** - Live stock prices and historical returns  
✅ **QAOA Optimization** - Quantum-inspired portfolio selection  
✅ **Risk Management** - Covariance-based risk assessment  
✅ **Smart Allocation** - Sharpe ratio-based share distribution  
✅ **Professional Visualizations** - Charts and performance metrics  
✅ **Classical Benchmark** - Compare QAOA vs exact solver  

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Understanding the Notebook](#understanding-the-notebook)
- [Sample Output](#sample-output)
- [Configuration \& Experimentation](#configuration--experimentation)
- [Technical Architecture](#technical-architecture)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)

---

## 🚀 Installation

### Prerequisites

- **Python 3.11+** (recommended)
- **pip** package manager
- **Jupyter Notebook** or **JupyterLab**
- **Virtual environment** (recommended)

### Step-by-Step Setup

```bash
# 1. Clone or navigate to the project directory
cd Quantum-Portfolio-Optimization-Demo

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch Jupyter Notebook
jupyter notebook
```

---

## ⚡ Quick Start

1. **Open the notebook**: Launch `main.ipynb` in Jupyter
2. **Run all cells**: Execute from top to bottom sequentially
3. **Provide user input** when prompted:
   - Number of shares to buy (minimum 5)
   - Maximum budget per share (e.g., $250)

4. **View results**:
   - Portfolio selection (which stocks to buy)
   - Share allocation (how many shares of each)
   - Expected profit and investment breakdown

**Expected Runtime**: ~30-60 seconds depending on market data fetch time

---

## 🧠 How It Works

### The Portfolio Optimization Problem

Given:
- A set of stocks (MSFT, TSLA, AAPL)
- Historical price data (returns and risk)
- Budget constraints

**Goal**: Select stocks that maximize expected return while minimizing portfolio risk.

### Mathematical Formulation

The objective function is:

```
minimize: q(x^T Σ x) - (r^T x) + P(Σx_i - k)²

where:
  x = binary selection vector (1 = selected, 0 = not selected)
  Σ = covariance matrix (risk)
  r = expected returns vector
  q = risk tolerance factor (default: 0.5)
  k = portfolio size (number of stocks to select)
  P = penalty weight for constraint violations
```

### Solution Approach

1. **Data Collection**: Fetch 6 months of historical data via Yahoo Finance API
2. **Statistical Analysis**: Calculate expected returns (μ) and covariance matrix (Σ)
3. **QUBO Formulation**: Convert portfolio problem to binary optimization
4. **Quantum Hamiltonian**: Map QUBO to Pauli operators
5. **QAOA Execution**: Run quantum algorithm to find optimal selection
6. **Share Allocation**: Distribute total shares based on Sharpe ratios
7. **Result Interpretation**: Decode bitstring and calculate investment details

---

## 📖 Understanding the Notebook

The notebook is organized into **7 main sections**:

### Section 1: User Configuration
**What it does**: Collects your investment parameters  
**Inputs**: Total shares, budget per share  
**Output**: Validated configuration  

```python
total_shares = 5
budget_per_share = 250
total_budget = 1250  # Calculated
```

---

### Section 2: Market Data Acquisition
**What it does**: Fetches and processes historical stock data  
**Data source**: Yahoo Finance API (6-month lookback)  
**Outputs**: 
- `expected_returns`: Daily return for each stock
- `risk_matrix`: Covariance matrix (risk relationships)
- `price_data`: Historical prices for visualization

**Visualizations**:
- Stock price trends over 6 months
- Expected returns table with annualized projections
- Covariance matrix heatmap

---

### Section 3: QUBO Formulation
**What it does**: Translates the portfolio problem into quantum form  
**Key components**:
- **Linear terms**: Individual stock returns
- **Quadratic terms**: Risk correlations between stocks
- **Penalty terms**: Ensures exactly k stocks are selected

**Output**: A QUBO problem object ready for quantum solving

---

### Section 4: Quantum Hamiltonian
**What it does**: Converts QUBO to quantum operators  
**Technical details**:
- Maps binary variables to qubits
- Represents objective function as Pauli Z operators
- Includes energy offset correction

**Output**: Hamiltonian in SparsePauliOp format

---

### Section 5: Classical Benchmark
**What it does**: Solves the problem exactly using classical methods  
**Purpose**: Establishes ground truth for comparison  
**Algorithm**: NumPyMinimumEigensolver (brute force)  
**Output**: Minimum achievable energy (optimal solution)

---

### Section 6: QAOA Optimization
**What it does**: Runs the quantum approximate optimization  
**Parameters**:
- Circuit depth (reps): 1 layer
- Max iterations: 100
- Optimizer: COBYLA

**Process**:
1. Initializes QAOA circuit
2. Optimizes variational parameters
3. Measures quantum state
4. Returns best bitstring solution

**Output**: Selected stocks as binary string (e.g., "111" = all 3 selected)

---

### Section 7: Share Allocation
**What it does**: Distributes your total shares across selected stocks  
**Algorithm**:
1. Calculate Sharpe-like ratios: `return / sqrt(variance)`
2. Normalize weights to sum to 1
3. Allocate shares proportionally
4. Round to whole numbers while respecting budget

**Outputs**:
- Shares per stock
- Cost per position
- Expected profit per stock
- Total investment breakdown

---

## 📊 Sample Output

### Example Execution

**User Input**:
```
Total Shares: 5
Budget per Share: $250.00
Total Budget: $1250.00
```

**QAOA Solution**:
```
Solution bitstring: 111

● MSFT: Selected
● TSLA: Selected  
● AAPL: Selected

Portfolio: ['MSFT', 'TSLA', 'AAPL']
Assets selected: 3
```

**Share Allocation**:
```
🎯 SHARE ALLOCATION RECOMMENDATION
======================================================================

Recommended Portfolio (5 shares):
Stock      Shares     Price/Share     Total Cost      Weight    
----------------------------------------------------------------------
MSFT       2          $459.86         $919.72         40.0%
TSLA       2          $437.50         $875.00         40.0%
AAPL       1          $255.53         $255.53         20.0%

Total Investment: $2,050.25
Budget: $1,250.00
⚠️ Over Budget: $800.25

Daily Expected Change: $0.35
Monthly Projection (21 days): $7.35
Quarterly Projection (63 days): $22.05
```

**Interpretation**:
- All 3 stocks were selected by QAOA
- MSFT and TSLA receive 2 shares each (higher Sharpe ratios)
- AAPL receives 1 share (lower relative performance)
- Portfolio is over budget, suggesting need to adjust parameters

---

## 🔧 Configuration & Experimentation

### Adjustable Parameters

| Parameter | Location | Default | What it controls |
|-----------|----------|---------|------------------|
| `lookback` | Cell: Market Data | `'6mo'` | Historical data period (`'3mo'`, `'1y'`, `'2y'`) |
| `total_shares` | Cell: User Input | User provided | Total shares to buy |
| `budget_per_share` | Cell: User Input | User provided | Maximum spend per share |
| `q` (risk factor) | Cell: QUBO | `0.5` | Risk aversion (0=max return, 1=min risk) |
| `k` (portfolio size) | Cell: QUBO | `len(assets)` | Number of stocks to select |
| `depth` | Cell: QAOA | `1` | QAOA circuit layers (1-3 recommended) |
| `maxiter` | Cell: QAOA | `100` | Optimization iterations |

### Experimentation Ideas

#### 1. **Change the stock universe**
Edit the `assets` list in Cell: Market Data
```python
assets = ['GOOGL', 'AMZN', 'META']  # Tech giants
# OR
assets = ['JPM', 'BAC', 'WFC']      # Banks
```

#### 2. **Adjust risk tolerance**
Modify `q` in QUBO cell:
```python
q = 0.2  # Aggressive (prioritize returns)
q = 0.8  # Conservative (prioritize safety)
q = 1.0  # Very risk-averse
```

#### 3. **Force different portfolio sizes**
Change `k` in QUBO cell:
```python
k = 2  # Select only 2 best stocks
k = 5  # Select from 5 stocks (add more to assets list)
```

#### 4. **Increase QAOA circuit depth**
Edit in QAOA cell:
```python
depth = 2  # More accurate but slower
depth = 3  # Even more accurate
```

#### 5. **Compare different time periods**
```python
lookback = '1mo'   # Very recent data
lookback = '2y'    # Long-term trends
```

### Understanding Quality Metrics

After running QAOA, compare:
- **QAOA Energy**: Energy found by quantum algorithm
- **Classical Energy**: Optimal energy (ground truth)
- **Quality**: `(Classical Energy / QAOA Energy) × 100%`

**Target**: 95%+ quality indicates QAOA found near-optimal solution

---

## 🏗️ Technical Architecture

### Project Structure

```
Quantum-Portfolio-Optimization-Demo/
├── main.ipynb                 # Interactive demonstration notebook
├── src/
│   ├── __init__.py           # Package initialization
│   ├── data_loader.py        # Market data fetching and processing
│   ├── portfolio_model.py    # QUBO formulation logic
│   └── quantum_solver.py     # QAOA and classical solvers
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

### Core Components

#### 1. **data_loader.py**
**Purpose**: Fetch and preprocess stock data  
**Key functions**:
- `fetch_market_data(tickers, period)`: Downloads historical prices
- Returns: `(expected_returns, covariance_matrix, price_dataframe)`

**Algorithm**:
```python
1. Download adjusted close prices via yfinance
2. Calculate daily percentage returns: (price_t - price_{t-1}) / price_{t-1}
3. Compute expected return: mean(returns)
4. Compute covariance matrix: cov(returns)
```

---

#### 2. **portfolio_model.py**
**Purpose**: Build QUBO mathematical model  
**Class**: `ManualPortfolioQUBO`  
**Key method**: `.build()` → Returns QuadraticProgram object

**QUBO Construction**:
```python
# Objective function terms:
Linear (single stocks):
  - return_i (negative because we minimize)

Quadratic (pairs of stocks):
  + risk_factor × covariance_{i,j}

Penalty (constraint):
  + P × (Σx_i - k)²  where P = max_return × penalty_multiplier
```

---

#### 3. **quantum_solver.py**
**Purpose**: Solve QUBO using QAOA and classical methods  
**Class**: `PortfolioSolver`  
**Key methods**:
- `.solve_classical()`: Exact solution (NumPyMinimumEigensolver)
- `.solve_qaoa(reps, maxiter)`: Quantum approximate solution

**QAOA Workflow**:
```python
1. Convert Hamiltonian to Qiskit format
2. Initialize QAOA ansatz circuit
3. Set up COBYLA optimizer
4. Run variational optimization loop
5. Sample final quantum state
6. Decode most probable bitstring
7. Return: (selected_stocks, bitstring)
```

---

### Mathematical Deep Dive

#### Mean-Variance Model (Markowitz)

The portfolio return and risk are:
```
Return: R_p = Σ w_i × μ_i
Risk:   σ_p² = w^T Σ w

where:
  w_i = weight of stock i
  μ_i = expected return of stock i
  Σ = covariance matrix
```

#### QUBO Mapping

For binary selection (not continuous weights):
```
Minimize:
  q × Σ Σ x_i x_j Σ_{ij}  [risk term]
  - Σ x_i μ_i              [return term]
  + P × (Σ x_i - k)²      [constraint]

Subject to: x_i ∈ {0, 1}
```

#### Ising Hamiltonian

QUBO → Ising mapping:
```
x_i = (1 - z_i) / 2  where z_i ∈ {-1, +1}

Hamiltonian:
H = Σ h_i Z_i + Σ J_{ij} Z_i Z_j + constant

where Z_i are Pauli-Z operators
```

---

## 📦 Dependencies

```
# Quantum Computing
qiskit>=1.0.0                    # IBM Quantum framework
qiskit-algorithms>=0.3.0         # QAOA implementation
qiskit-optimization>=0.6.0       # QUBO utilities

# Data & Analysis
numpy>=1.24.0                    # Numerical arrays
pandas>=2.0.0                    # Data structures
yfinance>=0.2.28                 # Stock data API

# Visualization
matplotlib>=3.7.0                # Plotting

# Optimization
scipy>=1.11.0                    # Scientific computing
```

**Install all at once**:
```bash
pip install -r requirements.txt
```

**Verify installation**:
```python
import qiskit
import yfinance as yf
print(f"Qiskit: {qiskit.__version__}")
print(f"yfinance: {yf.__version__}")
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. **Yahoo Finance Data Fetch Fails**
**Error**: `TypeError: 'NoneType' object is not subscriptable`

**Solution**: API rate limits or network issues
```python
# In the "Current Stock Prices" cell, the code now includes fallback:
try:
    hist = stock.history(period='2d')
    if hist is None or hist.empty:
        # Falls back to using already-fetched historical data
        current_price = price_data[ticker].iloc[-1]
except:
    current_price = price_data[ticker].iloc[-1]
```

**Prevention**: Increase `period` parameter or add delays between requests

---

#### 2. **KeyError when accessing risk_matrix**
**Error**: `KeyError: (0, 0)`

**Cause**: `risk_matrix` is a pandas DataFrame, not numpy array

**Solution**: Use `.values` accessor
```python
# Wrong:
variance = risk_matrix[i, i]

# Correct:
variance = risk_matrix.values[i, i]
# OR
variance = risk_matrix.iloc[i, i]
```

---

#### 3. **QAOA Takes Too Long**
**Issue**: QAOA optimization running for \u003e60 seconds

**Solutions**:
- Reduce circuit depth: `depth = 1` instead of `depth = 2`
- Lower max iterations: `maxiter = 50` instead of `100`
- Use fewer stocks: Select only 2-3 stocks instead of 5+
- Switch to classical solver for quick testing

---

#### 4. **Over Budget Warnings**
**Issue**: Allocated investment exceeds user budget

**Why**: Share prices don't divide evenly into budget  
**Solutions**:
- Increase `budget_per_share`
- Decrease `total_shares`
- Modify allocation logic to enforce strict budget (current code prioritizes optimal weights)

---

#### 5. **Import Errors**
**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Ensure virtual environment is activated
```bash
# Activate venv first
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac

# Reinstall if needed
pip install -r requirements.txt
```

---

#### 6. **Jupyter Kernel Crashes**
**Solution**: Increase memory or restart kernel
```python
# Restart kernel: Kernel → Restart \u0026 Clear Output
# Then run all cells again
```

---

## 🚀 Future Enhancements

### Planned Features

- [ ] **Multi-period optimization**: Rebalancing strategies over time
- [ ] **Transaction costs**: Include brokerage fees in optimization
- [ ] **Short selling**: Allow negative positions for hedging
- [ ] **Risk-free asset**: Add cash position for complete Markowitz model
- [ ] **Dynamic risk tolerance**: User-adjustable risk slider
- [ ] **Backtesting engine**: Historical performance simulation
- [ ] **Real quantum hardware**: IBM Quantum or AWS Braket integration
- [ ] **Portfolio rebalancing**: Optimal adjustments to existing positions
- [ ] **Web interface**: Flask/Streamlit dashboard
- [ ] **Database persistence**: Store results and historical allocations

### Research Directions

- [ ] VQE (Variational Quantum Eigensolver) comparison
- [ ] ADMM-based hybrid quantum-classical approach
- [ ] Quantum annealing on D-Wave hardware
- [ ] Graph neural networks for stock correlation
- [ ] Reinforcement learning for dynamic allocation

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Algorithm Enhancements**
   - Better penalty weight auto-tuning
   - Adaptive QAOA layer selection
   - Improved share rounding logic

2. **Data Sources**
   - Additional APIs (Alpha Vantage, Quandl)
   - Cryptocurrency support
   - International markets

3. **Visualization**
   - Efficient frontier plots
   - Correlation network graphs
   - Interactive Plotly charts

4. **Documentation**
   - Video tutorials
   - Example notebooks for different scenarios
   - Technical deep-dives

**How to contribute**:
1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Submit a pull request with description

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Qiskit Team** - IBM's quantum computing framework
- **Yahoo Finance** - Free market data API
- **Modern Portfolio Theory** - Harry Markowitz (1952)
- **QAOA Algorithm** - Farhi, Goldstone, Gutmann (2014)

---

## 📞 Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Email: [your-email@example.com]
- LinkedIn: [Your Profile]

---

## ⚠️ Disclaimer

**This is an educational and research project**. It demonstrates quantum computing applications in finance but is **NOT financial advice**. 

- Do not use this for actual investment decisions without consulting qualified financial professionals
- Past performance does not guarantee future returns
- Quantum algorithms provide approximations, not guaranteed optimal solutions
- Market data may have delays or inaccuracies

Always perform your own due diligence before making investment decisions.

---

**Built with ❤️ using Quantum Computing and Modern Portfolio Theory**
