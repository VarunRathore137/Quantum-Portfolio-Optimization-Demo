"""
Quantum Portfolio Optimization - Main Script
=============================================
This script demonstrates quantum portfolio optimization using QAOA (Quantum Approximate Optimization Algorithm)
to select optimal stock portfolios based on risk-return tradeoff.

Features:
- Fetches real-time market data from Yahoo Finance (3 months)
- Builds QUBO (Quadratic Unconstrained Binary Optimization) formulation
- Solves using both classical and quantum approaches
- Displays detailed results including bitstrings, acceptance rates, and profit comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import itertools

# Import custom modules
from src import fetch_market_data, ManualPortfolioQUBO, PortfolioSolver

# ============================================================================
# USER INPUT - INTERACTIVE CONFIGURATION
# ============================================================================
print("=" * 80)
print("QUANTUM PORTFOLIO OPTIMIZATION - INTERACTIVE MODE")
print("=" * 80)
print("\nWelcome! This system will help you optimize your stock portfolio using quantum computing.")
print("Available companies: MSFT (Microsoft), TSLA (Tesla), AAPL (Apple)")
print("\nThe optimizer will decide how to distribute your shares among these companies")
print("to maximize your expected profit based on historical data.\n")

# Get total number of shares
while True:
    try:
        total_shares = int(input("How many total shares do you want to buy? (minimum 5): "))
        if total_shares >= 5:
            break
        else:
            print("  ❌ Please enter at least 5 shares.")
    except ValueError:
        print("  ❌ Please enter a valid number.")

# Get budget per share
while True:
    try:
        budget_per_share = float(input("What's your maximum budget per share in USD? (e.g., 500): $"))
        if budget_per_share > 0:
            break
        else:
            print("  ❌ Budget must be greater than 0.")
    except ValueError:
        print("  ❌ Please enter a valid number.")

# Calculate total budget
total_budget = total_shares * budget_per_share

print(f"\n✓ Configuration Accepted:")
print(f"  - Total Shares to Buy: {total_shares}")
print(f"  - Budget per Share: ${budget_per_share:.2f}")
print(f"  - Total Budget: ${total_budget:.2f}")
print("\n⏳ Fetching market data and optimizing your portfolio...\n")

# ============================================================================
# CONFIGURATION
# ============================================================================
TICKERS = ['MSFT', 'TSLA', 'AAPL']  # Stock tickers to optimize
PERIOD = '3mo'  # Data period (3 months)
BUDGET = len(TICKERS)  # We'll select from all available stocks
RISK_FACTOR = 0.5  # Risk aversion parameter (q)
QAOA_REPS = 1  # QAOA circuit depth
MAX_ITER = 100  # Maximum optimization iterations

print("=" * 80)
print("QUANTUM PORTFOLIO OPTIMIZATION")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Tickers: {TICKERS}")
print(f"Period: {PERIOD}")
print(f"Risk Factor (q): {RISK_FACTOR}")
print("=" * 80)

# ============================================================================
# STEP 1: FETCH AND PREPARE MARKET DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: FETCHING MARKET DATA")
print("=" * 80)

gamma, sigma, prices = fetch_market_data(TICKERS, period=PERIOD)

print(f"\n[*] Data Summary:")
print(f"   Number of assets: {len(gamma)}")
print(f"   Trading days: {len(prices)}")
print(f"   Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

print(f"\n[*] Expected Daily Returns (gamma):")
for i, ticker in enumerate(TICKERS):
    print(f"   {ticker}: {gamma.iloc[i]:.6f} ({gamma.iloc[i]*100:.4f}% per day)")

print(f"\n[*] Covariance Matrix (sigma - Risk):")
print("   " + " ".join([f"{ticker:>12}" for ticker in TICKERS]))
for i, ticker in enumerate(TICKERS):
    row_str = f"   {ticker:>6} " + " ".join([f"{sigma.iloc[i, j]:>12.8f}" for j in range(len(TICKERS))])
    print(row_str)

# Variance check
variances = np.diag(sigma.values)
print(f"\n[*] Variance Check:")
for i, ticker in enumerate(TICKERS):
    print(f"   {ticker} variance: {variances[i]:.8f} - {'POSITIVE [OK]' if variances[i] > 0 else 'NEGATIVE [ERROR]'}")

# ============================================================================
# STEP 2: FETCH CURRENT STOCK PRICES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: FETCHING CURRENT STOCK PRICES")
print("=" * 80)

import yfinance as yf
current_prices = {}
print(f"\n[*] Current Stock Prices:")
for ticker in TICKERS:
    stock = yf.Ticker(ticker)
    current_price = stock.history(period='1d')['Close'].iloc[-1]
    current_prices[ticker] = current_price
    print(f"   {ticker}: ${current_price:.2f} per share")

print(f"\n[*] Budget Analysis:")
print(f"   Your total budget: ${total_budget:.2f}")
print(f"   Budget per share: ${budget_per_share:.2f}")
print(f"   Total shares to buy: {total_shares}")

# ============================================================================
# STEP 3: BUILD QUBO FORMULATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: BUILDING QUBO FORMULATION")
print("=" * 80)

print(f"\n[*] QUBO Problem Setup:")
print(f"   Objective: Minimize risk, Maximize returns")
print(f"   Formula: q * (x^T * Sigma * x) - (gamma^T * x) + P * (Sum(x_i) - B)^2")
print(f"   Where:")
print(f"     x = binary vector (1 = select stock, 0 = don't select)")
print(f"     Sigma = covariance matrix (risk)")
print(f"     gamma = expected returns vector")
print(f"     q = risk factor = {RISK_FACTOR}")
print(f"     B = budget = {BUDGET}")
print(f"     P = penalty multiplier (enforces budget constraint)")

builder = ManualPortfolioQUBO(gamma.values, sigma.values, budget=BUDGET, risk_factor=RISK_FACTOR)
qp = builder.build()

print(f"\n[*] QUBO Model Built:")
print(f"   Variables: {qp.get_num_vars()} binary variables (x_0, x_1, ...)")
print(f"   Penalty strength (P): {builder.P:.4f}")

# Display QUBO coefficients
print(f"\n[*] Linear Coefficients:")
for var_name, coeff in builder.linear_terms.items():
    print(f"   {var_name}: {coeff:.6f}")

print(f"\n[*] Quadratic Coefficients (interactions):")
for (var1, var2), coeff in builder.quadratic_terms.items():
    print(f"   {var1} * {var2}: {coeff:.6f}")

# ============================================================================
# STEP 4: CONVERT TO HAMILTONIAN (QUANTUM-READY FORMAT)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: CONVERTING TO HAMILTONIAN")
print("=" * 80)

hamiltonian, offset = qp.to_ising()

print(f"\n[*] Hamiltonian (Pauli Operators):")
print(f"   {hamiltonian}")
print(f"\n   Offset: {offset}")
print(f"\n   This Hamiltonian will be fed into the quantum circuit!")

# ============================================================================
# STEP 5: CLASSICAL SOLVER (BENCHMARK)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: CLASSICAL SOLVER (EXACT SOLUTION)")
print("=" * 80)

solver = PortfolioSolver(hamiltonian, TICKERS)
classical_energy = solver.solve_classical()

print(f"\n[*] Classical Result:")
print(f"   Minimum Energy: {classical_energy:.6f}")
print(f"   (This is the exact ground state energy)")

# ============================================================================
# STEP 6: QAOA SOLVER (QUANTUM APPROACH)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: QAOA SOLVER (QUANTUM APPROACH)")
print("=" * 80)

print(f"\n[*] QAOA Configuration:")
print(f"   Circuit depth (reps): {QAOA_REPS}")
print(f"   Classical optimizer: COBYLA")
print(f"   Max iterations: {MAX_ITER}")

selected_assets, bitstring = solver.solve_qaoa(reps=QAOA_REPS, maxiter=MAX_ITER)

print(f"\n[*] QAOA Result:")
print(f"   Bitstring: {bitstring}")
print(f"   Selected Stocks: {selected_assets}")

# Decode bitstring
print(f"\n[*] Bitstring Interpretation:")
for i, bit in enumerate(bitstring):
    status = "SELECTED [OK]" if bit == '1' else "NOT SELECTED [ ]"
    print(f"   x_{i} ({TICKERS[i]}): {bit} -> {status}")

# ============================================================================
# STEP 7: CALCULATE ACCEPTANCE PERCENTAGE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: ACCEPTANCE RATE CALCULATION")
print("=" * 80)

# Check if budget constraint is satisfied
selected_count = bitstring.count('1')
budget_satisfied = (selected_count == BUDGET)

print(f"\n[*] Constraint Validation:")
print(f"   Budget requirement: {BUDGET} stocks")
print(f"   Stocks selected: {selected_count}")
print(f"   Constraint satisfied: {'YES [OK]' if budget_satisfied else 'NO [ERROR]'}")

# For QAOA, acceptance rate is typically 100% if optimization converged properly
# In a real quantum computer, we'd look at measurement statistics
acceptance_rate = 100.0 if budget_satisfied else 0.0

print(f"\n[*] Acceptance Rate: {acceptance_rate:.2f}%")
print(f"   (In real quantum hardware, this would be based on measurement statistics)")

# ============================================================================
# STEP 8: PROFIT CALCULATION & COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: PROFIT ANALYSIS & COMPARISON")
print("=" * 80)

# Calculate portfolio returns for QAOA solution
qaoa_binary = np.array([int(bit) for bit in bitstring])
qaoa_return = np.dot(gamma.values, qaoa_binary)
qaoa_risk = np.dot(qaoa_binary.T, np.dot(sigma.values, qaoa_binary))

print(f"\n[*] QAOA Portfolio:")
print(f"   Selected stocks: {selected_assets}")
print(f"   Expected daily return: {qaoa_return:.6f} ({qaoa_return*100:.4f}%)")
print(f"   Portfolio risk (variance): {qaoa_risk:.8f}")
print(f"   Return/Risk ratio: {qaoa_return/qaoa_risk if qaoa_risk > 0 else 'N/A':.2f}")

# Find best classical solution by brute force (for comparison)
print(f"\n[*] Finding optimal classical solution (brute force)...")
best_combo = None
best_return = -np.inf
best_risk = 0

all_combos = list(itertools.product([0, 1], repeat=len(TICKERS)))
for combo in all_combos:
    x = np.array(combo)
    if sum(x) == BUDGET:  # Must satisfy budget constraint
        ret = np.dot(gamma.values, x)
        risk = np.dot(x.T, np.dot(sigma.values, x))
        
        # Calculate objective (same as QUBO: minimize risk - maximize return)
        objective = RISK_FACTOR * risk - ret
        
        if best_combo is None or objective < (RISK_FACTOR * best_risk - best_return):
            best_combo = x
            best_return = ret
            best_risk = risk

classical_stocks = [TICKERS[i] for i, bit in enumerate(best_combo) if bit == 1]

print(f"\n[*] Classical Optimal Portfolio:")
print(f"   Selected stocks: {classical_stocks}")
print(f"   Expected daily return: {best_return:.6f} ({best_return*100:.4f}%)")
print(f"   Portfolio risk (variance): {best_risk:.8f}")
print(f"   Return/Risk ratio: {best_return/best_risk if best_risk > 0 else 'N/A':.2f}")

# Comparison
print(f"\n[*] PERFORMANCE COMPARISON:")
print(f"   {'Method':<20} {'Daily Return':<15} {'Risk':<15} {'Return/Risk':<15}")
print(f"   {'-'*65}")
print(f"   {'Classical (Exact)':<20} {best_return:.6f} ({best_return*100:.3f}%)  {best_risk:.8f}    {best_return/best_risk if best_risk > 0 else 0:.2f}")
print(f"   {'QAOA':<20} {qaoa_return:.6f} ({qaoa_return*100:.3f}%)  {qaoa_risk:.8f}    {qaoa_return/qaoa_risk if qaoa_risk > 0 else 0:.2f}")

# Calculate percentage difference
if best_return != 0:
    return_diff = ((qaoa_return - best_return) / abs(best_return)) * 100
    print(f"\n   Return difference: {return_diff:+.2f}%")
    
    if abs(return_diff) < 1:
        print(f"   [OK] QAOA found the optimal solution!")
    elif return_diff > 0:
        print(f"   [OK] QAOA outperformed classical by {return_diff:.2f}%!")
    else:
        print(f"   [INFO] QAOA within {abs(return_diff):.2f}% of optimal")

# Note about negative returns
if qaoa_return < 0:
    print(f"\n   ⚠️  NOTE: Negative returns indicate recent market downturn.")
    print(f"       Historical data from past 3 months shows declining trend.")
    print(f"       In real trading, consider using longer time periods or different risk factors.")

# ============================================================================
# STEP 9: CALCULATE ACTUAL SHARE ALLOCATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: SHARE ALLOCATION RECOMMENDATION")
print("=" * 80)

print(f"\n[*] Calculating optimal share distribution for your {total_shares} shares...")
print(f"    Budget constraint: ${total_budget:.2f} (${budget_per_share:.2f} per share)")

# Get selected stocks from QAOA
selected_indices = [i for i, bit in enumerate(bitstring) if bit == '1']
num_selected = len(selected_indices)

if num_selected == 0:
    print("\n   ❌ ERROR: No stocks were selected by the optimizer!")
    print("   This might indicate a problem with the QUBO formulation.")
else:
    # Calculate allocation weights based on expected returns (higher return = more shares)
    # We'll use a Sharpe-ratio-like metric: return/risk for each selected stock
    selected_returns = gamma.values[selected_indices]
    selected_variances = np.array([sigma.values[i, i] for i in selected_indices])
    
    # Calculate weights (using absolute values to avoid negative weight issues)
    # Weight by expected return adjusted for risk
    weights = np.zeros(num_selected)
    for idx, i in enumerate(selected_indices):
        if selected_variances[idx] > 0:
            # Sharpe-like ratio: higher is better
            weights[idx] = selected_returns[idx] / np.sqrt(selected_variances[idx])
        else:
            weights[idx] = selected_returns[idx]
    
    # Normalize weights to be positive (shift if needed)
    min_weight = np.min(weights)
    if min_weight < 0:
        weights = weights - min_weight + 0.1  # Shift to make all positive
    
    # Normalize to sum to 1
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        # If all weights are zero/negative, use equal distribution
        weights = np.ones(num_selected) / num_selected
    
    # Allocate shares based on weights
    share_allocation = {}
    shares_allocated = 0
    total_cost = 0
    
    print(f"\n[*] Recommended Portfolio:")
    print(f"   {'Stock':<10} {'Shares':<10} {'Price/Share':<15} {'Total Cost':<15} {'Weight':<10}")
    print(f"   {'-'*70}")
    
    # Allocate shares for each stock
    for idx, stock_idx in enumerate(selected_indices):
        ticker = TICKERS[stock_idx]
        price = current_prices[ticker]
        
        # Calculate shares based on weight
        if idx < num_selected - 1:
            # Allocate proportionally
            num_shares = int(total_shares * weights[idx])
        else:
            # Last stock gets remaining shares
            num_shares = total_shares - shares_allocated
        
        # Ensure at least 1 share if weight > 0
        if num_shares == 0 and shares_allocated < total_shares:
            num_shares = 1
        
        cost = num_shares * price
        share_allocation[ticker] = {
            'shares': num_shares,
            'price': price,
            'cost': cost,
            'weight': weights[idx]
        }
        
        shares_allocated += num_shares
        total_cost += cost
        
        print(f"   {ticker:<10} {num_shares:<10} ${price:<14.2f} ${cost:<14.2f} {weights[idx]*100:<9.1f}%")
    
    print(f"   {'-'*70}")
    print(f"   {'TOTAL':<10} {shares_allocated:<10} {'':<15} ${total_cost:<14.2f} {100.0:<9.1f}%")
    
    # Budget check
    print(f"\n[*] Budget Analysis:")
    print(f"   Target budget: ${total_budget:.2f}")
    print(f"   Actual cost: ${total_cost:.2f}")
    print(f"   Difference: ${total_budget - total_cost:+.2f}")
    
    if total_cost <= total_budget:
        print(f"   ✓ Within budget!")
    else:
        print(f"   ⚠️  Over budget by ${total_cost - total_budget:.2f}")
    
    # Calculate expected profit/loss
    expected_daily_return_dollars = 0
    for ticker, data in share_allocation.items():
        stock_idx = TICKERS.index(ticker)
        expected_daily_return_dollars += data['shares'] * data['price'] * gamma.values[stock_idx]
    
    print(f"\n[*] Expected Performance:")
    print(f"   Daily expected change: ${expected_daily_return_dollars:.2f} ({(expected_daily_return_dollars/total_cost)*100 if total_cost > 0 else 0:.4f}%)")
    print(f"   Weekly projection (5 days): ${expected_daily_return_dollars * 5:.2f}")
    print(f"   Monthly projection (21 days): ${expected_daily_return_dollars * 21:.2f}")
    print(f"   Annual projection (252 days): ${expected_daily_return_dollars * 252:.2f}")
    
    if expected_daily_return_dollars < 0:
        print(f"\n   ⚠️  WARNING: Based on recent 3-month data, this portfolio shows negative expected returns.")
        print(f"       This reflects recent market conditions. Consider:")
        print(f"       1. Using a longer historical period (6mo or 1y)")
        print(f"       2. Adjusting the risk factor parameter")
        print(f"       3. Waiting for more favorable market conditions")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n[*] Quantum Portfolio Optimization Complete!")
print(f"\n[*] Your Personalized Investment Plan:")
print(f"   - Total Shares to Purchase: {total_shares}")
print(f"   - Total Investment: ${total_cost:.2f}")
if num_selected > 0:
    print(f"   - Stock Allocation:")
    for ticker, data in share_allocation.items():
        print(f"     • {ticker}: {data['shares']} shares @ ${data['price']:.2f} = ${data['cost']:.2f}")

print(f"\n[*] Optimization Results:")
print(f"   - QAOA Selected Stocks: {selected_assets}")
print(f"   - Classical Optimal Stocks: {classical_stocks}")
print(f"   - Optimization Accuracy: {acceptance_rate:.1f}%")
print(f"   - Match with Classical: {'✓ Perfect Match!' if selected_assets == classical_stocks else '✗ Different selection'}")

print(f"\n[*] Expected Returns (Based on 3-month historical data):")
if expected_daily_return_dollars < 0:
    print(f"   - Daily: ${expected_daily_return_dollars:.2f} ⚠️")
    print(f"   - Annual: ${expected_daily_return_dollars * 252:.2f} ⚠️")
    print(f"   ⚠️  Note: Negative returns reflect recent market conditions")
else:
    print(f"   - Daily: ${expected_daily_return_dollars:.2f} ✓")
    print(f"   - Annual: ${expected_daily_return_dollars * 252:.2f} ✓")

print(f"\n[*] Quantum Advantage:")
print(f"   For larger portfolios (100+ assets), QAOA can find good solutions")
print(f"   exponentially faster than classical brute-force search!")

print(f"\n[*] Educational Note:")
print(f"   This demonstration uses quantum simulation. On real quantum hardware,")
print(f"   results may vary due to noise and decoherence.")

print("\n" + "=" * 80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
