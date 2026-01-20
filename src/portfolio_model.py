from qiskit_optimization import QuadraticProgram
import numpy as np

class ManualPortfolioQUBO:
    def __init__(self, gamma, sigma, budget=2, risk_factor=0.5, penalty_multiplier=10.0):
        """
        Manually constructs the Portfolio Optimization QUBO without using Qiskit converters.
        
        Args:
            gamma (numpy.ndarray): Expected Returns vector (Mean)
            sigma (numpy.ndarray): Covariance Matrix (Risk)
            budget (int): Number of assets to select
            risk_factor (float): 'q' value (Risk aversion parameter)
            penalty_multiplier (float): Strength of the constraint penalty
        """
        self.gamma = gamma
        self.sigma = sigma
        self.budget = budget
        self.q = risk_factor
        self.n_assets = len(gamma)
        
        # Initialize an empty Quadratic Program
        self.qp = QuadraticProgram("Manual_Portfolio_Optimization")
        
        # Calculate Penalty Strength (P)
        # P must be large enough to make invalid portfolios "expensive" in energy terms
        max_value = np.max(np.abs(gamma)) + np.max(np.abs(sigma))
        self.P = max_value * penalty_multiplier 
        
        self.linear_terms = {}
        self.quadratic_terms = {}
        
        print(f"[Model Builder] Initialized with Budget={budget}, Risk Factor={risk_factor}")

    def build(self):
        """Constructs the QUBO and returns the Qiskit QuadraticProgram object."""
        
        # 1. Register Binary Variables (x_0, x_1, ...)
        for i in range(self.n_assets):
            self.qp.binary_var(name=f"x_{i}")

        # 2. Add Objective Terms (Risk & Return)
        self._add_objective_terms()

        # 3. Add Constraint Penalty Terms (Budget)
        self._add_constraint_penalty()

        # 4. Finalize Model
        self.qp.minimize(linear=self.linear_terms, quadratic=self.quadratic_terms)
        print("[Model Builder] QUBO constructed successfully.")
        return self.qp

    def _add_objective_terms(self):
        """
        Minimize: q * (x.T * Sigma * x) - (gamma.T * x)
        """
        # Linear Part (-Returns)
        for i in range(self.n_assets):
            current_val = self.linear_terms.get(f"x_{i}", 0)
            # Subtract Return (gamma) because we minimize negative return
            self.linear_terms[f"x_{i}"] = current_val - self.gamma[i]

        # Quadratic Part (Risk)
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                # The term is q * sigma_ij * x_i * x_j
                risk_val = self.q * self.sigma[i][j]
                
                if i == j: 
                    # Diagonal (Variance): x_i^2 = x_i (for binary), so it adds to Linear
                    self.linear_terms[f"x_{i}"] += risk_val
                else: 
                    # Off-Diagonal (Covariance): Keeps as quadratic pair
                    # Store sorted tuple key to avoid duplicates (x_i, x_j) vs (x_j, x_i)
                    if (f"x_{j}", f"x_{i}") in self.quadratic_terms:
                        self.quadratic_terms[(f"x_{j}", f"x_{i}")] += risk_val
                    else:
                        self.quadratic_terms[(f"x_{i}", f"x_{j}")] = risk_val

    def _add_constraint_penalty(self):
        """
        Constraint: (Sum(x) - B)^2 = 0
        Adds P * (Sum(x) - B)^2 to the cost function.
        """
        # Linear Penalty contribution: P * (1 - 2B)
        linear_penalty = self.P * (1 - 2 * self.budget)
        for i in range(self.n_assets):
            self.linear_terms[f"x_{i}"] += linear_penalty

        # Quadratic Penalty contribution: 2P
        quadratic_penalty = 2 * self.P
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets): # Iterate unique pairs
                term = (f"x_{i}", f"x_{j}")
                current_q_val = self.quadratic_terms.get(term, 0)
                self.quadratic_terms[term] = current_q_val + quadratic_penalty