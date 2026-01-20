from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from scipy.optimize import minimize
from qiskit_algorithms import NumPyMinimumEigensolver
import numpy as np

class PortfolioSolver:
    def __init__(self, hamiltonian, tickers):
        self.hamiltonian = hamiltonian
        self.tickers = tickers # List of strings e.g. ['MSFT', 'TSLA']

    def solve_classical(self):
        """Runs NumPy exact solver to find the ground truth."""
        print("\n[Solver] Running Classical Benchmark...")
        exact_solver = NumPyMinimumEigensolver()
        result = exact_solver.compute_minimum_eigenvalue(self.hamiltonian)
        energy = result.eigenvalue.real
        print(f" > Classical Minimum Energy: {energy:.4f}")
        return energy

    def solve_qaoa(self, reps=1, maxiter=100):
        """
        Runs Manual QAOA using Qiskit Primitives V2.
        
        Args:
            reps (int): Depth of QAOA circuit (p).
            maxiter (int): Maximum optimization steps for COBYLA.
        """
        print(f"\n[Solver] Running QAOA (reps={reps})...")
        
        # 1. Build Circuit
        ansatz = QAOAAnsatz(cost_operator=self.hamiltonian, reps=reps, name="Portfolio_QAOA")
        pub_circuit = ansatz.decompose()
        
        # 2. Define Cost Function
        estimator = StatevectorEstimator()

        def cost_func(params):
            # Pass (Circuit, Hamiltonian, Params) to V2 Estimator
            pub = (pub_circuit, self.hamiltonian, params)
            job = estimator.run([pub])
            return job.result()[0].data.evs

        # 3. Optimize Angles
        print(" > Optimizing variational parameters...")
        # Start with random parameters
        x0 = 2 * np.pi * np.random.rand(ansatz.num_parameters)
        res = minimize(cost_func, x0, method='COBYLA', options={'maxiter': maxiter})
        print(f" > Optimal Energy Found: {res.fun:.4f}")
        
        # 4. Measure Final Circuit
        print(" > Measuring optimal circuit...")
        optimal_circuit = pub_circuit.assign_parameters(res.x)
        optimal_circuit.measure_all()
        
        sampler = StatevectorSampler()
        job = sampler.run([optimal_circuit])
        counts = job.result()[0].data.meas.get_counts()
        
        # 5. Decode Result
        best_bitstring = max(counts, key=counts.get)
        
        # Reverse bitstring (Qiskit is Little-Endian)
        reversed_bitstring = best_bitstring[::-1]
        
        selected_assets = []
        for i, bit in enumerate(reversed_bitstring):
            if bit == '1' and i < len(self.tickers):
                selected_assets.append(self.tickers[i])
                
        return selected_assets, reversed_bitstring