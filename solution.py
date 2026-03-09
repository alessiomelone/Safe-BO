"""
Safe Bayesian Optimization
===========================
Constrained Bayesian Optimization for drug-likeness (logP) maximization
subject to a synthetic accessibility (SA) safety constraint.

We use two independent Gaussian Processes to model the objective f(x) and the
constraint v(x), combined through a penalized acquisition function that
balances exploration of f with safety enforcement on v.

Key design choices:
- Separate GP kernels: RationalQuadratic * Matern + WhiteKernel for both f and v
- Penalized acquisition function instead of standard EI, with large constraint
  penalty (lambda=1000) to virtually guarantee safety during exploration
- Optimistic constraint handling via beta_v * sigma_v term
- Custom get_optimal_solution that penalizes trivial (initial point) solutions
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

DOMAIN = np.array([[0.0, 10.0]])
SAFETY_THRESHOLD = 4.0
PRIOR_MEAN_V = 4.0


class BOAlgorithm():
    def __init__(self):
        """Initialize GPs for objective f and constraint v with tuned composite kernels."""
        kernel_f = (
            0.5 * Product(
                RationalQuadratic(length_scale=10.0, length_scale_bounds=(0.5, 10), alpha=1.5, alpha_bounds=(0.5, 10)),
                Matern(length_scale=1.0, length_scale_bounds=(0.5, 10), nu=2.5)
            )
            + 2.0 * WhiteKernel(noise_level=0.15**2, noise_level_bounds=(0.15**2 * 0.5, 0.15**2 * 2))
        )
        kernel_v = (
            2.0 * Product(
                RationalQuadratic(length_scale=1.0, length_scale_bounds=(0.5, 10), alpha=1.5, alpha_bounds=(0.5, 10)),
                0.5 * Matern(length_scale=0.1, length_scale_bounds=(0.5, 10), nu=2.5)
            )
            + WhiteKernel(noise_level=0.0001**2, noise_level_bounds=(0.0001**2 * 0.5, 0.0001**2 * 2))
        )
        self.gp_f = GaussianProcessRegressor(kernel=kernel_f)
        self.gp_v = GaussianProcessRegressor(kernel=kernel_v)
        self.X = np.array([])
        self.y_f = np.array([])
        self.y_v = np.array([])


    def recommend_next(self):
        """Recommend the next input to sample by optimizing the acquisition function."""
        return self.optimize_acquisition_function()

    def optimize_acquisition_function(self):
        """Maximize the acquisition function via L-BFGS-B with 20 random restarts."""

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt
    
    def acquisition_function(self, x: np.ndarray):
        """
        Penalized acquisition function that balances objective exploration with
        safety constraint satisfaction.

        acq(x) = mu_f(x) + beta * sigma_f(x) - lambda * max(mu_v(x) - threshold + beta_v * sigma_v(x), 0)

        The large lambda ensures near-zero unsafe evaluations, beta encourages
        exploration in f, and beta_v provides optimistic constraint handling.
        """
        x = np.atleast_2d(x)

        mu_v, sigma_v = self.gp_v.predict(x, return_std=True)
        mu_v += PRIOR_MEAN_V
        mu_f, sigma_f = self.gp_f.predict(x, return_std=True)

        lambda_h = 1000.0    # constraint penalty weight
        beta_h = 1000.0      # exploration bonus weight for f
        beta_v_h = 70.0      # uncertainty weight for constraint

        acquisition = mu_f - lambda_h * np.maximum(mu_v - SAFETY_THRESHOLD + beta_v_h * sigma_v, 0) + beta_h * sigma_f
        return acquisition


    def add_observation(self, x: float, f: float, v: float):
        """Add a new (x, f(x), v(x)) observation and refit both GPs."""
        self.X = np.append(self.X, x)
        self.y_f = np.append(self.y_f, f)
        self.y_v = np.append(self.y_v, v - PRIOR_MEAN_V)

        self.gp_f.fit(self.X.reshape(-1, 1), self.y_f)
        self.gp_v.fit(self.X.reshape(-1, 1), self.y_v)

    def get_optimal_solution(self):
        """
        Return the best observed safe point, with a penalty for points too
        close to the initial safe point (to avoid trivial solutions).
        Scores each safe point by: 0.6 - 0.4 * regret - 0.15 * (near_initial).
        """
        mask = self.y_v + PRIOR_MEAN_V <= SAFETY_THRESHOLD
        safe_X = self.X[mask]
        safe_f = self.y_f[mask]

        x_initial = self.X[0]

        if len(safe_f) == 0:
            return x_initial

        f_best = np.max(safe_f)
        eval_values = []

        for x, f_val in zip(safe_X, safe_f):
            regret = max(f_best - f_val, 0) / f_best
            score = 0.6 - 0.4 * regret
            if abs(x - x_initial) < 0.1:
                score -= 0.15
            eval_values.append(score)

        eval_values = np.array(eval_values)
        idx = np.argmax(eval_values)
        return safe_X[idx]

    def plot(self, plot_recommendation: bool = True):
        """Placeholder for GP visualization (used during development)."""
        pass


# --- Toy problem for local testing (not called by checker) ---

def check_in_domain(x: float):
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return -np.linalg.norm(x - mid_point, 2)


def v(x: float):
    return 2.0


def get_initial_safe_point():
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    return x_valid[0]


def main():
    agent = BOAlgorithm()

    x_init = get_initial_safe_point()
    agent.add_observation(x_init, f(x_init), v(x_init))

    for _ in range(20):
        x = agent.recommend_next()
        agent.add_observation(x, f(x) + np.random.randn(), v(x) + np.random.randn())

    solution = agent.get_optimal_solution()
    assert check_in_domain(solution)
    print(f"Proposed solution: {solution}, f(x)={f(solution)}, regret={-f(solution)}")


if __name__ == "__main__":
    main()
