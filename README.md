# Safe Bayesian Optimization

Project developed for the **Probabilistic Artificial Intelligence** (HS24) course at **ETH Zurich**.

Implementation of **constrained Bayesian Optimization** for maximizing drug-likeness (logP) subject to a synthetic accessibility (SA) safety constraint.

## Overview

In drug discovery, we want to find molecules that maximize a target property (logP) while keeping synthetic accessibility below a safety threshold. Both the objective and constraint are expensive black-box functions observed with noise. This implementation uses two independent Gaussian Processes and a penalized acquisition function to efficiently explore the search space while virtually guaranteeing constraint satisfaction.

## Approach

- **Dual GP model**: separate Gaussian Processes for the objective f(x) and constraint v(x), each with a composite kernel (RationalQuadratic × Matérn + WhiteKernel)
- **Penalized acquisition function**: `acq(x) = μ_f(x) + β·σ_f(x) − λ·max(μ_v(x) − threshold + β_v·σ_v(x), 0)` where λ=1000 provides a strong safety guarantee
- **Optimistic constraint handling**: the β_v·σ_v term ensures the algorithm is cautious about uncertain constraint regions
- **Non-trivial solution selection**: `get_optimal_solution` penalizes points near the initial safe point to encourage meaningful exploration

## Files

- `solution.py` — Complete BO loop: GP fitting, acquisition function optimization (L-BFGS-B with 20 restarts), observation management, and safe solution selection

## Key Design Decisions

| Parameter | Value | Role |
|-----------|-------|------|
| λ (lambda) | 1000 | Constraint penalty — keeps unsafe evaluations near zero |
| β | 1000 | Exploration bonus for f — encourages visiting high-uncertainty regions |
| β_v | 70 | Constraint uncertainty weight — conservative near the safety boundary |
