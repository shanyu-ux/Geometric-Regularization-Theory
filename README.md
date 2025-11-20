# Non-Commutative Geometric Dynamics (NCGD)

> **A C++ framework for structure-preserving optimization on Lie Groups.**

[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Standard](https://img.shields.io/badge/C%2B%2B-17%2F20-blue)]()
[![Math](https://img.shields.io/badge/Math-Lie_Theory-purple)]()

## Abstract

Standard deep learning optimization algorithms (e.g., SGD, Adam) operate on the assumption of a flat Euclidean parameter space. While effective for unstructured data, this assumption violates the geometric constraints of physical systems governed by **Lie Group symmetries** (e.g., Rigid Body Dynamics in SE(3)).

**NCGD** is a computational library designed to enforce **Geometric Rigidity** and **Conformality**. Instead of Euclidean vector addition, it implements exact **Lie Algebraic operations** (Baker-Campbell-Hausdorff formula, Exponential Maps, and Lie Brackets) to ensure that optimization trajectories remain on the valid data manifold.

## Mathematical Formulation

### 1. Non-Commutativity and the Lie Bracket
Unlike vector spaces, operations on the Special Euclidean Group $SE(3)$ are non-commutative. NCGD explicitly computes the **Lie Bracket** to model the curvature of the state space:

$$
[\xi_1, \xi_2] = \text{ad}_{\xi_1}(\xi_2) \in \mathfrak{se}(3)
$$

This formulation allows for the precise modeling of coupled dynamics (e.g., rotational-translational coupling) that linear approximations fail to capture.

### 2. Riemannian Adversarial Regularization
To bound the generalization error on curved manifolds, we implement a **Jacobian Oracle** that computes the local **Pullback Metric Tensor** $G(x)$:

$$
G(x) = J_f(x)^\top J_f(x)
$$

The adversarial perturbation $\delta$ is then computed via the Riemannian gradient flow:

$$
\delta_{t+1} = \text{Exp}_{x_t} \left( \alpha \cdot G(x_t)^{-1} \nabla_{Euc} \mathcal{L} \right)
$$

This explicitly penalizes the Lipschitz constant of the network with respect to the intrinsic geometry of the data.

## Computational Architecture

The library is structured into two components:

* **`src/core/` (C++)**: A header-only, dependency-free implementation of Lie Algebra operations.
    * `se3_algebra.hpp`: Exact implementation of $\mathfrak{se}(3)$ operations.
    * `hessian_probe.cpp`: Lanczos-based estimation of spectral curvature.
* **`python_prototype/` (Python)**: Proof-of-concept implementation of Riemannian Adversarial Dynamics using PyTorch functional primitives for verification.

## Applications

This framework is intended for domains requiring strict adherence to physical conservation laws and geometric constraints:

* **Robotics:** Kinematic chain optimization with guaranteed rigidity.
* **Molecular Dynamics:** Energy landscape exploration respecting $SE(3)$ equivariance.
* **Astrophysics:** Inference on non-Euclidean manifolds (e.g., gravitational lensing).

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{yu2025ncgd,
  title={Non-Commutative Geometric Dynamics: A Framework for Rigid Body Mechanics and Riemannian Optimization},
  author={Yu, Shan},
  year={2025},
  publisher={GitHub},
  howpublished={\url{[https://github.com/](https://github.com/)[YOUR_USERNAME]/NCGD}},
}
