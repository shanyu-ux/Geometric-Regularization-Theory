"""
NCGD: Non-Commutative Geometric Dynamics
Module: Python Prototype / Riemannian Adversary

* This module implements the "Jacobian Oracle" and "Riemannian Gradient Flow" 
* using PyTorch functional primitives. 
* It serves as the numerical verification for the C++ core engine.

* Theoretical Basis:
* Update Rule: x_{t+1} = Exp_{x_t}( alpha * G(x)^{-1} * grad_{Euc} )
"""

import torch
import torch.nn as nn
import torch.linalg as la

class JacobianOracle:
    """
    Implements the Oracle Query for the local Pullback Metric Tensor G(x).
    
    Mathematical Definition:
        G(x) = J_f(x)^T * J_f(x)
        where J_f is the Jacobian of the mapping f: M -> N.
        
    Complexity:
        O(d^3) due to matrix multiplication and subsequent linear system solution.
    """
    @staticmethod
    def compute_metric(model, x):
        # 1. Query the full Jacobian matrix
        # Note: This is computationally expensive but necessary for geometric rigor.
        J = torch.autograd.functional.jacobian(model, x)
        
        # Reshape for batch processing [Batch, Out_Dim, In_Dim]
        if J.ndim > 3: 
            J = J.view(x.shape[0], -1, x.shape[1])
            
        # 2. Compute Pullback Metric G = J^T * J
        G = torch.bmm(J.transpose(1, 2), J)
        
        # 3. Regularization (Tikhonov damping) to ensure invertibility
        G += 1e-6 * torch.eye(G.shape[-1], device=x.device)
        return G

class RiemannianAdversary:
    """
    Implements the Riemannian Projected Gradient Ascent (RPGA).
    
    Role:
        Finds the worst-case perturbation x* that maximizes the local distortion
        measure, respecting the intrinsic geometry induced by the network.
    """
    def __init__(self, model, epsilon=0.1, alpha=0.01, num_steps=10):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps

    def perturb(self, x_seed):
        """
        Executes the adversarial attack loop using the Riemannian correction.
        """
        x_adv = x_seed.clone().detach().requires_grad_(True)
        
        for t in range(self.num_steps):
            # --- Step 1: Compute Euclidean Gradient (The Covector) ---
            # We aim to maximize the spectral energy (distortion)
            J_current = torch.autograd.functional.jacobian(self.model, x_adv)
            if J_current.ndim > 3: 
                J_current = J_current.view(x_adv.shape[0], -1, x_adv.shape[1])
            
            # Objective: Maximize operator norm (approx. via Frobenius or Spectral)
            # Here using Frobenius for differentiability stability in prototype
            obj = torch.linalg.norm(J_current, ord='fro', dim=(1,2)).sum()
            
            grad_euc = torch.autograd.grad(obj, x_adv)[0]
            
            # --- Step 2: The Riemannian Correction (G^{-1}) ---
            # Transform covector (dL) to tangent vector (grad L) via Inverse Metric.
            G = JacobianOracle.compute_metric(self.model, x_adv)
            
            # Solve linear system G * v = g_euc (equivalent to v = G^-1 * g_euc)
            grad_riem = torch.linalg.solve(G, grad_euc.unsqueeze(-1)).squeeze(-1)
            
            # --- Step 3: Geodesic Update (Exponential Map Approximation) ---
            # x_{t+1} = x_t + alpha * v_t
            x_adv.data = x_adv.data + self.alpha * grad_riem
            
            # --- Step 4: Projection (Ambient Constraint) ---
            delta = x_adv.data - x_seed.data
            delta = delta.renorm(p=2, dim=0, maxnorm=self.epsilon)
            x_adv.data = x_seed.data + delta
            
        return x_adv.detach()
