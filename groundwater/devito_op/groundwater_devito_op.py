# Adapted from https://github.com/devitocodes/devito/blob/master/examples/cfd/09_Darcy_flow_equation.ipynb

import numpy as np
from devito import (
    Eq,
    Function,
    Grid,
    Operator,
    TimeFunction,
    configuration,
    div,
    exp,
    grad,
    initialize_function,
    solve,
)


configuration["log-level"] = "ERROR"

NUM_PSEUDO_TIMESTEPS = 50000


class GroundwaterEquation:
    def __init__(self, size):
        self.size = size
        self.grid = Grid(
            shape=(size, size), extent=(1.0, 1.0), dtype=np.float32
        )

        self.p = TimeFunction(name="p", grid=self.grid, space_order=2)
        self.u = Function(name="u", grid=self.grid, space_order=2)
        self.f = Function(name="f", grid=self.grid, space_order=2)

        # Define adjoint variable
        self.lambda_adj = TimeFunction(
            name="lambda_adj",
            grid=self.grid,
            space_order=2,
        )
        self.f_adj = Function(name="f_adj", grid=self.grid, space_order=2)

        self.fwd_op = self.setup_foward_pde(self.p, self.u, self.f)
        self.adj_op = self.setup_adjoint_pde(
            self.lambda_adj,
            self.u,
            self.f_adj,
        )

        self.gradient = Function(name="gradient", grid=self.grid, space_order=2)

    def setup_foward_pde(self, p, u, f):
        x, y = self.grid.dimensions
        t = self.grid.stepping_dim

        # The PDE: -∇ · (e^u(x) ∇p(x)) = 0
        equation_p = Eq(-div(exp(u) * grad(p, shift=0.5), shift=-0.5), f)
        stencil = solve(equation_p, p)
        update = Eq(p.forward, stencil)

        # Boundary conditions:
        # p(x)|x2=0 = x1,
        # p(x)|x2=1 = 1 - x1,
        # ∂p(x)/∂x1|x1=0 = 0,
        # ∂p(x)/∂x1|x1=1 = 0.
        bc = [
            # p(x)|x2=0 = x1
            Eq(p[t + 1, x, 0], x * self.grid.spacing[0]),
            # p(x)|x2=1 = 1 - x1
            Eq(p[t + 1, x, y.symbolic_max], 1.0 - x * self.grid.spacing[0]),
            # ∂p(x)/∂x1|x1=0 = 0
            Eq(p[t + 1, -1, y], p[t + 1, 0, y]),
            # ∂p(x)/∂x1|x1=1 = 0
            Eq(p[t + 1, x.symbolic_max + 1, y], p[t + 1, x.symbolic_max, y]),
            #
            # Eq(p[t + 1, x, 0], 0),
            # Eq(p[t + 1, x, y.symbolic_max], 0),
            # Eq(p[t + 1, 0, y], 0),
            # Eq(p[t + 1, x.symbolic_max, y], 0),
        ]

        return Operator([update] + bc)

    def setup_adjoint_pde(self, lambda_adj, u, f_adj):
        x, y = self.grid.dimensions
        t = self.grid.stepping_dim

        # Adjoint equation: -∇ · (e^u ∇λ) = M*(Mp - d)
        adj_equation = Eq(
            -div(exp(u) * grad(lambda_adj, shift=-0.5), shift=0.5),
            f_adj,
        )
        stencil_adj = solve(adj_equation, lambda_adj)
        update_adj = Eq(lambda_adj.forward, stencil_adj)

        # Boundary conditions for adjoint problem:
        # λ(x)|x2=0 = 0,
        # λ(x)|x2=1 = 0,
        # ∂λ(x)/∂x1|x1=0 = ∂λ(x)/∂x1|x1=1.
        bc_adj = [
            # x1 = 0
            Eq(lambda_adj[t + 1, x, 0], 0),
            # x1 = 1
            Eq(lambda_adj[t + 1, x, y.symbolic_max], 0),
            # x2 = 0, transpose derivative = 0
            Eq(lambda_adj[t + 1, 0, y], lambda_adj[t + 1, 1, y]),
            # x2 = 1, transpose derivative = 0
            Eq(
                lambda_adj[t + 1, x.symbolic_max, y],
                lambda_adj[t + 1, x.symbolic_max - 1, y],
            ),
            #
            # Eq(p[t + 1, x, 0], 0),
            # Eq(p[t + 1, x, y.symbolic_max], 0),
            # Eq(p[t + 1, 0, y], 0),
            # Eq(p[t + 1, x.symbolic_max, y], 0),
        ]

        return Operator([update_adj] + bc_adj)

    def eval_fwd_op(
        self, f, u, time_steps=NUM_PSEUDO_TIMESTEPS, return_array=True
    ):
        self.f.data[:] = f[:]
        initialize_function(self.u, u, 0)
        self.p.data_with_halo.fill(0.0)
        self.fwd_op(time=time_steps)

        if return_array:
            return np.array(self.p.data[1])
        else:
            return self.p

    def eval_adj_op(
        self, u, residual, time_steps=NUM_PSEUDO_TIMESTEPS, return_array=True
    ):
        self.f_adj.data[:] = residual[:]
        initialize_function(self.u, u, 0)
        self.lambda_adj.data_with_halo.fill(0.0)
        self.adj_op(time=time_steps)

        if return_array:
            return np.array(self.lambda_adj.data[1])
        else:
            return self.lambda_adj

    def compute_gradient(self, u0, residual, p_fwd):
        # Compute the gradient: ∇_u J = e^u ∇λ · ∇p
        self.gradient.data.fill(0.0)

        lambda_adj = self.eval_adj_op(u0, residual, return_array=False)

        # e^u ∇λ · ∇p
        t = self.grid.stepping_dim
        grad_lambda = grad(lambda_adj, shift=0.5)._subs(t, 1)
        grad_p = grad(p_fwd, shift=0.5)._subs(t, 1)

        gradient_eq = Eq(
            self.gradient, -exp(self.u) * (grad_lambda.dot(grad_p))
        )
        op_gradient = Operator(gradient_eq)

        op_gradient()

        return self.gradient.data
