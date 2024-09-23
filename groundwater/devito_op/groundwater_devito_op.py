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

# Set Devito logging level to ERROR to suppress excessive output
configuration["log-level"] = "ERROR"

# Number of timesteps for pseudo-time integration in the PDE solver
NUM_PSEUDO_TIMESTEPS = 50000


class GroundwaterEquation:
    """
    Class representing the groundwater flow equation based on Darcy's law.

    Attributes:
        size (int): Size of the grid.
        grid (Grid): Devito grid object representing the spatial domain.
        p (TimeFunction): Time-dependent pressure function on the grid.
        u (Function): Coefficient function representing permeability (log of
            permeability).
        f (Function): Source term function.
        lambda_adj (TimeFunction): Time-dependent adjoint variable function.
        f_adj (Function): Adjoint source term function.
        gradient (Function): Gradient function to store the computed gradient
            of the objective function.
        fwd_op (Operator): Forward PDE operator for time-stepping.
        adj_op (Operator): Adjoint PDE operator for time-stepping.
    """

    def __init__(self, size: int):
        """
        Initialize the GroundwaterEquation class.

        Args:
            size (int): The size of the grid.
        """
        self.size = size
        self.grid = Grid(
            shape=(size, size), extent=(1.0, 1.0), dtype=np.float32
        )

        # Time-dependent pressure variable p(x,t)
        self.p = TimeFunction(name="p", grid=self.grid, space_order=2)

        # Permeability (log-permeability) field u(x)
        self.u = Function(name="u", grid=self.grid, space_order=2)

        # Source term function f(x)
        self.f = Function(name="f", grid=self.grid, space_order=2)

        # Adjoint variable lambda(x,t) and adjoint source f_adj(x)
        self.lambda_adj = TimeFunction(
            name="lambda_adj",
            grid=self.grid,
            space_order=2,
        )
        self.f_adj = Function(name="f_adj", grid=self.grid, space_order=2)

        # Set up forward and adjoint PDE operators
        self.fwd_op = self.setup_foward_pde(self.p, self.u, self.f)
        self.adj_op = self.setup_adjoint_pde(
            self.lambda_adj,
            self.u,
            self.f_adj,
        )

        # Gradient function
        self.gradient = Function(name="gradient", grid=self.grid, space_order=2)

    def setup_foward_pde(
        self,
        p: TimeFunction,
        u: Function,
        f: Function,
    ) -> Operator:
        """
        Set up the forward PDE operator for the pressure equation.

        Args:
            p (TimeFunction): Time-dependent pressure function.
            u (Function): Log-permeability function.
            f (Function): Source term function.

        Returns:
            Operator: Devito operator for forward PDE time-stepping.
        """
        x, y = self.grid.dimensions
        t = self.grid.stepping_dim

        # The PDE: -∇ · (e^u(x) ∇p(x)) = f
        equation_p = Eq(-div(exp(u) * grad(p, shift=0.5), shift=-0.5), f)
        stencil = solve(equation_p, p)
        update = Eq(p.forward, stencil)

        # Boundary conditions
        bc = [
            # p(x)|x2=0 = x1
            Eq(p[t + 1, x, 0], x * self.grid.spacing[0]),
            # p(x)|x2=1 = 1 - x1
            Eq(p[t + 1, x, y.symbolic_max], 1.0 - x * self.grid.spacing[0]),
            # ∂p(x)/∂x1|x1=0 = 0
            Eq(p[t + 1, -1, y], p[t + 1, 0, y]),
            # ∂p(x)/∂x1|x1=1 = 0
            Eq(p[t + 1, x.symbolic_max + 1, y], p[t + 1, x.symbolic_max, y]),
        ]

        return Operator([update] + bc)

    def setup_adjoint_pde(
        self,
        lambda_adj: TimeFunction,
        u: Function,
        f_adj: Function,
    ) -> Operator:
        """
        Set up the adjoint PDE operator for the adjoint equation.

        Args:
            lambda_adj (TimeFunction): Time-dependent adjoint variable.
            u (Function): Log-permeability function.
            f_adj (Function): Adjoint source term function.

        Returns:
            Operator: Devito operator for adjoint PDE time-stepping.
        """
        x, y = self.grid.dimensions
        t = self.grid.stepping_dim

        # Adjoint equation: -∇ · (e^u(x) ∇λ(x)) = f_adj
        adj_equation = Eq(
            -div(exp(u) * grad(lambda_adj, shift=-0.5), shift=0.5),
            f_adj,
        )
        stencil_adj = solve(adj_equation, lambda_adj)
        update_adj = Eq(lambda_adj.forward, stencil_adj)

        # Boundary conditions for the adjoint equation
        bc_adj = [
            # λ(x)|x2=0 = 0,
            Eq(lambda_adj[t + 1, x, 0], 0),
            # λ(x)|x2=1 = 0,
            Eq(lambda_adj[t + 1, x, y.symbolic_max], 0),
            # ∂λ(x)/∂x1|x1=0 = ∂λ(x)/∂x1|x1=1.
            Eq(lambda_adj[t + 1, 0, y], lambda_adj[t + 1, 1, y]),
            Eq(
                lambda_adj[t + 1, x.symbolic_max, y],
                lambda_adj[t + 1, x.symbolic_max - 1, y],
            ),
        ]

        return Operator([update_adj] + bc_adj)

    def eval_fwd_op(
        self,
        f: np.ndarray,
        u: np.ndarray,
        time_steps: int = NUM_PSEUDO_TIMESTEPS,
        return_array: bool = True,
    ) -> np.ndarray:
        """
        Evaluate the forward PDE operator for a given source and permeability
            field.

        Args:
            f (np.ndarray): Source term array.
            u (np.ndarray): Log-permeability array.
            time_steps (int, optional): Number of pseudo-timesteps for
                integration. Defaults to NUM_PSEUDO_TIMESTEPS.
            return_array (bool, optional): Whether to return the pressure data
                as a NumPy array. Defaults to True.

        Returns:
            np.ndarray: Pressure field after time-stepping.
        """
        self.f.data[:] = f[:]
        initialize_function(self.u, u, 0)
        self.p.data_with_halo.fill(0.0)
        self.fwd_op(time=time_steps)

        if return_array:
            return np.array(self.p.data[1])
        else:
            return self.p

    def eval_adj_op(
        self,
        u: np.ndarray,
        residual: np.ndarray,
        time_steps: int = NUM_PSEUDO_TIMESTEPS,
        return_array: bool = True,
    ) -> np.ndarray:
        """
        Evaluate the adjoint PDE operator for a given residual and permeability
            field.

        Args:
            u (np.ndarray): Log-permeability array.
            residual (np.ndarray): Residual array (difference between measured
                and simulated data).
            time_steps (int, optional): Number of pseudo-timesteps for
                integration. Defaults to NUM_PSEUDO_TIMESTEPS.
            return_array (bool, optional): Whether to return the adjoint
                variable data as a NumPy array. Defaults to True.

        Returns:
            np.ndarray: Adjoint variable field after time-stepping.
        """
        self.f_adj.data[:] = residual[:]
        initialize_function(self.u, u, 0)
        self.lambda_adj.data_with_halo.fill(0.0)
        self.adj_op(time=time_steps)

        if return_array:
            return np.array(self.lambda_adj.data[1])
        else:
            return self.lambda_adj

    def compute_gradient(
        self, u0: np.ndarray, residual: np.ndarray, p_fwd: TimeFunction
    ) -> np.ndarray:
        """
        Compute the gradient of the objective function with respect to the
            permeability field:  ∇_u J = -e^u ∇λ · ∇p

        Args:
            u0 (np.ndarray): Initial guess for the log-permeability
                field.
            residual (np.ndarray): Residual array (difference between measured
                and simulated data).
            p_fwd (TimeFunction): Forward pressure field.

        Returns:
            np.ndarray: Gradient of the objective function with respect to the
                permeability field.
        """
        self.gradient.data_with_halo.fill(0.0)

        # Evaluate adjoint variable
        lambda_adj = self.eval_adj_op(u0, residual, return_array=False)

        # -e^u ∇λ · ∇p term for gradient computation
        t = self.grid.stepping_dim
        grad_lambda = grad(lambda_adj, shift=0.5)._subs(t, 1)
        grad_p = grad(p_fwd, shift=0.5)._subs(t, 1)

        # Gradient of the objective function with respect to u
        gradient_eq = Eq(
            self.gradient, -exp(self.u) * (grad_lambda.dot(grad_p))
        )
        op_gradient = Operator(gradient_eq)

        # Compute the gradient
        op_gradient()

        return self.gradient.data
