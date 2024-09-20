import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from groundwater.devito_op import GroundwaterEquation
from groundwater.utils import GaussianRandomField


def objective(u, groundwater_eq, f, d_obs):
    """
    Define the objective function as the norm of the difference between
    the forward simulated solution and the observed data.

    Parameters:
    - u: Input field.
    - groundwater_eq: Instance of the GroundwaterEquation class.
    - f: Forcing term f(x).
    - d_obs: Observed data from the forward simulation.

    Returns:
    - objective value and gradient with respect to u.
    """
    # Run the forward model
    p_fwd = groundwater_eq.eval_fwd_op(f, u, return_array=False)

    # Compute residual (forward simulation - observed data)
    residual = p_fwd.data[0] - d_obs

    # Objective function (e.g., L2 norm of the residual)
    f_val = 0.5 * np.linalg.norm(residual) ** 2

    # Compute the gradient of the objective with respect to u
    gradient = groundwater_eq.compute_gradient(u, residual, p_fwd)

    return f_val, gradient


def gradient_test(groundwater_eq, u0, f, d_obs, dx, epsilon=1e-2, maxiter=10):
    """
    Perform a gradient test using Taylor expansion by perturbing u0 in
    the direction of dx and checking if the error decreases as expected.

    Parameters:
    - groundwater_eq: Instance of the GroundwaterEquation class.
    - u0: Initial guess for the input field u(x).
    - f: Forcing term f(x).
    - d_obs: Observed data d(x) (from the forward simulation).
    - dx: Perturbation direction.
    - epsilon: Initial step size.
    - maxiter: Number of iterations to reduce the perturbation.

    Returns:
    - errors: List of first-order and second-order errors at each step.
    """
    errors_first_order = []
    errors_second_order = []

    rate_first_order = []
    rate_second_order = []

    # Compute the objective value and gradient for the initial guess u0
    f0, g = objective(u0, groundwater_eq, f, d_obs)

    h = epsilon
    for j in range(maxiter):
        # Perturb u0 in the direction of dx
        f_perturbed, _ = objective(u0 + h * dx, groundwater_eq, f, d_obs)

        # First-order Taylor error: |f(u0 + h*dx) - f(u0)|
        err1 = np.abs(f_perturbed - f0)

        # Second-order Taylor error: |f(u0 + h*dx) - f(u0) - h * <g, dx>|
        err2 = np.abs(
            f_perturbed - f0 - h * np.dot(dx.reshape(-1), g.reshape(-1))
        )
        print(
            f"Step {j + 1}: Step size = {h:.5e}, First-order error = {err1:.5e}, gdx = {h*np.dot(dx.reshape(-1), g.reshape(-1)):.5e}"
        )

        errors_first_order.append(err1)
        errors_second_order.append(err2)
        rate_first_order.append(
            errors_first_order[j] / errors_first_order[max(0, j - 1)]
        )
        rate_second_order.append(
            errors_second_order[j] / errors_second_order[max(0, j - 1)]
        )

        print(
            f"Step {j + 1}: Step size = {h:.5e}, First-order error = {err1:.5e}, Second-order error = {err2:.5e} First-order rate = {rate_first_order[j]:.5e}, Second-order rate = {rate_second_order[j]:.5e}"
        )

        # Halve the step size
        h *= 0.5

    return errors_first_order, errors_second_order


# Example usage:
if __name__ == "__main__":
    size = 128
    epsilon = 5e-1

    # Randomly sample the true input field and initial guess
    u_true = GaussianRandomField(2, size, alpha=2, tau=4).sample(2)[0]

    # Smooth initial guess by smoothing the true field using a Gaussian filter
    u0 = gaussian_filter(u_true, sigma=3)

    # Forcing term f(x).
    f = np.zeros((size, size))

    # Setup the Groundwater equation problem
    groundwater_eq = GroundwaterEquation(size)

    # Evaluate the forward operator with the true input field
    p_true = groundwater_eq.eval_fwd_op(f, u_true)
    p_smooth = groundwater_eq.eval_fwd_op(f, u0)

    # Adjoint test
    # y1 = A^(-1) x
    p_fwd = groundwater_eq.eval_fwd_op(p_smooth, u0)
    # x1 = A^(-T) y
    lambda_adj = groundwater_eq.eval_adj_op(u0, p_smooth)

    # x1 . x
    term1 = np.dot(p_smooth.reshape(-1), lambda_adj.reshape(-1))
    # y1 . y
    term2 = np.dot(p_fwd.reshape(-1), p_smooth.reshape(-1))
    print(f"Adjoint test: {term1} = {term2}, ratio = {term1/term2}")

    # Use the forward simulation with u_true as the "observed" data
    d_obs = p_true

    # Set perturbation direction dx (difference between u and u0)
    dx = u_true - u0

    # Perform gradient test
    errors_first_order, errors_second_order = gradient_test(
        groundwater_eq, u0, f, d_obs, dx, epsilon=epsilon, maxiter=5
    )

    # Make start at error
    h = [errors_first_order[0] * 0.5**i for i in range(5)]
    h2 = [errors_second_order[0] * 0.5 ** (2 * i) for i in range(5)]

    # Plot the errors (log-log scale)
    plt.semilogy(errors_first_order, label="First-order error", base=2)
    plt.semilogy(errors_second_order, label="Second-order error", base=2)
    plt.semilogy(h, label="h^1", base=2)
    plt.semilogy(h2, label="h^2", base=2)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Gradient Test: Error vs Step Size")
    plt.grid(True)
    plt.show()
