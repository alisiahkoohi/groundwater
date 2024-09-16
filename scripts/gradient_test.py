import matplotlib.pyplot as plt
import numpy as np
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

        errors_first_order.append(err1)
        errors_second_order.append(err2)

        print(
            f"Step {j + 1}: Step size = {h:.5e}, First-order error = {err1:.5e}, Second-order error = {err2:.5e}"
        )

        # Halve the step size
        h *= 0.5

    return errors_first_order, errors_second_order


# Example usage:
if __name__ == "__main__":
    size = 40

    # Randomly sample the true input field and initial guess
    u_true = GaussianRandomField(2, size, alpha=3, tau=3).sample(1)[0]

    # Smooth initial guess by smoothing the true field using a Gaussian filter
    from scipy.ndimage import gaussian_filter

    u0 = gaussian_filter(u_true, sigma=3)

    # Forcing term f(x) (zero for simplicity)
    f = np.zeros((size, size))

    # Setup the Groundwater equation problem
    groundwater_eq = GroundwaterEquation(size)

    # Evaluate the forward operator with the true input field
    p_true = groundwater_eq.eval_fwd_op(f, u_true)

    # Use the forward simulation with u_true as the "observed" data
    d_obs = p_true

    # Set perturbation direction dx (difference between u and u0)
    dx = u_true - u0

    # Perform gradient test
    errors_first_order, errors_second_order = gradient_test(
        groundwater_eq, u0, f, d_obs, dx
    )

    # Plot the errors (log-log scale)
    plt.loglog(errors_first_order, label="First-order error")
    plt.loglog(errors_second_order, label="Second-order error")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Gradient Test: Error vs Step Size")
    plt.grid(True)
    plt.show()
