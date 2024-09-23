import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from groundwater.devito_op import GroundwaterEquation
from groundwater.utils import GaussianRandomField


def objective(
    u: np.ndarray,
    groundwater_eq: GroundwaterEquation,
    f: np.ndarray,
    d_obs: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Define the objective function, which evaluates the norm of the difference
    between the forward simulation and the observed data, and calculates the
    gradient.

    Parameters:
        u (np.ndarray): The input field, representing some physical property.
        groundwater_eq (GroundwaterEquation): The GroundwaterEquation object to
            handle forward and adjoint operations.
        f (np.ndarray): Forcing term f(x), affecting the physical process.
        d_obs (np.ndarray): Observed data from the forward simulation to compare
            against.

    Returns:
        tuple[float, np.ndarray]: The objective function value (scalar) and the
        gradient (array).
    """
    # Run the forward model using the current input field u
    p_fwd = groundwater_eq.eval_fwd_op(f, u, return_array=False)

    # Compute the residual (difference between forward model output and observed
    # data)
    residual = p_fwd.data[0] - d_obs

    # Compute the objective function value using L2 norm of the residual
    f_val = 0.5 * np.linalg.norm(residual) ** 2

    # Compute the gradient of the objective function with respect to u
    gradient = groundwater_eq.compute_gradient(u, residual, p_fwd)

    return f_val, gradient


def gradient_test(
    groundwater_eq: GroundwaterEquation,
    u0: np.ndarray,
    f: np.ndarray,
    d_obs: np.ndarray,
    dx: np.ndarray,
    epsilon: float = 1e-2,
    maxiter: int = 10,
) -> tuple[list[float], list[float]]:
    """
    Perform a gradient test using a Taylor expansion to verify the accuracy of
    the computed gradient. This is done by perturbing the input u0 in the
    direction of dx and checking if the errors decrease as expected.

    Parameters:
        groundwater_eq (GroundwaterEquation): Instance of the
            GroundwaterEquation class.
        u0 (np.ndarray): Initial guess for the input field u(x).
        f (np.ndarray): Forcing term f(x). d_obs (np.ndarray): Observed data
            from the forward simulation.
        dx (np.ndarray): Perturbation direction for testing gradient accuracy.
        epsilon (float, optional): Initial step size for the perturbation.
            Default is 1e-2.
        maxiter (int, optional): Maximum number of iterations for halving the
            step size. Default is 10.

    Returns:
        tuple[list[float], list[float]]: Lists of first-order and second-order
            errors.
    """
    errors_first_order = []
    errors_second_order = []

    rate_first_order = []
    rate_second_order = []

    # Compute the objective function value and gradient for the initial guess u0
    f0, g = objective(u0, groundwater_eq, f, d_obs)

    h = epsilon
    for j in range(maxiter):
        # Perturb u0 in the direction of dx by h
        f_perturbed, _ = objective(u0 + h * dx, groundwater_eq, f, d_obs)

        # First-order Taylor error: |f(u0 + h*dx) - f(u0)|
        err1 = np.abs(f_perturbed - f0)

        # Second-order Taylor error: |f(u0 + h*dx) - f(u0) - h * <g, dx>|
        err2 = np.abs(
            f_perturbed - f0 - h * np.dot(dx.reshape(-1), g.reshape(-1))
        )

        # Print step size and errors for each iteration
        print(
            f"Step {j + 1}: Step size = {h:.5e}, First-order error "
            f"= {err1:.5e}, gdx = "
            f"{h * np.dot(dx.reshape(-1), g.reshape(-1)):.5e}"
        )

        # Append errors for analysis
        errors_first_order.append(err1)
        errors_second_order.append(err2)
        rate_first_order.append(
            errors_first_order[j] / errors_first_order[max(0, j - 1)]
        )
        rate_second_order.append(
            errors_second_order[j] / errors_second_order[max(0, j - 1)]
        )

        print(
            f"Step {j + 1}: Step size = {h:.5e}, First-order error = {err1:.5e}"
            f", Second-order error = {err2:.5e} First-order rate = "
            f"{rate_first_order[j]:.5e}, Second-order rate = "
            f"{rate_second_order[j]:.5e}"
        )

        # Halve the step size for the next iteration
        h *= 0.5

    return errors_first_order, errors_second_order


# Example usage:
if __name__ == "__main__":
    size = 128
    epsilon = 5e-1

    # Generate the true input field using a Gaussian random field
    u_true = GaussianRandomField(2, size, alpha=2, tau=4).sample(2)[0]

    # Create a smoothed initial guess by applying a Gaussian filter to the true
    # field
    u0 = gaussian_filter(u_true, sigma=3)

    # Forcing term (e.g., external influences) is initialized as zeros
    f = np.zeros((size, size))

    # Initialize the Groundwater equation problem
    groundwater_eq = GroundwaterEquation(size)

    # Simulate the forward operator with the true input field
    p_true = groundwater_eq.eval_fwd_op(f, u_true)
    p_smooth = groundwater_eq.eval_fwd_op(f, u0)

    # Adjoint test to verify correctness of adjoint operations
    p_fwd = groundwater_eq.eval_fwd_op(p_smooth, u0)
    lambda_adj = groundwater_eq.eval_adj_op(u0, p_smooth)

    # Verify adjoint property: <A^-1 x, x> = <A^-T y, y>
    term1 = np.dot(p_smooth.reshape(-1), lambda_adj.reshape(-1))
    term2 = np.dot(p_fwd.reshape(-1), p_smooth.reshape(-1))
    # print(f"Adjoint test: {term1} = {term2}, ratio = {term1/term2}")

    # Use forward simulation with u_true as the "observed" data
    d_obs = p_true

    # Set the perturbation direction (difference between true field and smoothed
    # guess)
    dx = u_true - u0

    # Perform a gradient test
    errors_first_order, errors_second_order = gradient_test(
        groundwater_eq, u0, f, d_obs, dx, epsilon=epsilon, maxiter=5
    )

    # Estimate the expected error decay for first and second order
    h = [errors_first_order[0] * 0.5**i for i in range(5)]
    h2 = [errors_second_order[0] * 0.5 ** (2 * i) for i in range(5)]

    # Plot the errors (log-log scale) for visual comparison
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
