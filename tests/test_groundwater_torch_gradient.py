import matplotlib
import matplotlib.pyplot as plt
import torch
from groundwater.devito_op import GroundwaterModel
from groundwater.utils import GaussianRandomField
from scipy.ndimage import gaussian_filter

matplotlib.use("TkAgg")


def objective_torch(
    u: torch.Tensor,
    groundwater_model: GroundwaterModel,
    f: torch.Tensor,
    d_obs: torch.Tensor,
) -> torch.Tensor:
    """
    Define the objective function, which evaluates the norm of the difference
    between the forward simulation and the observed data.

    Parameters:
        u (torch.Tensor): The input field, representing some physical property.
        groundwater_model (GroundwaterModel): The model wrapped with the custom PDE solver layer.
        f (torch.Tensor): Forcing term f(x), affecting the physical process.
        d_obs (torch.Tensor): Observed data from the forward simulation to compare against.

    Returns:
        torch.Tensor: The objective function value (scalar).
    """
    # Run the forward model using the current input field u
    p_fwd = groundwater_model(u, f)

    # Compute the residual (difference between forward model output and observed data)
    residual = p_fwd - d_obs

    # Compute the objective function value using L2 norm of the residual
    f_val = 0.5 * torch.norm(residual) ** 2

    return f_val


def gradient_test_torch(
    groundwater_model: GroundwaterModel,
    u0: torch.Tensor,
    f: torch.Tensor,
    d_obs: torch.Tensor,
    dx: torch.Tensor,
    epsilon: float = 1e-2,
    maxiter: int = 10,
) -> tuple[list[float], list[float]]:
    """
    Perform a gradient test using a Taylor expansion to verify the accuracy of
    the computed gradient using PyTorch's automatic differentiation.

    Parameters:
        groundwater_model (GroundwaterModel): Instance of the GroundwaterModel class.
        u0 (torch.Tensor): Initial guess for the input field u(x).
        f (torch.Tensor): Forcing term f(x).
        d_obs (torch.Tensor): Observed data from the forward simulation.
        dx (torch.Tensor): Perturbation direction for testing gradient accuracy.
        epsilon (float, optional): Initial step size for the perturbation.
            Default is 1e-2.
        maxiter (int, optional): Maximum number of iterations for halving the
            step size. Default is 10.

    Returns:
        tuple[list[float], list[float]]: Lists of first-order and second-order errors.
    """
    errors_first_order = []
    errors_second_order = []

    rate_first_order = []
    rate_second_order = []

    # Ensure gradients are tracked for u0
    u0 = u0.clone().detach().requires_grad_(True)

    # Compute the objective function value for the initial guess u0
    f0 = objective_torch(u0, groundwater_model, f, d_obs)

    # Compute the gradient with respect to u0
    f0.backward()  # This computes the gradient of f0 wrt u0, stored in u0.grad
    g = u0.grad.clone()

    h = epsilon
    for j in range(maxiter):
        # Perturb u0 in the direction of dx by h
        u_perturbed = u0 + h * dx

        # Compute the perturbed objective function
        f_perturbed = objective_torch(u_perturbed, groundwater_model, f, d_obs)

        # First-order Taylor error: |f(u0 + h*dx) - f(u0)|
        err1 = torch.abs(f_perturbed - f0).item()

        # Second-order Taylor error: |f(u0 + h*dx) - f(u0) - h * <g, dx>|
        err2 = torch.abs(
            f_perturbed - f0 - h * torch.dot(dx.view(-1), g.view(-1))
        ).item()

        # Print step size and errors for each iteration
        print(
            f"Step {j + 1}: Step size = {h:.5e}, First-order error "
            f"= {err1:.5e}, gdx = "
            f"{h * torch.dot(dx.view(-1), g.view(-1)).item():.5e}"
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
            f", Second-order error = {err2:.5e}, First-order rate = "
            f"{rate_first_order[j]:.5e}, Second-order rate = "
            f"{rate_second_order[j]:.5e}"
        )

        # Halve the step size for the next iteration
        h *= 0.5

    return errors_first_order, errors_second_order


# Example usage
if __name__ == "__main__":
    size = 128
    epsilon = 5e-1

    # Generate the true input field using a Gaussian random field
    u_true = torch.tensor(
        GaussianRandomField(2, size, alpha=2, tau=4).sample(2)[0],
        dtype=torch.float32,
    )

    # Create a smoothed initial guess by applying a Gaussian filter to the true field
    u0 = torch.tensor(gaussian_filter(u_true, sigma=3), dtype=torch.float32)

    # Forcing term (e.g., external influences) is initialized as zeros
    f = torch.zeros((size, size), dtype=torch.float32)

    # Initialize the Groundwater equation problem wrapped in the PyTorch model
    groundwater_model = GroundwaterModel(size)

    # Simulate the forward operator with the true input field
    p_true = groundwater_model(u_true, f)

    # Use forward simulation with u_true as the "observed" data
    d_obs = p_true.detach()

    # Set the perturbation direction (difference between true field and smoothed guess)
    dx = u_true - u0

    # Perform a gradient test using PyTorch
    errors_first_order, errors_second_order = gradient_test_torch(
        groundwater_model, u0, f, d_obs, dx, epsilon=epsilon, maxiter=5
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
