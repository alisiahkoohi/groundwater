import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter

from groundwater.devito_op import GroundwaterEquation
from groundwater.utils import GaussianRandomField, plot_fields

matplotlib.use("TkAgg")
np.random.seed(2)

# Example usage.
if __name__ == "__main__":
    size = 32
    num_iterations = 250  # Number of gradient descent iterations
    learning_rate = 40.0  # Learning rate for gradient descent

    # Sample random fields for u(x).
    u_true = (
        GaussianRandomField(2, size, alpha=2, tau=4)
        .sample(1)[0]
        .astype(np.float32)
    )

    # Choose initial guess to be sample from a much smoother field.
    u0 = (
        GaussianRandomField(2, size, alpha=4, tau=3)
        .sample(1)[0]
        .astype(np.float32)
    )
    u0_backup = u0.copy()

    # Zero forcing term f(x).
    f = np.zeros((size, size)).astype(np.float32)

    # Setup Groundwater equation problem
    groundwater_eq = GroundwaterEquation(size)

    # Evaluate the forward operator for the true input field.
    p = groundwater_eq.eval_fwd_op(f, u_true)
    p_fwd_backup = groundwater_eq.eval_fwd_op(f, u0, return_array=True).copy()

    # Define the masking operator M (assuming it's identity for simplicity)
    mask = np.ones((size, size)).astype(np.float32)

    # Observed data with mask applied
    d_obs = mask * p

    # Lists to store residual norm and error norm
    residual_norms = []
    error_norms = []

    # Gradient descent loop
    for i in range(num_iterations):
        # Evaluate the forward operator for the current estimate u0.
        p_fwd = groundwater_eq.eval_fwd_op(f, u0, return_array=False)

        # Compute the residual
        residual = mask.T * (mask * p_fwd.data[0] - d_obs)

        # Compute the gradient with respect to u0.
        u_grad = groundwater_eq.compute_gradient(u0, residual, p_fwd)

        # Update u0 using the gradient and the learning rate.
        u0 -= learning_rate * u_grad

        # Compute norms
        residual_norm = np.linalg.norm(residual)
        error_norm = np.linalg.norm(u0 - u_true)

        # Log norms
        residual_norms.append(residual_norm)
        error_norms.append(error_norm)

        if (i + 1) % 100 == 0:
            learning_rate *= 0.9

        # Optionally, print progress
        if (i + 1) % 1 == 0 or i == num_iterations - 1:
            print(
                f"Iteration {i + 1}/{num_iterations}, Residual norm: {residual_norm}, Error: {error_norm}"
            )

    # Final evaluation of the forward operator after gradient descent.
    p_fwd_final = groundwater_eq.eval_fwd_op(f, u0, return_array=False)

    # Plot the final result after all iterations
    plot_fields(
        [np.exp(_) for _ in [u_true, u0, u0_backup]],
        ["True u(x)", "Final u(x) after Gradient Descent", "Initial u(x)"],
        "Input Fields u(x)",
        contour=False,
    )
    plot_fields(
        [p, p_fwd_final.data[0], p_fwd_backup],
        ["True p(x)", "Final p(x) after Gradient Descent", "Initial p(x)"],
        "Output Fields p(x)",
        contour=False,
    )

    # Plot residual norm
    plt.figure(figsize=(10, 5))
    plt.plot(residual_norms, label="Residual Norm", color="blue")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title("Residual Norm over Iterations")
    plt.legend()
    plt.grid(True)

    # Plot error norm
    plt.figure(figsize=(10, 5))
    plt.plot(error_norms, label="Error Norm", color="green", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Error Norm")
    plt.title("Error Norm over Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()
