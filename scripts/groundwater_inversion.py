import numpy as np
import matplotlib.pyplot as plt

from groundwater.devito_op import GroundwaterEquation
from groundwater.utils import GaussianRandomField, plot_fields

# Set a random seed for reproducibility
np.random.seed(2)


# Example usage of gradient descent for groundwater inversion
if __name__ == "__main__":
    size: int = 32  # Grid size for the simulation
    num_iterations: int = 250  # Number of gradient descent iterations
    learning_rate: float = (
        40.0  # Learning rate for the gradient descent updates
    )

    # Generate a random true input field using a Gaussian random field
    u_true: np.ndarray = (
        GaussianRandomField(2, size, alpha=2, tau=4)
        .sample(1)[0]
        .astype(np.float32)
    )

    # Generate a smoother initial guess for the input field (u0) from a
    # different Gaussian random field
    u0: np.ndarray = (
        GaussianRandomField(2, size, alpha=4, tau=3)
        .sample(1)[0]
        .astype(np.float32)
    )
    u0_backup: np.ndarray = (
        u0.copy()
    )  # Backup of the initial guess for later comparison

    # Set the forcing term f(x) to be zero
    f: np.ndarray = np.zeros((size, size)).astype(np.float32)

    # Initialize the GroundwaterEquation instance for the simulation
    groundwater_eq = GroundwaterEquation(size)

    # Evaluate the forward operator for the true input field (u_true)
    p: np.ndarray = groundwater_eq.eval_fwd_op(f, u_true)

    # Evaluate the forward operator for the initial guess u0 and store a backup
    p_fwd_backup: np.ndarray = groundwater_eq.eval_fwd_op(
        f, u0, return_array=True
    ).copy()

    # Define a mask operator (M) to apply to the observed data (set as identity
    # in this example)
    mask: np.ndarray = np.ones((size, size)).astype(np.float32)

    # Apply the mask to the forward simulation output to get the observed data
    d_obs: np.ndarray = mask * p

    # Lists to store the residual and error norms at each iteration
    residual_norms: list[float] = []
    error_norms: list[float] = []

    # Perform gradient descent iterations
    for i in range(num_iterations):
        # Evaluate the forward operator for the current estimate of u0
        p_fwd = groundwater_eq.eval_fwd_op(f, u0, return_array=False)

        # Compute the residual between the forward model output and observed
        # data
        residual = mask.T * (mask * p_fwd.data[0] - d_obs)

        # Compute the gradient of the objective function with respect to u0
        u_grad = groundwater_eq.compute_gradient(u0, residual, p_fwd)

        # Update u0 using the gradient and learning rate
        u0 -= learning_rate * u_grad

        # Compute the norm of the residual (to track progress)
        residual_norm: float = np.linalg.norm(residual)

        # Compute the norm of the error (difference between u0 and the true
        # field u_true)
        error_norm: float = np.linalg.norm(u0 - u_true)

        # Log the norms for plotting later
        residual_norms.append(residual_norm)
        error_norms.append(error_norm)

        # Decrease the learning rate after every 100 iterations
        if (i + 1) % 100 == 0:
            learning_rate *= 0.9

        # Optionally, print progress at each iteration
        if (i + 1) % 1 == 0 or i == num_iterations - 1:
            print(
                f"Iteration {i + 1}/{num_iterations}, Residual norm: "
                f"{residual_norm}, Error: {error_norm}"
            )

    # Final evaluation of the forward operator with the optimized u0
    p_fwd_final = groundwater_eq.eval_fwd_op(f, u0, return_array=False)

    # Plot the true, initial, and final input fields u(x)
    plot_fields(
        [
            np.exp(_) for _ in [u_true, u0, u0_backup]
        ],  # Apply exponential for visualization purposes
        ["True u(x)", "Final u(x) after Gradient Descent", "Initial u(x)"],
        "Input Fields u(x)",
        contour=False,
    )

    # Plot the true, initial, and final output fields p(x) from the forward
    # operator
    plot_fields(
        [p, p_fwd_final.data[0], p_fwd_backup],
        ["True p(x)", "Final p(x) after Gradient Descent", "Initial p(x)"],
        "Output Fields p(x)",
        contour=False,
    )

    # Plot the residual norm over iterations
    plt.figure(figsize=(10, 5))
    plt.plot(residual_norms, label="Residual Norm", color="blue")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title("Residual Norm over Iterations")
    plt.legend()
    plt.grid(True)

    # Plot the error norm over iterations
    plt.figure(figsize=(10, 5))
    plt.plot(error_norms, label="Error Norm", color="green", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Error Norm")
    plt.title("Error Norm over Iterations")
    plt.legend()
    plt.grid(True)

    # Display the plots
    plt.show()
