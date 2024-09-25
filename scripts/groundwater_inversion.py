import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from groundwater.utils import (
    GaussianRandomField,
    plot_fields,
)
from groundwater.devito_op import GroundwaterModel

# Set a random seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)


matplotlib.use("TkAgg")


# Example usage of PyTorch for groundwater inversion
if __name__ == "__main__":
    size: int = 32  # Grid size for the simulation
    num_iterations: int = 250  # Number of gradient descent iterations
    learning_rate: float = (
        15.0  # Learning rate for the gradient descent updates
    )

    # Generate a random true input field using a Gaussian random field
    u_true: np.ndarray = (
        GaussianRandomField(2, size, alpha=2, tau=4)
        .sample(1)[0]
        .astype(np.float32)
    )
    u_true_tensor = torch.from_numpy(u_true)

    # Generate a smoother initial guess for the input field (u0) from a different Gaussian random field
    u0: np.ndarray = (
        GaussianRandomField(2, size, alpha=4, tau=3)
        .sample(1)[0]
        .astype(np.float32)
    )
    u0_tensor = torch.from_numpy(u0).requires_grad_(True)
    u0_backup = u0.copy()  # Backup of the initial guess for later comparison

    # Set the forcing term f(x) to be zero
    f: np.ndarray = np.zeros((size, size)).astype(np.float32)
    f_tensor = torch.from_numpy(f)

    # Define a mask operator (M) to apply to the observed data (set as identity in this example)
    mask: np.ndarray = np.ones((size, size)).astype(np.float32)
    mask_tensor = torch.from_numpy(mask)

    # Initialize the GroundwaterEquation instance and the PyTorch model
    model = GroundwaterModel(size)

    # Evaluate the forward operator for the true input field (u_true)
    with torch.no_grad():
        p_tensor = model(u_true_tensor, f_tensor)
        p = p_tensor.numpy()

    # Evaluate the forward operator for the initial guess u0 and store a backup
    with torch.no_grad():
        p_fwd_backup_tensor = model(torch.from_numpy(u0_backup), f_tensor)
        d_pred_backup = p_fwd_backup_tensor * mask_tensor
        d_pred_backup = d_pred_backup.numpy()

    # Apply the mask to the forward simulation output to get the observed data
    d_obs_tensor = mask_tensor * p_tensor
    d_obs = d_obs_tensor.numpy()

    # Lists to store the residual and error norms at each iteration
    residual_norms: list[float] = []
    error_norms: list[float] = []

    # Set up the optimizer
    optimizer = torch.optim.SGD([u0_tensor], lr=learning_rate)

    # Perform optimization iterations
    for i in range(num_iterations):
        optimizer.zero_grad()

        # Evaluate the forward operator for the current estimate of u0
        p_fwd_tensor = model(u0_tensor, f_tensor)

        # Compute the residual between the forward model output and observed data
        residual_tensor = mask_tensor * p_fwd_tensor - d_obs_tensor

        # Compute the loss (mean squared error of the residual)
        loss = torch.norm(residual_tensor.view(-1)) ** 2

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Compute the norms
        with torch.no_grad():
            residual_norm: float = torch.norm(residual_tensor.view(-1)).item()
            error_norm: float = torch.norm(
                u0_tensor.view(-1) - u_true_tensor.view(-1)
            ).item()

        # Log the norms for plotting later
        residual_norms.append(residual_norm)
        error_norms.append(error_norm)

        # Decrease the learning rate after every 100 iterations
        if (i + 1) % 100 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.9

        # Optionally, print progress at each iteration
        if (i + 1) % 1 == 0 or i == num_iterations - 1:
            print(
                f"Iteration {i + 1}/{num_iterations}, Residual norm: {residual_norm}, Error: {error_norm}"
            )

    # Final evaluation of the forward operator with the optimized u0
    with torch.no_grad():
        p_fwd_final_tensor = model(u0_tensor, f_tensor)
        d_pred_final = mask_tensor * p_fwd_final_tensor
        d_pred_final = d_pred_final.numpy()
        u0_final = u0_tensor.numpy()

    # Plot the true, initial, and final input fields u(x)
    plot_fields(
        [
            np.exp(_) for _ in [u_true, u0_final, u0_backup]
        ],  # Apply exponential for visualization purposes
        ["True u(x)", "Final u(x) after Optimization", "Initial u(x)"],
        "Input Fields u(x)",
        contour=False,
    )

    # Plot the true, initial, and final output fields p(x) from the forward operator
    plot_fields(
        [d_obs, d_pred_final, d_pred_backup],
        ["True p(x)", "Final p(x) after Optimization", "Initial p(x)"],
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
