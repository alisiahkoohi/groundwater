import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from groundwater.utils import GaussianRandomField, plot_fields
from groundwater.devito_op import GroundwaterModel

matplotlib.use("TkAgg")

# Number of pseudo-timesteps to simulate the groundwater equation. This should
# be increased for larger input sizes to ensure the forward operator converges.
NUM_PSEUDO_TIMESTEPS: int = 100000


def simulate_groundwater_eq(size: int = 256, num_samples: int = 3) -> None:
    """
    Simulate the groundwater equation using random input fields sampled from a
    Gaussian Random Field (GRF) and evaluate the forward operator for each
    input.

    Parameters:
        size: The size of the grid for the input fields (u(x)).
        num_samples: The number of random input field samples to generate.
    """

    # Step 1: Sample random input fields from a Gaussian Random Field (GRF)
    u_samples: list[np.ndarray] = GaussianRandomField(
        2,
        size,
        alpha=2,
        tau=4,
    ).sample(num_samples)

    # Step 2: Set up the zero forcing term f(x)
    f: torch.Tensor = torch.zeros((size, size), dtype=torch.float32)

    # Step 3: Initialize the GroundwaterModel instance
    groundwater_model = GroundwaterModel(size)

    # Step 4: Evaluate the forward operator for each input field u(x) using the
    # groundwater model
    with torch.no_grad():
        p: list[torch.Tensor] = [
            groundwater_model(torch.tensor(u, dtype=torch.float32), f)
            .detach()
            .numpy()
            for u in u_samples
        ]

    # Step 5: Print the norm of each output field (p(x)) to check the magnitude
    for i in range(num_samples):
        print(f"Output {i + 1} norm: {np.linalg.norm(p[i])}")

    # Step 6: Plot the input fields (u(x)) for visualization
    plot_fields(
        [
            np.exp(_) for _ in u_samples
        ],  # Apply exponential to inputs for better visualization
        [f"Input u(x) {i + 1}" for i in range(num_samples)],
        "Input Fields u(x)",
        contour=False,  # Disable contour plotting for input fields
    )

    # Step 7: Plot the output fields (p(x)) from the forward operator
    plot_fields(
        p,
        [f"Output p(x) {i + 1}" for i in range(num_samples)],
        "Output Fields p(x)",
        contour=True,  # Enable contour plotting for output fields
    )

    # Step 8: Show the plotted figures
    plt.show()


# Main entry point for the simulation
if __name__ == "__main__":
    # Call the function to simulate the groundwater equation
    simulate_groundwater_eq(size=256, num_samples=3)
