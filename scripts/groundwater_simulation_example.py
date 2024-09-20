import matplotlib.pyplot as plt
import numpy as np
from groundwater.devito_op import GroundwaterEquation
from groundwater.utils import GaussianRandomField, plot_fields

# Number of pseudo-timesteps to simulate the groundwater equation. For larger
# input sizes, this number should be increased to ensure convergence.
NUM_PSEUDO_TIMESTEPS = 50000


def simulate_groundwater_eq(size=256, num_samples=3):
    # Sample random fields for u(x).
    u_samples = GaussianRandomField(2, size, alpha=2, tau=4).sample(num_samples)

    # Zero forcing term f(x).
    f = np.zeros((size, size))

    # Setup Groundwater equation problem
    groundwater_eq = GroundwaterEquation(size)

    # Evaluate the forward operator for each input field.
    p = [
        groundwater_eq.eval_fwd_op(f, u, time_steps=NUM_PSEUDO_TIMESTEPS)
        for u in u_samples
    ]

    # Print norms of the outputs
    for i in range(num_samples):
        print(f"Output {i+1} norm: {np.linalg.norm(p[i])}")

    # Plot results.
    plot_fields(
        [np.exp(_) for _ in u_samples],
        [f"Input u(x) {i+1}" for i in range(num_samples)],
        "Input Fields u(x)",
        contour=False,
    )
    plot_fields(
        p,
        [f"Output p(x) {i+1}" for i in range(num_samples)],
        "Output Fields p(x)",
        contour=True,
    )
    plt.show()


if __name__ == "__main__":
    simulate_groundwater_eq(size=256, num_samples=3)
