import matplotlib.pyplot as plt
import numpy as np
from groundwater.devito_op import GroundwaterEquation
from groundwater.utils import GaussianRandomField, plot_fields

np.random.seed(42)


def simulate_groundwater_eq(size=256, num_samples=3, threshold=-1):
    # Sample random fields for u(x).
    u_samples = GaussianRandomField(2, size, alpha=2, tau=4).sample(num_samples)
    if threshold > 0:
        u_samples[u_samples >= 0] = 12
        u_samples[u_samples < 0] = threshold

    # Zero forcing term f(x).
    f = np.zeros((size, size))
    # f = np.ones((size, size))

    # Setup Groundwater equation problem
    groundwater_eq = GroundwaterEquation(size)

    # Evaluate the forward operator for each input field.
    p = [groundwater_eq.eval_fwd_op(f, u) for u in u_samples]

    # Print norms of the outputs
    for i in range(num_samples):
        print(f"Output {i+1} norm: {np.linalg.norm(p[i])}")

    # Plot results.
    plot_fields(
        [np.exp(_) for _ in u_samples],
        [f"Input u(x) {i+1}" for i in range(num_samples)],
        "Input Fields u(x)",
        contour=True,
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
