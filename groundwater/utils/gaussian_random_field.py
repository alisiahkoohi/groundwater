# Adapted from https://github.com/devitocodes/devito/blob/master/examples/cfd/09_Darcy_flow_equation.ipynb
import math
import numpy as np
import numpy.fft as fft

__all__ = ["GaussianRandomField"]


class GaussianRandomField:
    """
    Class for generating realizations of a Gaussian random field.

    Parameters:
    -----------
    dim : int
        Dimensionality of the random field (e.g., 2 for 2D fields).
    size : int
        Size of the grid for the random field in each dimension.
    alpha : float, optional
        Power exponent that controls the smoothness of the field. Defaults to 2.
    tau : float, optional
        Scale parameter that influences the correlation length of the field.
        Defaults to 3.
    sigma : float, optional
        Standard deviation of the field. If None, it is automatically calculated
        based on 'alpha', 'tau', and 'dim'. Defaults to None.
    boundary : str, optional
        Type of boundary conditions for the field. Currently, only "periodic" is
        supported. Defaults to "periodic".
    """

    def __init__(
        self,
        dim: int,
        size: int,
        alpha: float = 2,
        tau: float = 3,
        sigma: float = None,
        boundary: str = "periodic",
    ) -> None:
        self.dim = dim

        # If sigma is not provided, calculate it based on alpha, tau, and dim
        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))

        # Maximum wavenumber (k_max) is half of the grid size
        k_max = size // 2

        if dim == 2:
            # Create a grid of wavenumbers for a 2D field
            wavenumbers = np.concatenate(
                (np.arange(0, k_max, 1), np.arange(-k_max, 0, 1)),
                axis=0,
            )

            # Repeat wavenumbers across both dimensions
            wavenumbers = np.tile(wavenumbers, (size, 1))

            # k_x and k_y represent wavenumbers in the x and y directions,
            # respectively
            k_x = wavenumbers.transpose(1, 0)
            k_y = wavenumbers

            # Compute the square root of the eigenvalues of the covariance
            # matrix sqrt_eig corresponds to the Fourier coefficients'
            # magnitudes
            self.sqrt_eig = (
                (size**2)
                * math.sqrt(2.0)
                * sigma
                * (
                    (4 * (math.pi**2) * (k_x**2 + k_y**2) + tau**2)
                    ** (-alpha / 2.0)
                )
            )

            # Set the zero-frequency component to 0 to avoid a DC offset in the
            # field
            self.sqrt_eig[0, 0] = 0.0
        else:
            raise ValueError("Only 2D fields are supported.")

        # Store the grid size as a tuple with dimensions
        self.size = tuple([size] * self.dim)

    def sample(self, N: int) -> np.ndarray:
        """
        Generate N samples from the Gaussian random field.

        Parameters:
        -----------
        N : int
            Number of samples to generate.

        Returns:
        --------
        samples : np.ndarray
            An array of shape (N, size, size) containing N realizations of the
            Gaussian random field.
        """
        # Generate random coefficients for the Fourier space with normal
        # distribution
        coeff = np.random.randn(N, *self.size)

        # Multiply the random coefficients by the square root of the eigenvalues
        coeff = self.sqrt_eig * coeff

        # Compute the inverse Fourier transform and return the real part of the
        # field
        field = fft.ifftn(coeff).real

        # Normalize the field to have 0.5 maximum
        field /= np.max(field) / 2
        return field


# Example usage.
if __name__ == "__main__":
    # Generate a Gaussian random field
    grf = GaussianRandomField(2, size=128, alpha=3, tau=3)
    grf_samples = grf.sample(3)

    # Print the shape of the generated field
    print(grf_samples.shape)
    # Output: (3, 128, 128)
