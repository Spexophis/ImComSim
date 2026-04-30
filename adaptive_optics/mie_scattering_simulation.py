import numpy as np
from scipy.special import legendre
from scipy import special
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')
plt.ion()

class MieScatteringSimulation:
    def __init__(self, particle_radius, wavelength, refractive_index):
        """
        Initialize Mie scattering simulation parameters

        :param particle_radius: Radius of the spherical particle (in micrometers)
        :param wavelength: Incident light wavelength (in micrometers)
        :param refractive_index: Complex refractive index of the particle
        """
        self.particle_radius = particle_radius
        self.wavelength = wavelength
        self.refractive_index = refractive_index

        # Calculate size parameter
        self.size_parameter = 2 * np.pi * particle_radius / wavelength

    @staticmethod
    def _riccati_bessel_psi(n, x):
        """
        Riccati-Bessel function psi

        :param n: Order of the function
        :param x: Input value
        :return: Riccati-Bessel psi function value
        """
        return np.sqrt(np.pi / (2 * x)) * special.jv(n + 0.5, x)

    @staticmethod
    def _riccati_bessel_xi(n, x):
        """
        Riccati-Bessel function xi

        :param n: Order of the function
        :param x: Input value
        :return: Riccati-Bessel xi function value
        """
        return np.sqrt(np.pi / (2 * x)) * special.hankel1(n + 0.5, x)

    def mie_coefficients(self):
        """
        Calculate Mie scattering coefficients

        :return: An1 and Bn1 coefficients for scattering calculations
        """
        m = self.refractive_index
        x = self.size_parameter

        An1 = np.zeros(50, dtype=complex)
        Bn1 = np.zeros(50, dtype=complex)

        for n in range(1, 50):
            # Mie scattering coefficient calculations
            psi_x = self._riccati_bessel_psi(n, x)
            psi_mx = self._riccati_bessel_psi(n, m * x)
            xi_x = self._riccati_bessel_xi(n, x)

            psi_x_diff = self._riccati_bessel_psi(n - 1, x) - (n / x) * psi_x
            psi_mx_diff = self._riccati_bessel_psi(n - 1, m * x) - (n / (m * x)) * psi_mx
            xi_x_diff = self._riccati_bessel_xi(n - 1, x) - (n / x) * xi_x

            An1[n] = (m * psi_mx * psi_x_diff - psi_x * psi_mx_diff) / \
                     (m * psi_mx * xi_x_diff - xi_x * psi_mx_diff)

            Bn1[n] = (psi_mx * psi_x_diff - m * psi_x * psi_mx_diff) / \
                     (psi_mx * xi_x_diff - m * xi_x * psi_mx_diff)

        return An1, Bn1

    def monte_carlo_scattering(self, num_photons=10000):
        """
        Perform Monte Carlo simulation of photon scattering

        :param num_photons: Number of photons to simulate
        :return: Scattering angles and intensities
        """
        # Get Mie scattering coefficients
        An1, Bn1 = self.mie_coefficients()

        # Initialize arrays to store results
        scattering_angles = []
        scattering_intensities = []

        # Monte Carlo simulation
        for _ in range(num_photons):
            # Random initial direction
            theta = np.arccos(1 - 2 * np.random.random())
            phi = 2 * np.pi * np.random.random()

            # Calculate scattering phase function
            phase_function = 0
            for n in range(1, 50):
                # Legendre polynomials for phase function calculation
                Pn = legendre(n)(np.cos(theta))
                phase_function += (2 * n + 1) * (np.abs(An1[n]) ** 2 + np.abs(Bn1[n]) ** 2) * Pn

            phase_function /= 2

            scattering_angles.append(theta)
            scattering_intensities.append(phase_function)

        return np.array(scattering_angles), np.array(scattering_intensities)

    @staticmethod
    def plot_scattering(angles, intensities):
        """
        Create multiple visualizations of scattering results

        :param angles: Scattering angles
        :param intensities: Scattering intensities
        """
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 5))

        # Polar Plot
        ax1 = fig.add_subplot(131, projection='polar')
        sc1 = ax1.scatter(angles, intensities, alpha=0.5, c=intensities, cmap='viridis')
        ax1.set_title('Polar Scattering Distribution')
        plt.colorbar(sc1, ax=ax1, label='Intensity')

        # Histogram of Angles
        ax2 = fig.add_subplot(132)
        ax2.hist(np.degrees(angles), bins=50, edgecolor='black')
        ax2.set_title('Scattering Angle Distribution')
        ax2.set_xlabel('Angle (degrees)')
        ax2.set_ylabel('Frequency')

        # Intensity vs Angle
        ax3 = fig.add_subplot(133)
        ax3.scatter(np.degrees(angles), intensities, alpha=0.5, c=intensities, cmap='plasma')
        ax3.set_title('Intensity vs Scattering Angle')
        ax3.set_xlabel('Angle (degrees)')
        ax3.set_ylabel('Scattering Intensity')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Simulation parameters
    particle_radius = 0.5  # micrometers
    wavelength = 0.5  # micrometers
    refractive_index = 1.5 + 0.1j  # complex refractive index

    # Create simulation instance
    sim = MieScatteringSimulation(particle_radius, wavelength, refractive_index)

    # Run Monte Carlo simulation
    a, i = sim.monte_carlo_scattering(num_photons=10000)

    # statistical analysis of scattering data
    print("Scattering Analysis:")
    print(f"Total Photons Simulated: {len(a)}")
    print(f"Mean Scattering Angle: {np.degrees(np.mean(a)):.2f} degrees")
    print(f"Median Scattering Angle: {np.degrees(np.median(a)):.2f} degrees")
    print(f"Mean Scattering Intensity: {np.mean(i):.4f}")
    print(f"Max Scattering Intensity: {np.max(i):.4f}")
    print(f"Min Scattering Intensity: {np.min(i):.4f}")

    # Plot results
    sim.plot_scattering(a, i)
