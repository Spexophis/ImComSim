import matplotlib.pyplot as plt
import numpy as np


class PhotonTransportSimulation:
    def __init__(self, tissue_dimensions, scattering_properties):
        """
        Initialize tissue and photon transport simulation

        :param tissue_dimensions: (width, height, depth) in micrometers
        :param scattering_properties: Dictionary of scattering parameters
        """
        self.tissue_width, self.tissue_height, self.tissue_depth = tissue_dimensions

        # Scattering properties
        self.scattering_coefficient = scattering_properties.get('scattering_coefficient', 100)  # 1/mm
        self.anisotropy_factor = scattering_properties.get('anisotropy_factor', 0.9)
        self.absorption_coefficient = scattering_properties.get('absorption_coefficient', 10)  # 1/mm

        # Initial photon source
        self.source_position = np.array([
            tissue_dimensions[0] / 2,  # x-center
            tissue_dimensions[1] / 2,  # y-center
            0  # bottom of tissue
        ])

    def generate_initial_directions(self, num_photons):
        """
        Generate photon directions uniformly distributed within the entire cone angle
        :param num_photons: Number of photons to simulate
        :return: Array of initial photon directions
        """
        # Half of the opening angle in radians
        half_cone_angle = np.deg2rad(60)  # 60 degrees / 2

        # Generate uniform distribution within the cone
        # Use inverse transform sampling to ensure uniform volume distribution

        # Uniform distribution on azimuthal angle
        phi = 2 * np.pi * np.random.random(num_photons)

        # Sampling cosine to get uniform distribution in volume
        # This ensures uniform distribution throughout the cone
        cos_theta = np.cos(half_cone_angle) + (1 - np.cos(half_cone_angle)) * np.random.random(num_photons)
        theta = np.arccos(cos_theta)

        # Convert to Cartesian coordinates
        # Assumes the cone is aligned with the z-axis
        directions = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        return directions

    def calculate_scattering_angle(self, num_photons):
        """
        Calculate photon scattering angles using Henyey-Greenstein phase function

        :param num_photons: Number of photons
        :return: Scattering angles
        """
        g = self.anisotropy_factor

        # Henyey-Greenstein phase function for scattering angle
        rand = np.random.random(num_photons)
        cos_theta = ((1 + g ** 2 - ((1 - g ** 2) / (1 - g + 2 * g * rand)) ** 2)
                     / (2 * g))

        return np.arccos(cos_theta)

    def simulate_photon_transport(self, num_photons):
        """
        Simulate photon transport through tissue

        :param num_photons: Number of photons to simulate
        :return: Exit points and trajectories
        """
        # Initial photon states
        positions = np.tile(self.source_position, (num_photons, 1))
        directions = self.generate_initial_directions(num_photons)

        # Tracking arrays
        exit_points = []
        trajectories = [[] for _ in range(num_photons)]

        # Simulation parameters
        max_steps = 1600
        step_size = 0.0004  # mm

        for p in range(num_photons):
            current_pos = positions[p]
            current_dir = directions[p]

            for step in range(max_steps):
                # Record trajectory
                trajectories[p].append(current_pos.copy())

                # Calculate step length (Beer-Lambert law)
                step_length = -np.log(np.random.random()) / self.scattering_coefficient

                # Update position
                current_pos += step_size * step_length * current_dir

                # Scatter direction
                scatter_angle = self.calculate_scattering_angle(1)[0]
                rotation_axis = np.cross(current_dir, [0, 0, 1])
                rotation_matrix = self._rotation_matrix(rotation_axis, scatter_angle)
                current_dir = np.dot(rotation_matrix, current_dir)

                # Check exit conditions
                if (current_pos[0] < 0 or current_pos[0] > self.tissue_width or
                        current_pos[1] < 0 or current_pos[1] > self.tissue_height or
                        current_pos[2] < 0 or current_pos[2] > self.tissue_depth):
                    exit_points.append(current_pos)
                    break

        return np.array(exit_points), trajectories

    def _rotation_matrix(self, axis, theta):
        """
        Create rotation matrix for 3D vector rotation

        :param axis: Rotation axis
        :param theta: Rotation angle
        :return: 3D rotation matrix
        """
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)

        return np.array([
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]
        ])

    def visualize_results(self, exit_points, trajectories):
        """
        Visualize photon transport simulation results

        :param exit_points: Photon exit points
        :param trajectories: Photon trajectories
        """
        fig = plt.figure(figsize=(15, 5))

        # Exit Point Distribution
        ax1 = fig.add_subplot(131)
        ax1.scatter(exit_points[:, 0], exit_points[:, 1], alpha=0.5)
        ax1.set_title('Photon Exit Point Distribution')
        ax1.set_xlabel('X Position (μm)')
        ax1.set_ylabel('Y Position (μm)')

        # 3D Trajectory Visualization
        ax2 = fig.add_subplot(132, projection='3d')
        for traj in trajectories[:50]:  # Plot first 50 trajectories
            traj = np.array(traj)
            ax2.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.3)
        ax2.set_title('Photon Trajectories')
        ax2.set_xlabel('X (μm)')
        ax2.set_ylabel('Y (μm)')
        ax2.set_zlabel('Z (μm)')

        # Exit Point Histogram
        ax3 = fig.add_subplot(133)
        hist, x_edges, y_edges = np.histogram2d(
            exit_points[:, 0], exit_points[:, 1],
            bins=[20, 20]
        )
        ax3.imshow(hist.T, origin='lower', extent=[
            x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]
        ])
        ax3.set_title('Exit Point Density')
        ax3.set_xlabel('X Position (μm)')
        ax3.set_ylabel('Y Position (μm)')

        plt.tight_layout()
        plt.show()

        # Additional statistical analysis
        print("Photon Transport Statistics:")
        print(f"Total Photons Simulated: {len(exit_points)}")
        print(f"Mean Exit X Position: {np.mean(exit_points[:, 0]):.2f} μm")
        print(f"Mean Exit Y Position: {np.mean(exit_points[:, 1]):.2f} μm")
        print(f"Mean Exit Z Position: {np.mean(exit_points[:, 2]):.2f} μm")
        print(f"Exit Point Spread (X): {np.std(exit_points[:, 0]):.2f} μm")
        print(f"Exit Point Spread (Y): {np.std(exit_points[:, 1]):.2f} μm")


if __name__ == "__main__":
    # Tissue Dimensions (micrometers)
    tissue_dimensions = (100, 50, 100)

    # Scattering Properties (typical biological tissue)
    scattering_properties = {
        'scattering_coefficient': 100,  # 1/mm
        'anisotropy_factor': 0.9,  # forward scattering
        'absorption_coefficient': 10  # 1/mm
    }

    # Create simulation
    sim = PhotonTransportSimulation(tissue_dimensions, scattering_properties)

    # Simulate photon transport
    exit_points, trajectories = sim.simulate_photon_transport(num_photons=5000)

    # Visualize results
    sim.visualize_results(exit_points, trajectories)
