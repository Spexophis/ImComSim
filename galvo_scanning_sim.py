import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('QtAgg')
plt.ion()


class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.points = [self.origin.copy()]

    def propagate(self, distance):
        new_point = self.origin + self.direction * distance
        self.points.append(new_point.copy())
        return new_point


class Mirror:
    def __init__(self, center, normal, width=2.0):
        self.center = np.array(center)
        self.normal = np.array(normal) / np.linalg.norm(normal)
        self.width = width

    def rotate(self, angle, axis='x'):
        """Rotate mirror by given angle (in degrees) around specified axis"""
        angle_rad = np.radians(angle)
        if axis == 'x':
            R = np.array([[1, 0, 0],
                          [0, np.cos(angle_rad), -np.sin(angle_rad)],
                          [0, np.sin(angle_rad), np.cos(angle_rad)]])
        elif axis == 'y':
            R = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                          [0, 1, 0],
                          [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
        elif axis == 'z':
            R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                          [np.sin(angle_rad), np.cos(angle_rad), 0],
                          [0, 0, 1]])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

        self.normal = R @ self.normal
        self.normal = self.normal / np.linalg.norm(self.normal)

    def intersect(self, ray):
        """Calculate intersection point of ray with mirror"""
        denom = np.dot(ray.direction, self.normal)
        if abs(denom) < 1e-6:
            return None

        d = np.dot(self.normal, (self.center - ray.origin)) / denom

        # Allow both forward and backward intersections for debugging
        intersection = ray.origin + d * ray.direction

        # Simplified bound check - just check if it's within a square around center
        rel_pos = intersection - self.center
        if abs(rel_pos[0]) > self.width / 2 or abs(rel_pos[2]) > self.width / 2:
            return None

        return intersection

    def reflect(self, ray):
        """Reflect ray off mirror surface"""
        intersection = self.intersect(ray)
        if intersection is None:
            return None

        # Calculate reflected direction using the reflection formula
        reflected = ray.direction - 2 * np.dot(ray.direction, self.normal) * self.normal
        reflected = reflected / np.linalg.norm(reflected)

        # Create new ray starting at intersection point
        new_ray = Ray(intersection, reflected)

        # Add intersection point to original ray's path
        ray.points.append(intersection.copy())

        return new_ray


def run_simulation(num_points=5, scan_range=20):
    """Run galvo mirror scanning simulation"""
    # Storage for visualization
    all_rays = []
    screen_points = []

    # Create angle combinations
    angles = np.linspace(-scan_range / 2, scan_range / 2, num_points)

    # Initial ray parameters - adjust source point based on first mirror position
    source_point = np.array([0, 0, -2])  # Adjust this based on your needs
    initial_direction = np.array([0, 1, 1])

    for angle_x in angles:
        for angle_y in angles:
            # Create mirrors with new positions
            galvo_x = Mirror([0, 2, 0], [0, -1, 0])  # First mirror at y=2
            galvo_y = Mirror([0, 0, 2], [0, 1, 0])  # Second mirror at z=2

            # Rotate mirrors
            galvo_x.rotate(angle_x, 'x')
            galvo_y.rotate(angle_y, 'z')

            # Create initial ray
            initial_ray = Ray(source_point, initial_direction)
            ray_path = [initial_ray]

            # First reflection
            reflected1 = galvo_x.reflect(initial_ray)
            if reflected1:
                ray_path.append(reflected1)

                # Second reflection
                reflected2 = galvo_y.reflect(reflected1)
                if reflected2:
                    ray_path.append(reflected2)

                    # Propagate to screen - adjust screen_z based on your setup
                    screen_z = 7  # Moved further out due to new mirror positions
                    if abs(reflected2.direction[2]) > 1e-6:
                        t = (screen_z - reflected2.origin[2]) / reflected2.direction[2]
                        if t > 0:
                            screen_point = reflected2.origin + t * reflected2.direction
                            reflected2.points.append(screen_point)
                            screen_points.append(screen_point)

            all_rays.append(ray_path)

    return all_rays, screen_points


def visualize_simulation(all_rays, screen_points):
    """Visualize the simulation results"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot rays with different colors for each segment
    for ray_path in all_rays:
        # Plot initial ray
        if len(ray_path) >= 1:
            points = np.array(ray_path[0].points)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', alpha=0.3)

        # Plot first reflection
        if len(ray_path) >= 2:
            points = np.array(ray_path[1].points)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 'g-', alpha=0.3)

        # Plot second reflection
        if len(ray_path) >= 3:
            points = np.array(ray_path[2].points)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', alpha=0.3)

    # Plot screen points
    if screen_points:
        screen_points = np.array(screen_points)
        ax.scatter(screen_points[:, 0], screen_points[:, 1], screen_points[:, 2],
                   c='purple', alpha=0.5, s=20)

    # Plot mirrors
    mirror_size = 2.0
    xx, yy = np.meshgrid([-mirror_size / 2, mirror_size / 2], [-mirror_size / 2, mirror_size / 2])

    # First mirror at y=2
    ax.plot_surface(xx, np.full_like(xx, 2), yy, alpha=0.4, color='silver')

    # Second mirror at z=2
    ax.plot_surface(xx, np.full_like(xx, 0), yy + 2, alpha=0.4, color='silver')

    # Set view limits
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 5)
    ax.set_zlim(-3, 8)

    # Add legend
    ax.plot([], [], 'b-', label='Initial ray')
    ax.plot([], [], 'g-', label='First reflection')
    ax.plot([], [], 'r-', label='Second reflection')
    ax.scatter([], [], c='purple', s=20, label='Screen points')
    ax.legend()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Galvo Mirror Ray Tracing Simulation')

    # Set a better viewing angle
    ax.view_init(elev=20, azim=45)

    plt.show()


# Run simulation
all_rays, screen_points = run_simulation(num_points=5, scan_range=5)
visualize_simulation(all_rays, screen_points)

# Print debug information
print(f"Number of ray paths: {len(all_rays)}")
print(f"Number of complete paths (with reflections): {sum(1 for path in all_rays if len(path) > 2)}")
print(f"Number of screen points: {len(screen_points)}")
