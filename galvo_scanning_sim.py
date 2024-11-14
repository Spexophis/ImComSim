import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


class Ray:
    def __init__(self, origin, direction, width=0.001, divergence=0.001):
        self.origin = np.array(origin)
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.points = [self.origin]
        self.width = width  # beam width at origin
        self.divergence = divergence  # half-angle divergence in radians

    def propagate(self, distance):
        new_point = self.origin + self.direction * distance
        self.points.append(new_point)
        # Calculate new beam width due to divergence
        self.width = self.width + 2 * distance * np.tan(self.divergence)
        return new_point

    def get_beam_profile(self, distance, num_points=16):
        """Generate points representing beam profile at given distance"""
        center = self.origin + self.direction * distance
        width = self.width + 2 * distance * np.tan(self.divergence)

        # Create basis vectors perpendicular to beam direction
        v1 = np.array([1, 0, 0]) if not np.allclose(self.direction, [1, 0, 0]) else np.array([0, 1, 0])
        v1 = v1 - np.dot(v1, self.direction) * self.direction
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(self.direction, v1)

        # Generate circle of points
        angles = np.linspace(0, 2 * np.pi, num_points)
        points = []
        for angle in angles:
            point = center + width / 2 * (v1 * np.cos(angle) + v2 * np.sin(angle))
            points.append(point)
        return np.array(points)


class Mirror:
    def __init__(self, center, normal, width=1.0):
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

        self.normal = R @ self.normal
        self.normal = self.normal / np.linalg.norm(self.normal)

    def intersect(self, ray):
        """Calculate intersection point of ray with mirror"""
        denom = np.dot(ray.direction, self.normal)
        if abs(denom) < 1e-6:
            return None

        d = np.dot(self.normal, (self.center - ray.origin)) / denom
        if d < 0:
            return None

        intersection = ray.origin + d * ray.direction

        # Check if intersection point is within mirror bounds
        dist_to_center = np.linalg.norm(intersection - self.center)
        if dist_to_center > self.width / 2:
            return None

        return intersection

    def reflect(self, ray):
        """Reflect ray off mirror surface"""
        intersection = self.intersect(ray)
        if intersection is None:
            return None

        # Calculate reflected direction using the reflection formula
        reflected = ray.direction - 2 * np.dot(ray.direction, self.normal) * self.normal

        # Create new ray with updated beam properties
        new_ray = Ray(intersection, reflected, ray.width, ray.divergence)
        return new_ray


class DistortionAnalyzer:
    def __init__(self):
        self.ideal_points = []
        self.actual_points = []
        self.beam_sizes = []

    def add_point(self, ideal_angle_x, ideal_angle_y, actual_point, beam_size):
        self.ideal_points.append([ideal_angle_x, ideal_angle_y])
        self.actual_points.append([actual_point[0], actual_point[1], actual_point[2]])
        self.beam_sizes.append(beam_size)

    def analyze_distortion(self):
        if not self.actual_points:
            return None

        ideal_points = np.array(self.ideal_points)
        actual_points = np.array(self.actual_points)

        try:
            # Calculate scaling factors
            x_scale = np.ptp(actual_points[:, 0]) / np.ptp(ideal_points[:, 0])
            y_scale = np.ptp(actual_points[:, 1]) / np.ptp(ideal_points[:, 1])

            # Calculate average beam size
            avg_beam_size = np.mean(self.beam_sizes)
            max_beam_size = np.max(self.beam_sizes)

            # Calculate scan pattern area (projected onto XY plane)
            points_2d = actual_points[:, :2]
            if len(points_2d) >= 3:  # Need at least 3 points for ConvexHull
                try:
                    hull = ConvexHull(points_2d)
                    scan_area = hull.volume
                except:
                    scan_area = 0
            else:
                scan_area = 0

            # Calculate linearity error
            n = int(np.sqrt(len(actual_points)))
            if n * n == len(actual_points):  # Perfect square check
                x_coords = actual_points[:, 0].reshape(n, n)
                y_coords = actual_points[:, 1].reshape(n, n)

                linearity_error_x = np.max(np.abs(np.diff(np.diff(x_coords, axis=1), axis=1))) if n > 2 else 0
                linearity_error_y = np.max(np.abs(np.diff(np.diff(y_coords, axis=0), axis=0))) if n > 2 else 0
            else:
                linearity_error_x = linearity_error_y = 0

            return {
                'x_scale': x_scale,
                'y_scale': y_scale,
                'avg_beam_size': avg_beam_size,
                'max_beam_size': max_beam_size,
                'scan_area': scan_area,
                'linearity_error_x': linearity_error_x,
                'linearity_error_y': linearity_error_y
            }
        except Exception as e:
            print(f"Error in distortion analysis: {e}")
            return None


def run_simulation(num_points=50, scan_range=30, beam_width=0.001, beam_divergence=0.001):
    """Run galvo mirror scanning simulation"""
    # Setup mirrors
    galvo_x = Mirror([0, 0, 0], [0, 1, 1])
    galvo_y = Mirror([0, 2, 0], [0, -1, 1])

    # Setup initial ray with beam properties
    source_ray = Ray([0, -2, -2], [0, 1, 1], beam_width, beam_divergence)

    # Setup analyzer
    analyzer = DistortionAnalyzer()

    # Storage for visualization
    all_rays = []
    screen_points = []
    beam_profiles = []

    angles = np.linspace(-scan_range / 2, scan_range / 2, num_points)

    for angle_x in angles:
        for angle_y in angles:
            # Reset mirrors to initial position
            galvo_x = Mirror([0, 0, 0], [0, 1, 1])
            galvo_y = Mirror([0, 2, 0], [0, -1, 1])

            # Rotate mirrors
            galvo_x.rotate(angle_x, 'x')
            galvo_y.rotate(angle_y, 'y')

            # Trace ray through system
            ray = source_ray
            ray_path = [ray]

            # Reflect off first mirror
            reflected = galvo_x.reflect(ray)
            if reflected:
                ray_path.append(reflected)

                # Reflect off second mirror
                reflected2 = galvo_y.reflect(reflected)
                if reflected2:
                    ray_path.append(reflected2)

                    # Propagate to screen
                    screen_z = 5
                    t = (screen_z - reflected2.origin[2]) / reflected2.direction[2]
                    screen_point = reflected2.origin + t * reflected2.direction
                    screen_points.append(screen_point)

                    # Get beam profile at screen
                    beam_profile = reflected2.get_beam_profile(t)
                    beam_profiles.append(beam_profile)

                    # Add to analyzer
                    analyzer.add_point(angle_x, angle_y, screen_point, reflected2.width)

            all_rays.append(ray_path)

    return all_rays, screen_points, beam_profiles, analyzer


def visualize_simulation(all_rays, screen_points, beam_profiles, analyzer):
    """Visualize the simulation results with distortion analysis"""
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)

    # 3D visualization
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    # Plot rays
    for ray_path in all_rays:
        for ray in ray_path:
            points = np.array(ray.points)
            if len(points) > 1:
                ax1.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', alpha=0.1)

    # Plot screen points
    if screen_points:
        screen_points_array = np.array(screen_points)
        ax1.scatter(screen_points_array[:, 0], screen_points_array[:, 1],
                    screen_points_array[:, 2], c='r', alpha=0.5, s=1)

    # Plot mirrors
    mirror_size = 1
    xx, yy = np.meshgrid([-mirror_size / 2, mirror_size / 2], [-mirror_size / 2, mirror_size / 2])
    ax1.plot_surface(xx, np.zeros_like(xx), yy, alpha=0.3, color='gray')
    ax1.plot_surface(xx, yy + 2, np.zeros_like(xx), alpha=0.3, color='gray')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Galvo Mirror Ray Tracing')

    # 2D scan pattern
    ax2 = fig.add_subplot(gs[0, 1])
    if screen_points:
        screen_points_array = np.array(screen_points)
        ax2.scatter(screen_points_array[:, 0], screen_points_array[:, 1],
                    c='r', alpha=0.5, s=1)

        # Plot beam profiles
        if beam_profiles:
            for profile in beam_profiles[::max(1, len(beam_profiles) // 25)]:  # Plot subset
                profile_2d = profile[:, [0, 1]]  # Project to 2D
                try:
                    hull = ConvexHull(profile_2d)
                    for simplex in hull.simplices:
                        ax2.plot(profile_2d[simplex, 0], profile_2d[simplex, 1],
                                 'b-', alpha=0.1)
                except:
                    continue

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Scan Pattern with Beam Profiles')
    ax2.axis('equal')

    # Distortion analysis
    ax3 = fig.add_subplot(gs[1, 0])
    analysis = analyzer.analyze_distortion()

    if analysis:
        metrics = [
            f"X Scale: {analysis['x_scale']:.3f}",
            f"Y Scale: {analysis['y_scale']:.3f}",
            f"Avg Beam Size: {analysis['avg_beam_size'] * 1000:.3f} mm",
            f"Max Beam Size: {analysis['max_beam_size'] * 1000:.3f} mm",
            f"Scan Area: {analysis['scan_area']:.3f} sq units",
            f"X Linearity Error: {analysis['linearity_error_x']:.3f}",
            f"Y Linearity Error: {analysis['linearity_error_y']:.3f}"
        ]

        ax3.text(0.1, 0.9, '\n'.join(metrics), transform=ax3.transAxes,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    ax3.set_title('Distortion Analysis')
    ax3.axis('off')

    # Beam size variation plot
    ax4 = fig.add_subplot(gs[1, 1])
    if analyzer.beam_sizes:
        n = int(np.sqrt(len(analyzer.beam_sizes)))
        if n * n == len(analyzer.beam_sizes):
            beam_size_grid = np.array(analyzer.beam_sizes).reshape(n, n)
            im = ax4.imshow(beam_size_grid * 1000, cmap='viridis')
            plt.colorbar(im, ax=ax4, label='Beam Size (mm)')
            ax4.set_title('Beam Size Variation')
            ax4.set_xlabel('X Scan Position')
            ax4.set_ylabel('Y Scan Position')

    plt.tight_layout()
    plt.show()


# Run simulation with error handling
try:
    all_rays, screen_points, beam_profiles, analyzer = run_simulation(
        num_points=15,
        scan_range=30,
        beam_width=0.001,  # 1mm initial beam width
        beam_divergence=0.001  # 1mrad divergence
    )
    visualize_simulation(all_rays, screen_points, beam_profiles, analyzer)
except Exception as e:
    print(f"Error during simulation: {e}")
