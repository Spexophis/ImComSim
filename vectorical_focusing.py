import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

matplotlib.use('Qt5Agg')
plt.ion()


class VectFoc:

    def __init__(self):
        self.wavelength = 0.488e-6
        self.k0 = 2 * np.pi / self.wavelength
        self.f = 200 / 63
        self.na = 1.4
        self.n = 1.512
        self.k = self.n * self.k0
        self.alpha_max = np.arcsin(self.na / self.n)
        self.theta = None
        self.phi = None
        self.d_theta = None
        self.d_phi = None
        self.x_r = None
        self.y_r = None
        self.z_r = None
        self.x_v = None
        self.y_v = None
        self.efd_focus_x = None
        self.efd_focus_y = None
        self.zernike_terms = [
            (2, 2, 0.5),  # astigmatism
        ]

    def map_pupil_angular(self, n_theta=128, n_phi=512):
        self.theta = np.linspace(0, self.alpha_max, n_theta)
        self.phi = np.linspace(0, 2 * np.pi, n_phi)
        self.d_theta = (self.theta[1] - self.theta[0]) / (n_theta - 1)
        self.d_phi = (self.phi[1] - self.phi[0]) / (n_phi - 1)

    def map_focal_space(self, xr=(-1.024e-6, 1.024e-6), yr=(-1.024e-6, 1.024e-6), zr=(-0.512e-6, 0.512e-6), xyz=(16, 192, 192)):
        self.x_r = np.linspace(xr[0], xr[1], xyz[1])
        self.y_r = np.linspace(yr[0], yr[1], xyz[2])
        self.z_r = np.linspace(zr[0], zr[1], xyz[0])
        self.x_v, self.y_v = np.meshgrid(self.x_r, self.y_r, indexing='ij')
        self.efd_focus_x = np.zeros(xyz, dtype=np.complex128)
        self.efd_focus_y = np.zeros(xyz, dtype=np.complex128)

    def pupil_amplitude(self, rho, phi):
        return 1.0

    def pupil_phase(self, rho, phi):
        return zernike_phase(rho, phi, self.zernike_terms)

    def half_moon(self, rho, phi):
        if phi < np.pi:
            return 0.0
        else:
            return np.pi

    def pupil_aberration(self, rho, phi):
        return zernike_phase(rho, phi, self.zernike_terms)

    def compute_focal_field_lr_pol(self):
        for it, theta in enumerate(self.theta):
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)

            if self.alpha_max != 0:
                rho = sin_t / np.sin(self.alpha_max)
            else:
                rho = 0.0

            for ip, phi_ in enumerate(self.phi):
                amp = self.pupil_amplitude(rho, phi_)
                pha = self.pupil_phase(rho, phi_)

                # The local field in the pupil plane (before projection):
                #   E_pupil_x = amp * cos(pha)    (assuming polarization along x)
                #   E_pupil_y = amp * sin(pha)
                epx = amp * np.exp(1j * pha)  # x-polarized; ignoring y component in the pupil
                epy = 0.0

                # Now, Richards-Wolf says we must project the incoming polarization onto
                # the local spherical basis.  For purely x-polarized input, the field in
                # spherical coords is a combination of θ-hat and φ-hat components.
                #
                # For a beam polarized along x, the component in the spherical basis can be found as:
                #   E_theta = E0 cos(phi_)
                #   E_phi   = -E0 sin(phi_)
                #
                # Then reconvert (E_theta, E_phi) to Cartesian in the focal region:
                #   Ex = E_theta cos(phi_) cos_t - E_phi sin(phi_)
                #   Ey = E_theta sin(phi_) cos_t + E_phi cos(phi_)
                #
                # Let's do that. We'll treat E0 = Epx (the complex amplitude).
                e_theta = epx * np.cos(phi_)
                e_phi = -epx * np.sin(phi_)

                # Convert Etheta, Ephi back to Cartesian at the focal plane:
                # Etheta is along the direction "θ-hat" (which is "in-plane radial" at angle φ_)
                # Ephi is along "φ-hat" (azimuthal direction)
                #
                #   Ex =  Etheta cos(phi_) cos_t - Ephi sin(phi_)
                #   Ey =  Etheta sin(phi_) cos_t + Ephi cos(phi_)

                ex_polar = (e_theta * np.cos(phi_) * cos_t - e_phi * np.sin(phi_))
                ey_polar = (e_theta * np.sin(phi_) * cos_t + e_phi * np.cos(phi_))

                # For each (x,y), compute the phase factor
                #   exp[- i k (x sinθ cosφ + y sinθ sinφ + z cosθ )]
                # at z=0 => phase = exp[- i k (x sinθ cosφ + y sinθ sinφ)]
                #
                # We'll do this in a vectorized way:
                phase_factor = np.exp(
                    -1j * self.k * (self.x_v * sin_t * np.cos(phi_) + self.y_v * sin_t * np.sin(phi_)))

                # The contribution dE to the focal plane from this ring element is
                #   dE_x = Ex_polar * phase_factor * sinθ dθ dφ
                #   dE_y = Ey_polar * phase_factor * sinθ dθ dφ
                # plus any additional common prefactor, e.g., i*k*exp(i k f)/(2π f)...
                # We'll include a simple prefactor at the end or incorporate it now:

                d_ex = ex_polar * phase_factor * sin_t
                d_ey = ey_polar * phase_factor * sin_t

                # Accumulate:
                self.efd_focus_x += d_ex
                self.efd_focus_y += d_ey

        # Multiply by the constant prefactor outside the integral:
        pre_factor = 1j * self.k * np.exp(1j * self.k * self.f) / (2.0 * np.pi * self.f) * self.d_theta * self.d_phi
        self.efd_focus_x *= pre_factor
        self.efd_focus_y *= pre_factor
        int_focus = np.abs(self.efd_focus_x) ** 2 + np.abs(self.efd_focus_y) ** 2
        return int_focus

    def compute_focal_field_circ_pol(self):
        for it, theta in enumerate(self.theta):
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)

            if self.alpha_max != 0:
                rho = sin_t / np.sin(self.alpha_max)
            else:
                rho = 0.0

            for ip, phi_ in enumerate(self.phi):
                amp = self.pupil_amplitude(rho, phi_)
                pha = self.pupil_phase(rho, phi_)
                common = amp * np.exp(1j * pha)

                # For LEFT-HANDED circular in the pupil:
                #   E0x = 1/sqrt(2)*common
                #   E0y = (i / sqrt(2))*common
                # You can swap the sign of 'i' for right-handed if needed.
                e0x = (1.0 / np.sqrt(2)) * common
                e0y = (-1j / np.sqrt(2)) * common

                #  X-polarization portion
                #    Etheta_x = E0x cos(phi_)
                #    Ephi_x   = - E0x sin(phi_)
                e_theta_x = e0x * np.cos(phi_)
                e_phi_x = -e0x * np.sin(phi_)

                ex_x = (e_theta_x * np.cos(phi_) * cos_t - e_phi_x * np.sin(phi_))
                ey_x = (e_theta_x * np.sin(phi_) * cos_t + e_phi_x * np.cos(phi_))

                #  Y-polarization portion
                #    Etheta_y = E0y sin(phi_)
                #    Ephi_y   = E0y cos(phi_)
                e_theta_y = e0y * np.sin(phi_)
                e_phi_y = e0y * np.cos(phi_)

                ex_y = (e_theta_y * np.cos(phi_) * cos_t - e_phi_y * np.sin(phi_))
                ey_y = (e_theta_y * np.sin(phi_) * cos_t + e_phi_y * np.cos(phi_))

                # Total Ex, Ey from both x- and y-polarization
                ex_polar = ex_x + ex_y
                ey_polar = ey_x + ey_y

                # For each (x,y), compute the phase factor
                #   exp[- i k (x sinθ cosφ + y sinθ sinφ + z cosθ )]
                # at z=0 => phase = exp[- i k (x sinθ cosφ + y sinθ sinφ)]
                phase_factor = np.exp(
                    -1j * self.k * (self.x_v * sin_t * np.cos(phi_) + self.y_v * sin_t * np.sin(phi_)))

                # The contribution dE to the focal plane from this ring element is
                #   dE_x = Ex_polar * phase_factor * sinθ dθ dφ
                #   dE_y = Ey_polar * phase_factor * sinθ dθ dφ
                # plus any additional common prefactor, e.g., i*k*exp(i k f)/(2π f)...
                # We'll include a simple prefactor at the end or incorporate it now:

                d_ex = ex_polar * phase_factor * sin_t
                d_ey = ey_polar * phase_factor * sin_t

                # Accumulate:
                self.efd_focus_x += d_ex
                self.efd_focus_y += d_ey

        # Multiply by the constant prefactor outside the integral:
        pre_factor = 1j * self.k * np.exp(1j * self.k * self.f) / (2.0 * np.pi * self.f) * self.d_theta * self.d_phi
        self.efd_focus_x *= pre_factor
        self.efd_focus_y *= pre_factor
        int_focus = np.abs(self.efd_focus_x) ** 2 + np.abs(self.efd_focus_y) ** 2
        return int_focus

    def compute_focal_field_radi_pol(self):
        for it, theta in enumerate(self.theta):
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)

            if self.alpha_max != 0:
                rho = sin_t / np.sin(self.alpha_max)
            else:
                rho = 0.0

            for ip, phi_ in enumerate(self.phi):
                amp = self.pupil_amplitude(rho, phi_)
                pha = self.pupil_phase(rho, phi_)
                common = amp * np.exp(1j * pha)

                # ----- Radial polarization in the pupil plane -----
                # E0x = amp*cos(phi_),  E0y = amp*sin(phi_)
                # plus overall phase factor exp(i * pha)
                # so net is (E0x, E0y) = amp*exp(i*pha)*(cos(phi_), sin(phi_))
                e0x = common * np.cos(phi_)
                e0y = common * np.sin(phi_)

                # Now do the standard Richards-Wolf approach
                # The local field can be decomposed into spherical basis (theta-hat, phi-hat).
                #
                #   For x-polarized only:
                #       Eθ = E0x*cos(phi_),   Eφ = -E0x*sin(phi_)
                #   For y-polarized only:
                #       Eθ = E0y*sin(phi_),   Eφ =  E0y*cos(phi_)
                #
                # For general (E0x, E0y):
                #   Eθ = E0x*cos(phi_) + E0y*sin(phi_)
                #   Eφ = -E0x*sin(phi_) + E0y*cos(phi_)

                e_theta = e0x * np.cos(phi_) + e0y * np.sin(phi_)
                e_phi = -e0x * np.sin(phi_) + e0y * np.cos(phi_)

                # Convert (Etheta, Ephi) back to Cartesian in the focal plane:
                #
                #   Ex = Eθ * cos(phi_) * cos_t - Eφ * sin(phi_)
                #   Ey = Eθ * sin(phi_) * cos_t + Eφ * cos(phi_)
                ex_polar = e_theta * np.cos(phi_) * cos_t - e_phi * np.sin(phi_)
                ey_polar = e_theta * np.sin(phi_) * cos_t + e_phi * np.cos(phi_)

                # For each (x,y), compute the phase factor
                #   exp[- i k (x sinθ cosφ + y sinθ sinφ + z cosθ )]
                # at z=0 => phase = exp[- i k (x sinθ cosφ + y sinθ sinφ)]
                phase_factor = np.exp(
                    -1j * self.k * (self.x_v * sin_t * np.cos(phi_) + self.y_v * sin_t * np.sin(phi_)))

                # The contribution dE to the focal plane from this ring element is
                #   dE_x = Ex_polar * phase_factor * sinθ dθ dφ
                #   dE_y = Ey_polar * phase_factor * sinθ dθ dφ
                # plus any additional common prefactor, e.g., i*k*exp(i k f)/(2π f)...
                # We'll include a simple prefactor at the end or incorporate it now:

                d_ex = ex_polar * phase_factor * sin_t
                d_ey = ey_polar * phase_factor * sin_t

                # Accumulate:
                self.efd_focus_x += d_ex
                self.efd_focus_y += d_ey

        # Multiply by the constant prefactor outside the integral:
        pre_factor = 1j * self.k * np.exp(1j * self.k * self.f) / (2.0 * np.pi * self.f) * self.d_theta * self.d_phi
        self.efd_focus_x *= pre_factor
        self.efd_focus_y *= pre_factor
        int_focus = np.abs(self.efd_focus_x) ** 2 + np.abs(self.efd_focus_y) ** 2
        return int_focus

    def compute_focal_volume_lr_pol(self):
        pre_factor = (1j * self.k * np.exp(1j * self.k * self.f)) / (2.0 * np.pi * self.f)

        for iz, z_ in enumerate(self.z_r):
            ex_plane = np.zeros(self.x_v.shape, dtype=np.complex128)
            ey_plane = np.zeros(self.x_v.shape, dtype=np.complex128)

            for it, theta in enumerate(self.theta):
                sin_t = np.sin(theta)
                cos_t = np.cos(theta)

                if self.alpha_max != 0:
                    rho = sin_t / np.sin(self.alpha_max)
                else:
                    rho = 0.0

                for ip, phi_ in enumerate(self.phi):
                    amp = self.pupil_amplitude(rho, phi_)
                    pha = self.pupil_phase(rho, phi_)

                    # x-polarized across pupil => (Epx, Epy) = (amp*exp(i*pha), 0)
                    epx = amp * np.exp(1j * pha)
                    epy = 0.0

                    # Decompose into spherical basis:
                    #   Etheta = Epx*cos(phi_) + Epy*sin(phi_)
                    #   Ephi   = -Epx*sin(phi_) + Epy*cos(phi_)
                    e_theta = epx * np.cos(phi_) + epy * np.sin(phi_)
                    e_phi = -epx * np.sin(phi_) + epy * np.cos(phi_)

                    # Convert to Cartesian in focal region
                    #   Ex = Etheta*cos(phi_)*cos_t - Ephi*sin(phi_)
                    #   Ey = Etheta*sin(phi_)*cos_t + Ephi*cos(phi_)
                    ex_polar = (e_theta * np.cos(phi_) * cos_t - e_phi * np.sin(phi_))
                    ey_polar = (e_theta * np.sin(phi_) * cos_t + e_phi * np.cos(phi_))

                    # Phase factor for (x,y,z_) location: exp[-i*k ( x*sin_t*cos(phi_) + y*sin_t*sin(phi_) + z_*cos_t )]
                    phase_factor = np.exp(-1j * self.k * (self.x_v * sin_t * np.cos(phi_) + self.y_v * sin_t * np.sin(phi_) + z_ * cos_t))

                    # Contribution to field
                    d_ex = ex_polar * phase_factor * sin_t
                    d_ey = ey_polar * phase_factor * sin_t

                    ex_plane += d_ex
                    ey_plane += d_ey

            # Multiply by integration measure and prefactor
            ex_plane *= (pre_factor * self.d_theta * self.d_phi)
            ey_plane *= (pre_factor * self.d_theta * self.d_phi)

            # Store in our 3D arrays
            self.efd_focus_x[iz] = ex_plane
            self.efd_focus_y[iz] = ey_plane

        int_focus = np.abs(self.efd_focus_x) ** 2 + np.abs(self.efd_focus_y) ** 2
        return int_focus

#############################
# Zernike polynomials
#############################
def zernike_radial(n, m, rho):
    m_abs = abs(m)
    if (n < m_abs) or ((n - m_abs) % 2 != 0):
        return np.zeros_like(rho)
    ra = np.zeros_like(rho)
    kmax = (n - m_abs) // 2
    for k in range(kmax + 1):
        c = ((-1.0) ** k * scipy.special.factorial(n - k)
             / (scipy.special.factorial(k)
                * scipy.special.factorial((n + m_abs) // 2 - k)
                * scipy.special.factorial((n - m_abs) // 2 - k)))
        ra += c * rho ** (n - 2 * k)
    return ra


def zernike_2d(n, m, rho, phi):
    if m >= 0:
        return zernike_radial(n, m, rho) * np.cos(m * phi)
    else:
        return zernike_radial(n, -m, rho) * np.sin(-m * phi)


def zernike_phase(rho, phi, zernike_terms):
    total = 0.0
    for (n, m, c) in zernike_terms:
        total += c * zernike_2d(n, m, rho, phi)
    return total


if __name__ == "__main__":
    vf = VectFoc()
    vf.map_pupil_angular(n_theta=128, n_phi=384)
    vf.map_focal_space(xr=(-1.024e-6, 1.024e-6), yr=(-1.024e-6, 1.024e-6), zr=(-0.512e-6, 0.512e-6), xyz=(17, 128, 128))

    ifc = vf.compute_focal_field_lr_pol()


    # fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    #
    # extent = [vf.x_r[0] * 1e6, vf.x_r[-1] * 1e6, vf.y_r[0] * 1e6, vf.y_r[-1] * 1e6]
    #
    # im1 = axs[0, 0].imshow(np.abs(vf.efd_focus_x) ** 2, extent=extent, cmap='inferno', origin='lower',
    #                        interpolation='none')
    # axs[0, 0].set_title('Intensity |E_x|²')
    # axs[0, 0].set_xlabel('x (µm)')
    # axs[0, 0].set_ylabel('y (µm)')
    # fig.colorbar(im1, ax=axs[0, 0])
    #
    # im2 = axs[0, 1].imshow(np.abs(vf.efd_focus_y) ** 2, extent=extent, cmap='inferno', origin='lower',
    #                        interpolation='none')
    # axs[0, 1].set_title('Intensity |E_y|²')
    # axs[0, 1].set_xlabel('x (µm)')
    # axs[0, 1].set_ylabel('y (µm)')
    # fig.colorbar(im2, ax=axs[0, 1])

    # im3 = axs[1, 0].imshow(itn_z, extent=extent, cmap='inferno', origin='lower', interpolation='none')
    # axs[1, 0].set_title('Intensity |E_z|²')
    # axs[1, 0].set_xlabel('x (µm)')
    # axs[1, 0].set_ylabel('y (µm)')
    # fig.colorbar(im3, ax=axs[1, 0])

    # im4 = axs[1, 1].imshow(ifc, extent=extent, cmap='inferno', origin='lower', interpolation='none')
    # axs[1, 1].set_title('Total Intensity I')
    # axs[1, 1].set_xlabel('x (µm)')
    # axs[1, 1].set_ylabel('y (µm)')
    # fig.colorbar(im4, ax=axs[1, 1])
    #
    # plt.tight_layout()
    # plt.show()
