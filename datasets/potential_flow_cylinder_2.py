from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import matplotlib.pylab as plt


FREESTREAM_VEL_MAGNITUDE = 0.1


@dataclass
class Cylinder:
    radius: float
    center_position: Tuple[float, float]


@dataclass
class Domain:
    center: Tuple[float, float]
    size: float
    nx: int
    ny: int


def save_image(
    x_coords_matrix: NDArray,
    y_coords_matrix: NDArray,
    variable: NDArray,
    filename: str,
    vmax: int = 0.3,
    vmin: int = -0.3,
) -> None:
    img_size = 256
    plt.figure(figsize=(img_size, img_size), dpi=1)
    plt.pcolor(
        x_coords_matrix,
        y_coords_matrix,
        variable,
        cmap="gray",
        vmax=vmax,
        vmin=vmin,
    )
    plt.gca().set_aspect("equal")
    # plt.gcf().set_facecolor("gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=1, transparent=True)
    plt.close("all")
    # plt.show()


def create_mesh(domain: Domain) -> Tuple[NDArray]:
    x0 = domain.center[0] - 0.5 * domain.size
    xf = domain.center[0] + 0.5 * domain.size
    y0 = domain.center[1] - 0.5 * domain.size
    yf = domain.center[1] + 0.5 * domain.size

    x = np.linspace(x0, xf, num=domain.nx)
    y = np.linspace(y0, yf, num=domain.ny)

    xgrid, ygrid = np.meshgrid(x, y)

    return xgrid, ygrid


def get_vortex_potential(angle: float, circulation: float) -> float:
    """Add effect of clockwise velocity due to point vortex of a given circulation"""
    return -0.5 * angle * circulation / np.pi


def get_radius_and_theta(
    point: Tuple[float, float], cylinder: Cylinder
) -> Tuple[float, float]:
    point_cylinder_vec = (
        point[0] - cylinder.center_position[0],
        point[1] - cylinder.center_position[1],
    )
    point_radius = np.sqrt(point_cylinder_vec[0] ** 2 + point_cylinder_vec[1] ** 2)
    point_theta = np.arctan2(point_cylinder_vec[1], point_cylinder_vec[0])

    return point_radius, point_theta


def get_induced_potential(
    point: Tuple[float, float], cylinder: Cylinder
) -> NDArray[np.float32]:
    point_radius, point_theta = get_radius_and_theta(point, cylinder)

    return (
        FREESTREAM_VEL_MAGNITUDE
        * (point_radius + cylinder.radius**2 / point_radius)
        * np.cos(point_theta)
    )


def get_induced_velocities(
    point: Tuple[float, float], cylinder: Cylinder
) -> NDArray[np.float32]:
    point_radius, point_theta = get_radius_and_theta(point, cylinder)

    induced_vel_radial = (
        FREESTREAM_VEL_MAGNITUDE
        * (1 - (cylinder.radius / point_radius) ** 2)
        * np.cos(point_theta)
    )

    induced_vel_theta = (
        -FREESTREAM_VEL_MAGNITUDE
        * (1 + (cylinder.radius / point_radius) ** 2)
        * np.sin(point_theta)
    )

    induced_vel_x = induced_vel_radial * np.cos(
        point_theta
    ) - point_radius * induced_vel_theta * np.sin(point_theta)

    induced_vel_y = induced_vel_radial * np.sin(
        point_theta
    ) + point_radius * induced_vel_theta * np.cos(point_theta)

    return induced_vel_x, induced_vel_y


def main() -> None:
    domain = Domain(center=(0, 0), size=2, nx=256, ny=256)
    gridx, gridy = create_mesh(domain)

    points = np.stack([gridx.ravel(), gridy.ravel()], axis=0)

    cylinder = Cylinder(radius=0.1, center_position=(1.1, 0))

    field_potential = get_induced_potential(points, cylinder)

    velocity_x, velocity_y = get_induced_velocities(points, cylinder)

    max_value_vel = max(velocity_x.max(), velocity_y.max())
    min_value_vel = min(velocity_x.min(), velocity_y.min())
    max_vel_plot = max(abs(max_value_vel), abs(min_value_vel))

    max_value_pot = field_potential.max()
    min_value_pot = field_potential.min()
    max_pot_plot = max(abs(max_value_pot), abs(min_value_pot))

    save_image(
        gridx,
        gridy,
        field_potential.reshape(gridx.shape),
        "potential.png",
        vmax=max_pot_plot,
        vmin=-max_pot_plot,
    )

    save_image(
        gridx,
        gridy,
        velocity_x.reshape(gridx.shape),
        "velocity_x.png",
        vmax=max_vel_plot,
        vmin=-max_vel_plot,
    )
    save_image(
        gridx,
        gridy,
        velocity_y.reshape(gridx.shape),
        "velocity_y.png",
        vmax=max_vel_plot,
        vmin=-max_vel_plot,
    )

    # plt.close("all")
    # plt.pcolor(gridx, gridy, gridy, cmap="gray")
    # plt.show()


if __name__ == "__main__":
    main()
