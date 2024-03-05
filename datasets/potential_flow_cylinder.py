from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pylab as plt
import numpy as np
from numpy.typing import NDArray

from metrics import Metrics
from vector_ops import gradient_of_scalar

FREESTREAM_VEL_MAGNITUDE = 0.1


@dataclass
class CylinderField:
    init_radius: float
    final_radius: float
    num_points_r: int
    num_points_theta: int


def stretch_fn(
    nondimensional_eta: float, p_param: float = 1.7, q_param: float = 2.0
) -> float:
    return p_param * nondimensional_eta + (1.0 - p_param) * (
        1.0 - np.tanh(q_param * (1.0 - nondimensional_eta)) / np.tanh(q_param)
    )


def create_polar_mesh(cylinder: CylinderField) -> Tuple[NDArray]:
    nondimensional_eta = -(
        np.arange(cylinder.num_points_r) - 0.5 * cylinder.num_points_r
    ) / (0.5 * cylinder.num_points_r)

    mid_radius = 0.5 * (cylinder.init_radius + cylinder.final_radius)

    radius = mid_radius + stretch_fn(nondimensional_eta) * (
        cylinder.init_radius - mid_radius
    )

    theta = np.linspace(0, 2 * np.pi, num=cylinder.num_points_theta)

    r_matrix, theta_matrix = np.meshgrid(radius, theta)

    return r_matrix, theta_matrix


# def create_mesh(cylinder: CylinderField) -> Tuple[NDArray]:
#     nondimensional_eta = -(
#         np.arange(cylinder.num_points_r) - 0.5 * cylinder.num_points_r
#     ) / (0.5 * cylinder.num_points_r)

#     mid_radius = 0.5 * (cylinder.init_radius + cylinder.final_radius)

#     radius = mid_radius + stretch_fn(nondimensional_eta) * (
#         cylinder.init_radius - mid_radius
#     )

#     theta = np.linspace(0, 2 * np.pi, num=cylinder.num_points_theta)

#     r_matrix, theta_matrix = np.meshgrid(radius, theta)

#     x_coords_matrix = r_matrix * np.cos(theta_matrix)
#     y_coords_matrix = r_matrix * np.sin(theta_matrix)

#     return x_coords_matrix, y_coords_matrix


def save_image(
    x_coords_matrix: NDArray,
    y_coords_matrix: NDArray,
    variable: NDArray,
    filename: str,
    vmax: int = 0.3,
    vmin: int = -0.3,
) -> None:
    img_size = 1024
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


def vortex_potential(angle: float, circulation: float) -> float:
    """Add effect of clockwise velocity due to point vortex of a given circulation"""
    return -0.5 * angle * circulation / np.pi


def add_cylinder(
    cylinder: CylinderField,
    centroid: Tuple[float, float],
    circulation: Optional[float] = 0,
) -> Tuple[NDArray]:
    radius, theta = create_polar_mesh(cylinder=cylinder)

    induced_potential = (
        FREESTREAM_VEL_MAGNITUDE
        * (radius + cylinder.init_radius**2 / radius)
        * np.cos(theta)
    )

    induced_vel_radial = (
        FREESTREAM_VEL_MAGNITUDE
        * (1 - (cylinder.init_radius / radius) ** 2)
        * np.cos(theta)
    )

    induced_vel_theta = (
        -FREESTREAM_VEL_MAGNITUDE
        * (1 + (cylinder.init_radius / radius) ** 2)
        * np.sin(theta)
    )

    if circulation:
        induced_potential += vortex_potential(theta, circulation)
        induced_vel_radial -= circulation / (2 * np.pi * radius)

    x_coords_matrix = radius * np.cos(theta) + centroid[0]
    y_coords_matrix = radius * np.sin(theta) + centroid[1]

    induced_vel_x = induced_vel_radial * np.cos(
        theta
    ) - radius * induced_vel_theta * np.sin(theta)

    induced_vel_y = induced_vel_radial * np.sin(
        theta
    ) + radius * induced_vel_theta * np.cos(theta)

    return (
        x_coords_matrix,
        y_coords_matrix,
        induced_potential,
        induced_vel_x,
        induced_vel_y,
    )


def main() -> None:
    cylinder_1 = CylinderField(
        init_radius=2, final_radius=5, num_points_r=200, num_points_theta=361
    )

    (
        x_coords_matrix,
        y_coords_matrix,
        field_potential,
        velocity_x,
        velocity_y,
    ) = add_cylinder(cylinder=cylinder_1, centroid=(0, 0), circulation=0)

    freestream_potential = FREESTREAM_VEL_MAGNITUDE * x_coords_matrix

    # field_potential += freestream_potential

    # metrics = Metrics(x_coords_matrix, y_coords_matrix)

    # velocity_x, velocity_y = gradient_of_scalar(scalar=field_potential, metrics=metrics)

    print(f"Umax = {velocity_x.max()}, Umin = {velocity_x.min()}")
    print(f"Vmax = {velocity_y.max()}, Vmin = {velocity_y.min()}")

    max_value_vel = max(velocity_x.max(), velocity_y.max())
    min_value_vel = min(velocity_x.min(), velocity_y.min())
    max_vel_plot = max(abs(max_value_vel), abs(min_value_vel))

    max_value_pot = field_potential.max()
    min_value_pot = field_potential.min()
    max_pot_plot = max(abs(max_value_pot), abs(min_value_pot))

    save_image(
        x_coords_matrix,
        y_coords_matrix,
        field_potential,
        "potential.png",
        vmax=max_pot_plot,
        vmin=-max_pot_plot,
    )
    save_image(
        x_coords_matrix,
        y_coords_matrix,
        velocity_x,
        "velocity_x.png",
        vmax=max_vel_plot,
        vmin=-max_vel_plot,
    )
    save_image(
        x_coords_matrix,
        y_coords_matrix,
        velocity_y,
        "velocity_y.png",
        vmax=max_vel_plot,
        vmin=-max_vel_plot,
    )


if __name__ == "__main__":
    main()
