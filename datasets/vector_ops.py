# pylint: disable=invalid-name

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from metrics import Metrics


def contravariant_from_cartesian(
    vector_x: NDArray, vector_y: NDArray, metrics: Metrics
) -> Tuple[NDArray, NDArray]:
    component_1 = vector_x * metrics.dAdx + vector_y * metrics.dAdy
    component_2 = vector_x * metrics.dBdx + vector_y * metrics.dBdy

    return component_1, component_2


def gradient_of_scalar(scalar: NDArray, metrics: Metrics) -> Tuple[NDArray, NDArray]:
    dsdA, dsdB = np.gradient(scalar, edge_order=2)

    dsdx = dsdA * metrics.dAdx + dsdB * metrics.dBdx
    dsdy = dsdA * metrics.dAdy + dsdB * metrics.dBdy

    return dsdx, dsdy


def divergence_of_vector(
    vector_x: NDArray, vector_y: NDArray, metrics: Metrics
) -> NDArray:
    """
    Evaluates the divergence of a vector using the Voss-Weyl formula.
    The components of the input vector must be in the Cartesian system.
    """
    component_1, component_2 = contravariant_from_cartesian(vector_x, vector_y, metrics)

    part1 = np.gradient(component_1 * metrics.jacobian, edge_order=2)
    part2 = np.gradient(component_2 * metrics.jacobian, edge_order=2)
    return (part1 + part2) / metrics.jacobian
