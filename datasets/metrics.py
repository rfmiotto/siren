# pylint: disable=invalid-name

from typing import Literal
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


@dataclass
class Metrics:
    """
    Definitions:
    x, y -> Cartesian coordinates
    A, B -> Curvilinear coordinates
    """

    # pylint: disable=too-many-instance-attributes

    x: NDArray
    y: NDArray

    normal_x: NDArray = field(init=False)
    normal_y: NDArray = field(init=False)

    dAdx: NDArray = field(init=False)
    dBdx: NDArray = field(init=False)
    dAdy: NDArray = field(init=False)
    dBdy: NDArray = field(init=False)

    dxdA: NDArray = field(init=False)
    dxdB: NDArray = field(init=False)
    dydA: NDArray = field(init=False)
    dydB: NDArray = field(init=False)

    # scale factors
    h1: NDArray = field(init=False)
    h2: NDArray = field(init=False)

    jacobian: NDArray = field(init=False)

    orientation: Literal["clockwise", "counterclockwise"] = field(init=False)

    def __post_init__(self) -> None:
        self.dxdA, self.dxdB = np.gradient(self.x, edge_order=2)
        self.dydA, self.dydB = np.gradient(self.y, edge_order=2)

        self.jacobian = -self.dxdB * self.dydA + self.dxdA * self.dydB

        self.dAdx = self.dydB / self.jacobian
        self.dAdy = -self.dxdB / self.jacobian
        self.dBdx = -self.dydA / self.jacobian
        self.dBdy = self.dxdA / self.jacobian

        self.h1 = np.sqrt(self.dxdA**2 + self.dydA**2)
        self.h2 = np.sqrt(self.dxdB**2 + self.dydB**2)

        is_grid_counterclockwise = max(self.jacobian[0]) > 0
        if is_grid_counterclockwise:
            self.normal_x = -self.dydA
            self.normal_y = self.dxdA
            self.orientation = "counterclockwise"
        else:
            self.normal_x = self.dydA
            self.normal_y = -self.dxdA
            self.orientation = "clockwise"

    @property
    def dx(self):
        return self.dxdA

    @property
    def dy(self):
        return self.dydA
