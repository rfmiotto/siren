from typing import Annotated, Literal, TypeVar
import numpy as np
from numpy.typing import NDArray


T = TypeVar("T", bound=np.generic)

Array4 = Annotated[NDArray[T], Literal[4]]
ArrayNx2 = Annotated[NDArray[T], Literal["N", 2]]
ArrayNxN = Annotated[NDArray[T], Literal["N", "N"]]
ArrayNxNx1 = Annotated[NDArray[T], Literal["N", "N", 1]]
ArrayNxNx2 = Annotated[NDArray[T], Literal["N", "N", 2]]
