import math
<<<<<<< HEAD
=======
from typing import Literal
>>>>>>> chinyemba/feature/clustering-sjcs


def semilog(
    value: float,
    epsilon: float = 1e-3,
) -> float:
    if abs(value) < epsilon:
        return value
    else:
<<<<<<< HEAD
        sign: int = 1 if value >= 0 else -1
=======
        sign: Literal[1, -1] = 1 if value >= 0 else -1
>>>>>>> chinyemba/feature/clustering-sjcs
        # log10 here is safe, since we know the value is not close to zero
        return sign * epsilon * math.log1p(abs(value) / epsilon)
