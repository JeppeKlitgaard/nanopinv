from collections.abc import Sequence
from typing import Any

import jax
from beartype import beartype as typechecker
from jaxtyping import Bool as Bool
from jaxtyping import Float as Float
from jaxtyping import Int as Int
from jaxtyping import Key as Key
from jaxtyping import Shaped as Shaped
from jaxtyping import jaxtyped
from numpy import ndarray as _np_ndarray

# For now we use Any which maps to np.ndarray and all other array types,
# since we are probably compatible with all of them.
Array = Any
ArrayNumpy = _np_ndarray
ArrayJax = jax.Array
NDGrid = Sequence[Float[Array, "ax"]]


# typecheck = partial(jaxtyped, typechecker=typechecker)
def typecheck(func):
    return jaxtyped(func, typechecker=typechecker)


# typecheck = partial(jaxtyped, typechecker=typechecker)
def debug_typecheck(func):
    return jaxtyped(func, typechecker=...)
