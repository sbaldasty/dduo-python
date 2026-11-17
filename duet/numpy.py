from .duet_core import *
from .duet_envs import *
from .duet_vals import _sensitive
import pandas as pd
import numpy as np
import builtins
import math


def _unbounded(x):
    return get_senv(x).scale(float('inf'))

def _elementwise(x, fn, lipschitz=None):
    """`fn` applied to each element of `x`.

    Each output element depends only on the matching input element, so a
    row-counting LInf sensitivity carries over unchanged. An L1/L2 value
    distance moves by at most `lipschitz` times as much -- or without bound
    when `fn` has no Lipschitz constant -- and must not be relabelled LInf
    with its number kept, since that number is not a row count.
    """
    r = fn(unwrap(x))
    senv, mode = get_senv(x), get_mode(x)
    if isinstance(mode, LInf):
        return DuetWrapper(r, senv, LInf())
    if lipschitz is None:
        return DuetWrapper(r, _unbounded(x), LInf())
    return DuetWrapper(r, senv.scale(lipschitz), mode)

def _common_mode(*xs):
    """The mode shared by every sensitive operand, or None if they differ."""
    modes = []
    for x in xs:
        if _sensitive(x) and get_mode(x) not in modes:
            modes.append(get_mode(x))
    if len(modes) > 1:
        return None
    return modes[0] if modes else LInf()

def load(filename):
    source = DataSource(filename)
    senv = SensEnv({source: 1})
    return DuetWrapper(np.load(filename), senv, LInf())

def zeros(x):
    if isinstance(x, DuetWrapper):
        y = unwrap(x)
        r = np.zeros(y)
        return DuetWrapper(r, x.senv, LInf())
    else:
        return np.zeros(x)

# def zeros(x):
#     y = unwrap(x)
#     r = np.zeros(y)
#     return DuetWrapper(r, x.senv.truncate(0), LInf())

def exp(x):
    if isinstance(x,DuetWrapper):
        return _elementwise(x, np.exp)
    else:
        return np.exp(x)

# def exp(x):
#     if isinstance(x,DuetWrapper):
#         y = unwrap(x)
#         r = np.exp(y)
#         if hasattr(x, 'senv'):
#             return DuetWrapper(r, x.senv.exp(), LInf())
#         else:
#             return DuetWrapper(r, SensEnv({}), LInf())
#     else:
#         return np.exp(x)

def abs(x):
    return _elementwise(x, np.abs, lipschitz=1)

def sign(x):
    return _elementwise(x, np.sign)

# def sign(x):
#     y = unwrap(x)
#     r = np.sign(y)
#     return DuetWrapper(r, get_senv(x).truncate(2), LInf())

def sqrt(x):
    return _elementwise(x, np.sqrt)

# def sqrt(x):
#     y = unwrap(x)
#     r = np.sqrt(x)
#     return DuetWrapper(r, get_senv(x).sqrt(), LInf())

def dot(x,y):
    a = unwrap(x)
    b = unwrap(y)
    r = np.dot(a,b)
    senv = get_senv(x)+get_senv(y)
    if not _sensitive(y) and isinstance(get_mode(x), LInf) and np.ndim(a) == 2:
        # (n x d) @ public: each output row comes from one input row
        return DuetWrapper(r, senv, LInf())
    if not _sensitive(x) and not _sensitive(y):
        return DuetWrapper(r, senv, LInf())
    # A product of two sensitive values, or a public weighting summed over
    # a sensitive operand's records: unbounded, as for an unclipped sum.
    return DuetWrapper(r, senv.scale(float('inf')), LInf())

# def dot2(x,y):
#     a = unwrap(x)
#     b = unwrap(y)
#     return DuetWrapper(np.dot(a,b), length(x)*(get_senv(x)*get_senv(y)), LInf())

def subtract(x,y):
    if isinstance(x, DuetWrapper) or isinstance(y, DuetWrapper):
        a = unwrap(x)
        b = unwrap(y)
        r = np.subtract(a,b)
        senv = get_senv(x)+get_senv(y)
        mode = _common_mode(x, y)
        if mode is None:
            # operands in different metrics: no one metric bounds the result
            return DuetWrapper(r, senv.scale(float('inf')), LInf())
        return DuetWrapper(r, senv, mode)
    else:
        return np.subtract(x, y)

def sum(x,*args,**kwargs):
    if isinstance(x,DuetWrapper):
        y = unwrap(x)
        r = np.sum(y,*args,**kwargs)
        axis = kwargs.get('axis', args[0] if args else None)
        if isinstance(x.mode, L2):
            if axis == 0:
                # per-row L2-clipped vectors summed over the rows
                return DuetWrapper(r, x.senv.truncate(x.mode.bound), x.mode)
            # k elements summed into each output: Cauchy-Schwarz gives sqrt(k)
            k = np.size(y) / builtins.max(np.size(r), 1)
            return DuetWrapper(r, x.senv.scale(math.sqrt(k)), x.mode)
        if isinstance(x.mode, L1):
            return DuetWrapper(r, x.senv, x.mode)
        # Summing elements with no value bound (an unclipped LInf array) is
        # unbounded; this used to return an empty sensitivity environment.
        return DuetWrapper(r, _unbounded(x), x.mode)
    else:
        return np.sum(x,*args,**kwargs)

sum2 = sum


# sum = np.sum

random = np.random
old_norm = np.linalg.norm
where = np.where
# abs = np.abs

def where(a,b,c):
    r = np.where(unwrap(a),unwrap(b),unwrap(c))
    senv = get_senv(a)+get_senv(b)+get_senv(c)
    mode = _common_mode(b, c)
    if mode is None or (_sensitive(a) and not isinstance(mode, LInf)):
        # Branches in different metrics, or a sensitive condition switching
        # between value-bounded branches (a jump no value bound covers).
        return DuetWrapper(r, senv.scale(float('inf')), LInf())
    return DuetWrapper(r, senv, mode)

def where2(a):
    return _elementwise(a, np.where)

def where3(a):
    return np.where(a)

def norm(x, ord=None, axis=None):
    xp = unwrap(x)
    new_senv = get_senv(x).scale(float('inf'))
    return DuetWrapper(old_norm(xp, ord, axis), new_senv, LInf())

class _Linalg:
    """numpy.linalg with DDuo's `norm`; every other name passes through.

    This used to be `np.linalg.norm = norm`, which replaced norm for every
    numpy user in the process (pandas, sklearn, ...) once duet.numpy was
    imported.
    """
    norm = staticmethod(norm)

    def __getattr__(self, name):
        return getattr(np.linalg, name)

linalg = _Linalg()

def array_split(arr, n):
    arrp = unwrap(arr)
    split_arr = np.array_split(arrp, n)
    sv = get_senv(arr)
    # Each chunk is a subset of the rows, in the same metric. The object
    # array is filled element by element: np.array([...]) would probe each
    # wrapper for __array_struct__, which DuetWrapper forwards to the array.
    wrapped_arr = np.empty(len(split_arr), dtype=object)
    for i, x in enumerate(split_arr):
        wrapped_arr[i] = DuetWrapper(x, sv, get_mode(arr))
    return wrapped_arr
