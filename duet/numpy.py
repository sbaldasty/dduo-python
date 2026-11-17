from .duet_core import *
from .duet_envs import *
from .duet_vals import _sensitive, _bounds_of, _b_abs, _b_sub, _b_hull
import pandas as pd
import numpy as np
import builtins
import math


def _unbounded(x):
    return get_senv(x).scale(float('inf'))

def _elementwise(x, fn, lipschitz=None, bounds_fn=None):
    """`fn` applied to each element of `x`.

    Each output element depends only on the matching input element, so a
    row distance carries over unchanged. A value distance moves by at most
    `lipschitz` times as much -- or without bound when `fn` has no
    Lipschitz constant.

    `bounds_fn` maps the input's public bounds to the output's; without it
    the output's bounds are unknown.
    """
    r = fn(unwrap(x))
    senv, mode = get_senv(x), get_mode(x)
    bounds = bounds_fn(_bounds_of(x)) if bounds_fn else None
    if isinstance(mode, Rows):
        return DuetWrapper(r, senv, mode, bounds)
    if lipschitz is None:
        return DuetWrapper(r, _unbounded(x), LInf(), bounds)
    return DuetWrapper(r, senv.scale(lipschitz), mode, bounds)

def _monotone(fn):
    """Bounds of an increasing function: the endpoints' images."""
    def bounds_fn(b):
        with np.errstate(over='ignore', invalid='ignore'):
            lo, hi = float(fn(b[0])), float(fn(b[1]))
        return UNBOUNDED if math.isnan(lo) or math.isnan(hi) else (lo, hi)
    return bounds_fn

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
    return DuetWrapper(np.load(filename), senv, Rows())

def zeros(x):
    if isinstance(x, DuetWrapper):
        y = unwrap(x)
        r = np.zeros(y)
        return DuetWrapper(r, x.senv, LInf(), (0, 0))
    else:
        return np.zeros(x)

# def zeros(x):
#     y = unwrap(x)
#     r = np.zeros(y)
#     return DuetWrapper(r, x.senv.truncate(0), LInf())

def exp(x):
    if isinstance(x,DuetWrapper):
        return _elementwise(x, np.exp, bounds_fn=_monotone(np.exp))
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
    return _elementwise(x, np.abs, lipschitz=1, bounds_fn=_b_abs)

def sign(x):
    return _elementwise(x, np.sign, bounds_fn=_monotone(np.sign))

# def sign(x):
#     y = unwrap(x)
#     r = np.sign(y)
#     return DuetWrapper(r, get_senv(x).truncate(2), LInf())

def sqrt(x):
    # sqrt of a possibly negative input yields NaN, so bounds need lo >= 0
    return _elementwise(x, np.sqrt, bounds_fn=lambda b: _monotone(np.sqrt)(b) if b[0] >= 0 else UNBOUNDED)

# def sqrt(x):
#     y = unwrap(x)
#     r = np.sqrt(x)
#     return DuetWrapper(r, get_senv(x).sqrt(), LInf())

def dot(x,y):
    a = unwrap(x)
    b = unwrap(y)
    r = np.dot(a,b)
    senv = get_senv(x)+get_senv(y)
    if not _sensitive(y) and isinstance(get_mode(x), Rows) and np.ndim(a) == 2:
        # (n x d) @ public: each output row comes from one input row
        return DuetWrapper(r, senv, Rows())
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
        bounds = _b_sub(_bounds_of(x), _bounds_of(y))
        mode = _common_mode(x, y)
        if mode is None:
            # operands in different metrics: no one metric bounds the result
            return DuetWrapper(r, senv.scale(float('inf')), LInf(), bounds)
        return DuetWrapper(r, senv, mode, bounds)
    else:
        return np.subtract(x, y)

def sum(x,*args,**kwargs):
    if isinstance(x,DuetWrapper):
        y = unwrap(x)
        r = np.sum(y,*args,**kwargs)
        axis = kwargs.get('axis', args[0] if args else None)
        bounds = bounds_sum(x.bounds, args, kwargs)
        if isinstance(x.mode, L2):
            if axis == 0:
                # per-row L2-clipped vectors summed over the rows
                return DuetWrapper(r, x.senv.truncate(x.mode.bound), x.mode, bounds)
            # k elements summed into each output: Cauchy-Schwarz gives sqrt(k)
            k = np.size(y) / builtins.max(np.size(r), 1)
            return DuetWrapper(r, x.senv.scale(math.sqrt(k)), x.mode, bounds)
        if isinstance(x.mode, (L1, Rows)):
            # the same rule as Series.sum: rows become a value distance
            # through their public bounds
            senv, mode = sens_sum(x.senv, x.mode, args, kwargs, bounds=x.bounds, value=y)
            return DuetWrapper(r, senv, mode, bounds)
        # Summing an LInf value distance, which every element may use in
        # full, is unbounded. (Non-L2 sums used to return an empty
        # sensitivity environment.)
        return DuetWrapper(r, _unbounded(x), x.mode, bounds)
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
    bounds = _b_hull(_bounds_of(b), _bounds_of(c))
    mode = _common_mode(b, c)
    if mode is None or (_sensitive(a) and not isinstance(mode, Rows)):
        # Branches in different metrics, or a sensitive condition switching
        # between value-bounded branches (a jump no value bound covers).
        return DuetWrapper(r, senv.scale(float('inf')), LInf(), bounds)
    return DuetWrapper(r, senv, mode, bounds)

def where2(a):
    # the indices of the true elements
    return _elementwise(a, np.where, bounds_fn=lambda b: NONNEGATIVE)

def where3(a):
    return np.where(a)

def norm(x, ord=None, axis=None):
    xp = unwrap(x)
    new_senv = get_senv(x).scale(float('inf'))
    return DuetWrapper(old_norm(xp, ord, axis), new_senv, LInf(), NONNEGATIVE)

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
        wrapped_arr[i] = DuetWrapper(x, sv, get_mode(arr), _bounds_of(arr))
    return wrapped_arr
