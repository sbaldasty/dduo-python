import math
from functools import partial
from .duet_core import *
from .duet_envs import *
import pandas as pd
import numpy as np
from _collections_abc import dict_items
try:
    from pandas.api.typing import DataFrameGroupBy
except ImportError:  # pandas < 2.1
    from pandas.core.groupby import DataFrameGroupBy

class DistanceMetric():
    def __init__(self, bound=float('inf')):
        self.bound = bound

    # Two metrics are equal only when they are the same kind *and* bound:
    # L1() and LInf() share the default bound but measure distance
    # differently, and mode checks such as sens_sum's depend on telling them
    # apart.
    def __hash__(self):
        return hash((type(self), self.bound))

    def __eq__(self, other):
        if not isinstance(other, DistanceMetric):
            return NotImplemented
        return type(self) is type(other) and self.bound == other.bound

    def __repr__(self):
        return f'{type(self).__name__}({self.bound})'

# Modes say what a DuetWrapper's sensitivity number measures.
#
# Rows: a *row distance*. Sensitivity s means that one change to the data
#   source changes at most s rows (elements) of the value, each by an
#   arbitrary amount. Data sources start here (read_csv, numpy.load, ...).
#   Any per-row computation keeps the count; nothing about the values is
#   bounded until bounds or clipping say so.
# L1, L2, LInf: *value distances*. Sensitivity s bounds how far the value
#   itself moves, measured in that norm. Mechanisms release these.

class Rows(DistanceMetric):
    pass

class L1(DistanceMetric):
    pass

class L2(DistanceMetric):
    pass

class LInf(DistanceMetric):
    pass

# Data source
class DataSource:
    def __init__(self, desc):
        self.desc = desc

    def __str__(self):
        return f'DataSource({self.desc})'

    def __eq__(self, other):
        return isinstance(other, DataSource) and other.desc == self.desc

    def __hash__(self):
        return hash(self.desc)

# Utilities for the Duet Wrapper
def _sensitive(obj):
    """Whether `obj` depends on any tracked data source."""
    return isinstance(obj, DuetWrapper) and any(v != 0 for v in obj.senv.sens.values())

def unwrap(obj):
    if isinstance(obj, DuetWrapper):
        return unwrap(obj.val)
    else:
        return obj

def get_senv(obj):
    if isinstance(obj, DuetWrapper):
        return obj.senv
    else:
        return SensEnv()

def get_mode(obj):
    if isinstance(obj, DuetWrapper):
        return obj.mode
    else:
        return LInf()


def scale_by_constant(senv, mode, c):
    """The sensitivity environment of `c * x`, where `x` has (senv, mode).

    Scaling multiplies a value distance by |c|, but leaves a row distance
    alone: scaling every element by c does not change how many rows differ.
    (c == 0 makes the result constant, which is 0-sensitive either way.)
    """
    c = abs(c)
    if isinstance(mode, Rows):
        c = 0 if c == 0 else 1
    return senv.scale(c)

def _inverse(c):
    """The factor dividing by the constant `c` scales a distance by."""
    return float('inf') if c == 0 else 1 / c

def floor_div_by_constant(senv, mode, c):
    """The sensitivity environment of `x // c`, where `x` has (senv, mode).

    Flooring is not Lipschitz: inputs that differ by a hair can land on
    either side of a multiple of c. For a row distance that is harmless --
    the result still differs in only the rows that changed. A value
    distance gets no bound (every changed element may jump by 1), so the
    result is unbounded; floor before clipping instead.
    """
    if isinstance(mode, Rows):
        return senv
    return senv.scale(float('inf'))

# ── Public bounds ─────────────────────────────────────────────────────────
#
# A DuetWrapper's `bounds` is a pair (lo, hi) such that every element of the
# wrapped value lies in [lo, hi] -- for *every* possible dataset, not just
# this one. Bounds therefore come only from public constants (clip limits,
# literal operands, "a count is never negative"), never from the data, and
# (-inf, inf) means nothing is known. They propagate by interval arithmetic.

UNBOUNDED = (float('-inf'), float('inf'))
BOOLEAN = (0.0, 1.0)
NONNEGATIVE = (0.0, float('inf'))

def _bounds_of(x):
    """Public bounds of an operand: a wrapper's own, or a public constant's."""
    if isinstance(x, DuetWrapper):
        if x.bounds != UNBOUNDED or _sensitive(x):
            return x.bounds
        x = x.val  # a wrapper around public data: its own values are public
    if isinstance(x, (bool, int, float, np.number)):
        v = float(x)
        return UNBOUNDED if math.isnan(v) else (v, v)
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return UNBOUNDED
    if arr.size == 0:
        return UNBOUNDED
    lo, hi = np.min(arr), np.max(arr)
    if np.isnan(lo) or np.isnan(hi):
        return UNBOUNDED
    return (float(lo), float(hi))

def get_bounds(obj):
    return _bounds_of(obj)

def _b_neg(a):
    return (-a[1], -a[0])

def _b_add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def _b_sub(a, b):
    return _b_add(a, _b_neg(b))

def _b_mul(a, b):
    # 0 * inf is 0 here: a factor known to be exactly 0 zeroes any finite value
    corners = [0.0 if x == 0 or y == 0 else x * y for x in a for y in b]
    return (min(corners), max(corners))

def _b_div(a, b):
    if b[0] <= 0 <= b[1]:
        return UNBOUNDED
    return _b_mul(a, (1 / b[1], 1 / b[0]))

def _b_floordiv(a, b):
    q = _b_div(a, b)
    return (float(np.floor(q[0])), float(np.floor(q[1])))

def _b_abs(a):
    if a[0] >= 0:
        return a
    if a[1] <= 0:
        return _b_neg(a)
    return (0.0, max(-a[0], a[1]))

def _b_hull(*bs):
    return (min(b[0] for b in bs), max(b[1] for b in bs))

def _b_intersect(a, b):
    return (max(a[0], b[0]), min(a[1], b[1]))

# ── From rows to values ───────────────────────────────────────────────────

def per_row_change(bounds):
    """How far one changed row can move an element with these public bounds.

    A neighbouring dataset that *replaces* a row moves the element by at
    most hi - lo; one that *adds or removes* a row moves it by the value
    itself, at most max(|lo|, |hi|). Taking the larger covers both, and
    equals hi - lo whenever 0 lies within the bounds.
    """
    lo, hi = bounds
    return max(hi - lo, abs(lo), abs(hi))

def _elements_per_row(value):
    """Elements in one row of `value`: 1 for scalars and flat sequences."""
    try:
        shape = np.shape(value)
    except Exception:
        return 1
    if len(shape) < 2 or shape[0] == 0:
        return 1
    return int(np.prod(shape[1:]))

def _reduces_rows(axis, value):
    """Whether a reduction over `axis` combines different rows."""
    if axis is None:
        return True
    try:
        ndim = max(np.ndim(value), 1)
    except Exception:
        ndim = 1
    axes = axis if isinstance(axis, tuple) else (axis,)
    for a in axes:
        if a == 'index' or (isinstance(a, (int, np.integer)) and a % ndim == 0):
            return True
    return False

def rows_to_value(senv, bounds, elements_per_row=1):
    """A row distance as an L1 value distance: each changed row moves each
    of its elements by at most per_row_change(bounds)."""
    return senv.scale(per_row_change(bounds) * elements_per_row)

def release_sensitivity(val):
    """The value distance a mechanism releasing `val` must calibrate to.

    A row distance is converted through the value's public bounds; with no
    bounds (nothing clipped) it is unbounded.
    """
    if isinstance(val.mode, Rows):
        return rows_to_value(val.senv, val.bounds, _elements_per_row(val.val)).get_max()
    return val.senv.get_max()

# Bounds rules: like the sensitivity rules, keyed by type and method name,
# but mapping the receiver's bounds (and the call's arguments) to the
# result's. A method with no rule gets UNBOUNDED, which is always sound.

def bounds_same(bounds, args, kwargs=None):
    """The result's elements are among the receiver's (views, conversions)."""
    return bounds

def bounds_count(bounds, args, kwargs=None):
    """Counts, sizes and shapes are never negative."""
    return NONNEGATIVE

def bounds_clip(bounds, args, kwargs=None):
    kwargs = kwargs or {}
    lower = args[0] if len(args) > 0 else kwargs.get('lower')
    upper = args[1] if len(args) > 1 else kwargs.get('upper')
    # A missing or data-dependent limit says nothing on its side.
    inf = float('inf')
    l = (-inf, -inf) if lower is None or _sensitive(lower) else _bounds_of(lower)
    u = (inf, inf) if upper is None or _sensitive(upper) else _bounds_of(upper)
    # each element becomes max(min(x, u_i), l_i), which is monotone in x
    return (max(min(bounds[0], u[0]), l[0]), max(min(bounds[1], u[1]), l[1]))

def bounds_sum(bounds, args, kwargs=None):
    """Only the sign survives: n * hi would depend on the private n."""
    return (0.0 if bounds[0] >= 0 else float('-inf'),
            0.0 if bounds[1] <= 0 else float('inf'))

# These are the sensitivity "transformers" used in the lookup table below
def sens_id(senv, mode, args, kwargs=None, bounds=UNBOUNDED, value=None):
    return senv, mode

def sens_inf(senv, mode, args, kwargs=None, bounds=UNBOUNDED, value=None):
    #print(f'called inf {senv}')
    return senv.scale(float('inf')), mode


def sens_clip(senv, mode, args, kwargs=None, bounds=UNBOUNDED, value=None):
    # pandas' Series.clip(lower=None, upper=None, ...): positional or keyword
    kwargs = kwargs or {}
    lower = args[0] if len(args) > 0 else kwargs.get('lower')
    upper = args[1] if len(args) > 1 else kwargs.get('upper')
    if _sensitive(lower) or _sensitive(upper):
        # A limit that depends on the data moves every element when it moves.
        return senv.scale(float('inf')), mode
    # Clipping works element by element: it keeps a row distance (its
    # effect is the bounds it sets, see bounds_clip, which a sum or a
    # release turns into a value distance) and is 1-Lipschitz for value
    # distances.
    return senv, mode

def sens_count(senv, mode, args, kwargs=None, bounds=UNBOUNDED, value=None):
    """Counts, sizes and shapes: s changed rows move each count by at most s,
    so a row distance becomes a value distance with the same number."""
    return senv, LInf()

def sens_sum(senv, mode, args, kwargs=None, bounds=UNBOUNDED, value=None):
    if mode == L1():
        return senv, mode
    if isinstance(mode, Rows):
        kwargs = kwargs or {}
        axis = kwargs.get('axis', args[0] if args else None)
        if _reduces_rows(axis, value):
            return rows_to_value(senv, bounds, _elements_per_row(value)), L1()
        # reducing within each row is row-local
        return senv, mode
    # Summing elements that are not bounded in L1 (e.g. an unclipped column)
    # has unbounded sensitivity: clip before summing.
    return senv.scale(float('inf')), mode

def sens_getitem(senv, mode, args, kwargs=None, bounds=UNBOUNDED, value=None):
    # TODO: this should probably use the args in some cases
    return senv, mode

# This is the "lookup table" for sensitivities of methods
sens_effects = {
    np.ndarray:{
        'size': sens_count,
        'copy': sens_id,
        'shape': sens_count,
        'sum': sens_sum,
        'astype': sens_id,
        'flatten': sens_id,
        'dot': sens_id,
        '__getitem__': sens_getitem
    },
    pd.Series: {
        'shape': sens_count,
        'value_counts': sens_count,
        'to_dict': sens_id,
        'clip':  sens_clip,
        'sum':   sens_sum,
        # the same values in a numpy container
        'values': sens_id,
        'to_numpy': sens_id
    },
    pd.DataFrame: {
        'shape': sens_count, # conservative: the column count is public
        'count': sens_count,
        '__getitem__': sens_getitem,
        'values': sens_id,
        'to_numpy': sens_id,
        # partitions the rows: a changed row lands in one group
        'groupby': sens_id
    },
    DataFrameGroupBy: {
        # one count per group; a changed row moves one of them by 1
        'size': sens_count
    },
    tuple: {
        '__getitem__': sens_getitem
    },
    dict: {
        'update': sens_id,
        'keys': sens_id,
        '__contains__': sens_id,
        'items': sens_id
    },
    dict_items: {
        '__getitem__': sens_getitem,
    }
}

bounds_effects = {
    np.ndarray: {
        'size': bounds_count,
        'shape': bounds_count,
        'copy': bounds_same,
        'flatten': bounds_same,
        'sum': bounds_sum
    },
    pd.Series: {
        'shape': bounds_count,
        'value_counts': bounds_count,
        'clip': bounds_clip,
        'sum': bounds_sum,
        'values': bounds_same,
        'to_numpy': bounds_same
    },
    pd.DataFrame: {
        'shape': bounds_count,
        'count': bounds_count,
        'values': bounds_same,
        'to_numpy': bounds_same
    },
    DataFrameGroupBy: {
        'size': bounds_count
    },
}


class DuetWrapper:
    global sens_effects


    def __init__(self, obj, senv, mode=LInf(), bounds=None):
        bounds = UNBOUNDED if bounds is None else (float(bounds[0]), float(bounds[1]))
        if isinstance(obj, DuetWrapper):
            self.senv = senv + obj.senv
            self.val = obj.val
            self.mode = mode
            self.bounds = _b_intersect(bounds, obj.bounds)
        else:
            self.senv = senv
            self.val = obj
            self.mode = mode
            self.bounds = bounds


    def __getattr__(self, name):
        typ = type(self.val)
        attr = getattr(self.val, name)

        sens_mod = sens_effects.get(typ, {}).get(name)
        if sens_mod is None:
            # No rule says how this method moves distances (cumsum, for
            # one, spreads a change in one row across every later row), so
            # assume the worst. Register a rule in `sens_effects` for
            # methods known to be safe.
            sens_mod = sens_inf

        bounds_mod = bounds_effects.get(typ, {}).get(name)

        if callable(attr):
            def _missing(*args, **kwargs):
                # Every DuetWrapper argument, positional or keyword, feeds
                # its sensitivity into the result; the method itself gets
                # the plain values, as __getitem__ does with its key.
                new_senv = self.senv
                for a in [*args, *kwargs.values()]:
                    if isinstance(a, DuetWrapper):
                        new_senv = new_senv + a.senv

                result = attr(*[unwrap(a) for a in args],
                              **{k: unwrap(v) for k, v in kwargs.items()})

                new_senv, new_mode = sens_mod(new_senv, self.mode, args, kwargs,
                                              bounds=self.bounds, value=self.val)
                new_bounds = bounds_mod(self.bounds, args, kwargs) if bounds_mod else UNBOUNDED

                return DuetWrapper(result, new_senv, new_mode, new_bounds)
            return _missing
        else:
            new_senv, new_mode = sens_mod(self.senv, self.mode, [],
                                          bounds=self.bounds, value=self.val)
            new_bounds = bounds_mod(self.bounds, [], {}) if bounds_mod else UNBOUNDED

            return DuetWrapper(attr, new_senv, new_mode, new_bounds)

    def __getitem__(self, key):
        name = '__getitem__'
        typ = type(self.val)

        try:
            sens_mod = sens_effects[typ][name]
        except:
            sens_mod = sens_id
            # raise Exception(f'No sensitivity definition for {typ}.{name}')

        new_senv, new_mode = sens_mod(self.senv, self.mode, [key])
        if hasattr(self.val, '__getitem__'):
            result = self.val.__getitem__(unwrap(key))
        else:
            result = list(self.val).__getitem__(unwrap(key))

        # a selection's elements are elements of the whole
        return DuetWrapper(result, new_senv, new_mode, self.bounds)

    def __bool__(self):
        raise RuntimeError('Sensitive values cannot be used in guard position of a conditional')

    def __str__(self):
        sub = str(type(self.val))
        if self.bounds == UNBOUNDED:
            return f'DuetWrapper({sub}, {self.senv}, {self.mode})'
        return f'DuetWrapper({sub}, {self.senv}, {self.mode}, bounds={self.bounds})'

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        new_env = self.senv.scale(float('inf'))
        return DuetWrapper(self.val == unwrap(other), new_env, self.mode, BOOLEAN)

    def __abs__(self):
        return DuetWrapper(abs(self.val), self.senv, self.mode, _b_abs(self.bounds))

    def __gt__(self, other):
        new_env = self.senv.scale(float('inf'))
        newMode = self.mode
        return DuetWrapper(self.val > unwrap(other), new_env, newMode, BOOLEAN)

    def __lt__(self, other):
        new_env = self.senv.scale(float('inf'))
        newMode = self.mode
        return DuetWrapper(self.val < unwrap(other), new_env, newMode, BOOLEAN)

    def __le__(self, other):
        new_env = self.senv.scale(float('inf'))
        newMode = self.mode
        return DuetWrapper(self.val <= unwrap(other), new_env, newMode, BOOLEAN)

    def __ge__(self, other):
        new_env = self.senv.scale(float('inf'))
        newMode = self.mode
        return DuetWrapper(self.val >= unwrap(other), new_env, newMode, BOOLEAN)

    def __add__(self, other):
        return DuetWrapper(self.val + unwrap(other), self.senv + get_senv(other), self.mode,
                           _b_add(self.bounds, _bounds_of(other)))

    def __sub__(self, other):
        return DuetWrapper(self.val - unwrap(other), self.senv + get_senv(other), self.mode,
                           _b_sub(self.bounds, _bounds_of(other)))


    def __rsub__(self, other):
        return DuetWrapper(unwrap(other)-self.val, self.senv + get_senv(other), self.mode,
                           _b_sub(_bounds_of(other), self.bounds))

    def __radd__(self, other):
        return DuetWrapper(self.val + unwrap(other), self.senv + get_senv(other), self.mode,
                           _b_add(_bounds_of(other), self.bounds))

    def __and__(self, other):
        # & of non-negative integers (or booleans) is at most either operand
        b = _bounds_of(other)
        bounds = (0.0, min(self.bounds[1], b[1])) if self.bounds[0] >= 0 and b[0] >= 0 else UNBOUNDED
        return DuetWrapper(self.val & unwrap(other), self.senv + get_senv(other), self.mode, bounds)


    # Dividing *by* a sensitive value is unbounded (c / x blows up as x
    # approaches 0), as it already is when both operands are sensitive.
    # Python only calls these when `other` is not a DuetWrapper.
    def __rtruediv__(self, other):
        return DuetWrapper(unwrap(other)/self.val, self.senv.scale(float('inf')), self.mode,
                           _b_div(_bounds_of(other), self.bounds))

    def __rfloordiv__(self, other):
        return DuetWrapper(unwrap(other)//self.val, self.senv.scale(float('inf')), self.mode,
                           _b_floordiv(_bounds_of(other), self.bounds))

    # No __itruediv__/__ifloordiv__: Python then evaluates `x /= c` as
    # `x = x / c` through __truediv__. (The in-place versions used to compute
    # c / x instead.)

    def __int__(self):
        return int(self.val)

    # def to_dict(self):
    #     y = unwrap(self)
    #     r = pd.Series.to_dict(y)
    #     return DuetWrapper(r, self.senv, LInf())

    def __neg__(self):
        return DuetWrapper(-1 * (self.val), self.senv, self.mode, _b_neg(self.bounds))

    def __float__(self):
        return float(self.val)

    def __truediv__(self,other):
        if isinstance(other, DuetWrapper):
            new_env = self.senv.scale(float('inf'))
        elif isinstance(other, (int, float)):
            new_env = scale_by_constant(self.senv, self.mode, _inverse(other))
        else:
            new_env = self.senv.scale(float('inf'))
        # print(self.val)
        # print(unwrap(other))
        return DuetWrapper(self.val / unwrap(other), new_env, self.mode,
                           _b_div(self.bounds, _bounds_of(other)))

    def __floordiv__(self,other):
        if isinstance(other, DuetWrapper):
            new_env = self.senv.scale(float('inf'))
        elif isinstance(other, (int, float)):
            new_env = floor_div_by_constant(self.senv, self.mode, other)
        else:
            new_env = self.senv.scale(float('inf'))

        return DuetWrapper(self.val // unwrap(other), new_env, self.mode,
                           _b_floordiv(self.bounds, _bounds_of(other)))


    def __matmul__(self,other):
        if isinstance(other, DuetWrapper):
            new_env = self.senv.scale(float('inf'))
        elif isinstance(other, (int, float)):
            new_env = scale_by_constant(self.senv, self.mode, other)
        else:
            new_env = self.senv.scale(float('inf'))

        return DuetWrapper(self.val @ unwrap(other), new_env, self.mode)

    def __mul__(self, other):
        if isinstance(other, DuetWrapper):
            new_env = self.senv.scale(float('inf'))
        elif isinstance(other, (int, float)):
            new_env = scale_by_constant(self.senv, self.mode, other)
        else:
            new_env = self.senv.scale(float('inf'))

        return DuetWrapper(self.val * unwrap(other), new_env, self.mode,
                           _b_mul(self.bounds, _bounds_of(other)))

    def __rmul__(self, other):
        if isinstance(other, DuetWrapper):
            new_env = self.senv.scale(float('inf'))
        elif isinstance(other, (int, float)):
            new_env = scale_by_constant(self.senv, self.mode, other)
        else:
            new_env = self.senv.scale(float('inf'))

        return DuetWrapper(self.val * unwrap(other), new_env, self.mode,
                           _b_mul(self.bounds, _bounds_of(other)))
