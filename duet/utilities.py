import math
import string
import random
from .duet_vals import *
from .duet_vals import unwrap as uw
from .duet_vals import _b_hull, UNBOUNDED
import itertools
import string
import random
import functools
from collections.abc import Iterable



def theorem_2_11_K(epses,ε,δ):
    β = (ε * ε) / (28.04 * math.log(1/δ))
    return lemma_6_4(epses, β, δ)

# advcomp odometer
def lemma_6_4(epses, β, δ):
    t1 = sum(old_list(old_map(lambda ε: (math.exp(ε) - 1) * ε/2, epses)))
    t2a = sum(old_list(old_map(lambda ε: (ε * ε) + β, epses)))
    a = sum(old_list(old_map(lambda ε: (ε * ε), epses)))
    t2b = 1 + (math.log((a/β)+1))/2
    if δ == 0:
        δ = 0.0001
    t2c = math.log(2/δ)
    print(f't1: {t1}')
    print(f't2: {math.sqrt(2 * t2a * t2b * t2c)}')
    return t1 + math.sqrt(2 * t2a * t2b * t2c)

def penvlmax(penvl):
    x = EDPrivEnv(fp={}, sp={})
    for p in penvl:
        x = x.penv_max(p)
    return x

def iterable(obj):
    """
       :param obj:


       :return:

    """
    return isinstance(obj, Iterable)

old_list = list
old_zip = zip

def zip(x,y):
    """
       :param x:
       :param y:


       :return:

    """
    if isinstance(x,DuetWrapper) or isinstance(y,DuetWrapper):
        return DuetWrapper(old_zip(x,y),get_senv(x)+get_senv(y),get_mode(x),
                           _b_hull(get_bounds(x), get_bounds(y)))
    else:
        return old_zip(x,y)

def list(l):
    """
       :param l:


       :return:

    """
    l1,l2,l3 = itertools.tee(l,3)
    # TODO
    # modes = map(lambda x: x.mode.bound, old_list(l2))
    modes = old_map(lambda x: x.mode, old_list(l2))
    # modes = set(old_list(modes))
    dssources = old_map(lambda x: old_list(x.senv.sens.keys()), old_list(l3))
    fldss = old_list(itertools.chain(*old_list(dssources)))
    s = set(old_list(modes))

    if len(s) == 1 and len(set(fldss)) == 1:
        mode = s.pop()
        if hasattr(l, 'senv'):
            sns = l.senv.sens
        else:
            sns = {set(fldss).pop(): mode.bound}
        sens = sns
    else:
        # The elements disagree on metric or data source, so the list's
        # sensitivity cannot be stated in any one metric: treat every source
        # it depends on as unbounded. (Returning {} here claimed the list
        # depends on no data at all.)
        mode = LInf()
        sources = set(fldss)
        if hasattr(l, 'senv'):
            sources |= set(l.senv.sens)
        sens = {src: float('inf') for src in sources}
    elements = old_list(l1)
    bounds = _b_hull(*[get_bounds(x) for x in elements]) if elements else UNBOUNDED
    r = DuetWrapper(elements,SensEnv(sens),mode,bounds)
    return r

old_dict = dict

def list2(l):
    """
       :param l:


       :return:

    """
    senv = SensEnv({'source': 1})
    return DuetWrapper(l, senv, LInf())

def dict(d):
    """
       :param d:


       :return:

    """
    senv = SensEnv({'source': 1})
    return DuetWrapper(d, senv, LInf())

L2 = L2


def random_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """
       :param size:
       :param chars:


       :return:

    """
    return ''.join(random.choice(chars) for x in range(size))

random_generator()
# random_generator(3, "6793YUIO")

old_map = map

def map(f,l):
    """
       :param f:
       :param l:


       :return:

    """
    sx  = DataSource(random_generator())
    # Over a row-distance list each element is one row, which a change to
    # the data replaces arbitrarily: probe `f` with a one-row element, so
    # clipping inside `f` works as it does on data. Over a value-distance
    # list, probe with a unit value distance to measure f's Lipschitz
    # constant.
    rows = isinstance(get_mode(l), Rows)
    vss = []
    xσs = []
    σs = []
    newσ = SensEnv()
    modes = []
    result_bounds = []
    for a in unwrap(l):
        # A fresh environment per element, and no in-place edits below: an
        # `f` that returns its argument (or any value sharing its
        # environment) used to have `sx` popped from the one environment all
        # elements shared, so every later element was tracked as
        # 0-sensitive in `sx`.
        x = f(DuetWrapper(a,SensEnv({sx : 1.0}),Rows() if rows else LInf()))
        # get the sens for "x" from each result
        s = x.senv.sens.get(sx,0)
        xσs.append(s)
        # get the "non-x" sens for each result
        rest = SensEnv({k: v for k, v in x.senv.sens.items() if k != sx})
        σs.append(rest)
        modes.append(x.mode)
        result_bounds.append(get_bounds(x))
        vss.append(unwrap(x))
        # parallel composition for "x"
        newσ += rest
    xσ  = max(xσs)
    # The result's sensitivity covers every source `f` touches. (It used
    # to be discarded -- leaving sensitivity {} -- unless `f` touched
    # exactly one source besides the element, so even
    # map(lambda x: x + 1, ls) came out as depending on no data.)
    # every result element lies within the hull of the elements' bounds,
    # which the probes derive from public constants only
    bounds = _b_hull(*result_bounds) if result_bounds else UNBOUNDED
    result_modes = set(old_list(modes))
    if len(result_modes) != 1:
        # Results in different metrics: no single metric bounds the list.
        sources = set(l.senv.sens) | set(newσ.sens)
        return DuetWrapper(vss, SensEnv({src: float('inf') for src in sources}), LInf(), bounds)
    mode = result_modes.pop()
    if isinstance(mode, Rows):
        # Row-local results: a changed row changes only its own result, so
        # the list's row count carries over (0 if `f` ignores the element).
        # A change to another source changes every element that reads it,
        # and how many elements there are is private, so that count is
        # unbounded. (Counting them would put the data-dependent length
        # into the sensitivity, and so into the noise scale.)
        own = l.senv.scale(1 if xσ > 0 else 0)
        s = own + newσ.scale(float('inf'))
    else:
        # Value-distance results: each changed row (or unit of value
        # distance) moves its result by at most xσ; other sources compose
        # sequentially across the elements.
        s = newσ + l.senv.scale(xσ)
    r = DuetWrapper(vss,s,mode,bounds)
    return r


def unwrap(x):
    """
       :param x:


       :return:

    """
    return uw(x)

def mode_switch(m1,m2):
    """
       :param m1:
       :param m2:


       :return:

    """
    def mode_switch(func):
        """
           :param func:


           :return:

        """
        def wrapper(v,b,*args,**kwargs):
            """
               :param val:
               :param **kwargs:


               :return:

            """
            d = func(v,b,*args,**kwargs)
            if d.mode != m1() :
                raise ValueError(f'mode_switch: unexpected mode {d.mode} instead of {m1}')
            else:
                d.senv = d.senv.clip(b)
                d.mode = m2(b)
            return d
        return wrapper
    return mode_switch

@mode_switch(Rows,L2)
def L2_clip(v, b):
    """
       :param v:
       :param b:


       :return:

    """
    from .numpy import norm as duet_norm  # numpy's own norm returns a plain number
    norm = duet_norm(v, ord=2)
    if norm.val > b:
        return b * (v / norm)
    else:
        return v

def debug(msg):
    """
       :param msg:
       :param delta:


       :return:

    """
    # print(f'DEBUG: {msg}')
    pass
