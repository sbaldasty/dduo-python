import math
import string
import random
from .duet_vals import *
from .duet_vals import unwrap as uw
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
        return DuetWrapper(old_zip(x,y),get_senv(x)+get_senv(y),get_mode(x))
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
    r = DuetWrapper(old_list(l1),SensEnv(sens),mode)
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
    vss = []
    xσs = []
    σs = []
    newσ = SensEnv()
    modes = []
    for a in unwrap(l):
        # A fresh environment per element, and no in-place edits below: an
        # `f` that returns its argument (or any value sharing its
        # environment) used to have `sx` popped from the one environment all
        # elements shared, so every later element was tracked as
        # 0-sensitive in `sx`.
        x = f(DuetWrapper(a,SensEnv({sx : 1.0}),LInf()))
        # get the sens for "x" from each result
        s = x.senv.sens.get(sx,0)
        xσs.append(s)
        # get the "non-x" sens for each result
        rest = SensEnv({k: v for k, v in x.senv.sens.items() if k != sx})
        σs.append(rest)
        modes.append(x.mode)
        vss.append(unwrap(x))
        # parallel composition for "x"
        newσ += rest
    xσ  = max(xσs)
    xσp = scale_by_constant(l.senv, l.mode, xσ)
    # sequential composition for everything else
    s  = newσ + xσp
    # `s` is the sensitivity of the result whatever sources `f` touches.
    # (This used to be discarded -- leaving sensitivity {} -- unless `f`
    # touched exactly one source besides the element, so even
    # map(lambda x: x + 1, ls) came out as depending on no data.)
    result_modes = set(old_list(modes))
    if len(result_modes) == 1:
        mode = result_modes.pop()
        sens = s.sens
    else:
        # Results in different metrics: no single metric bounds the list.
        mode = LInf()
        sens = {src: float('inf') for src in s.sens}
    r = DuetWrapper(vss,SensEnv(sens),mode)
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

@mode_switch(LInf,L2)
def L2_clip(v, b):
    """
       :param v:
       :param b:


       :return:

    """
    norm = np.linalg.norm(v, ord=2)
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
