import sys
sys.path.append('BayesianOptimization')
from bayes_opt_m import BayesianOptimization
from bayes_opt_m.observer import JSONLogger
from bayes_opt_m.event import Events


def black_box_function(**p):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -p['x1'] ** 2 - (p['x0'] - 1) ** 2 + 1


# Bounded region of parameter space
pbounds = {'x0': (2, 4), 'x1': (-3, 3)}
btypes = {'x0':int, 'x1':float}
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    ptypes=btypes,
    random_state=1,
)

#optimizer.probe(params={"x": 1, "y": 2})

#optimizer.register({"x": 2, "y": 2},4)
optimizer.maximize(
    init_points=2,
    n_iter=2,
)
print(optimizer.max)
