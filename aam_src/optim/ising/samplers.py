"""A module of samplers (Ising machines).

"""
import time
import numpy as np
import dwave 
from dwave.samplers import SimulatedAnnealingSampler
from ..core.solution import SolutionSet
from ..core.solver import Solver
from . import qubo

class NealSampler(Solver):
    """Simulated annealing sampler of dwave-neal.

    """

    def __init__(self):
        self._sampler = SimulatedAnnealingSampler()

    @property
    def properties(self):
        """A dictionary containing the solver's information.

        """
        return {
            'type': '{}.{}'.format(self.__class__.__module__,
                                   self.__class__.__name__)
        }

    def parameters(self, problem=None, **kwargs):
        """Return information of parameters that can be specified.

        Parameters
        ----------
        problem : Problem, optional
            An optimization problem one wants to solve by the solver.

        Returns
        -------
        dict [str, dict]
            A dictionary containing information of the parameters
            that can be specified for solving `problem`,
            in the format of ::

               {
                   param_name: {
                       'type': type,
                       'default': default,
                       'lb': lb,
                       'ub': ub
                   }
               }

            where `param_name` is a parameter name, `type`, `default`,
            `lb`, and `ub` are the type, the default value, the lower bound of
            the range, and the upper bound of the range of the parameter,
            respectively. The `type` is a `type` object or iterable of possible
            values. If the lower bound (the upper bound) does not exist or
            the type of the parameter is neither `float` nor `int`,
            `lb` (`ub`) should be `None`.

        """
        params_info = {
            'beta_init': {
                'type': float,
                'default': None,
                'lb': 0.0,
                'ub': None
            },
            'beta_fin': {
                'type': float,
                'default': None,
                'lb': 0.0,
                'ub': None
            },
            'beta_schedule_type': {
                'type': {'geometric', 'linear'},
                'default': 'geometric'
            },
            'num_sweeps': {
                'type': int,
                'default': 1000,
                'lb': 1,
                'ub': None
            },
            'num_reads': {
                'type': int,
                'default': 1,
                'lb': 1,
                'ub': None
            }
        }
        if problem is not None and problem.has_parameters:
            params_info.update(
                {
                    p.name: {
                        'type': p.type,
                        'default': p.default,
                        'lb': p.lb,
                        'ub': p.ub
                    }
                    for p in problem.parameters
                }
            )
        return params_info

    def solve(self, problem, pre_solutions=None, **params):
        """Solve a quadratic unconstrained binary optimization problem.

        Parameters
        ----------
        problem : Problem
            An optimization problem one wants to solve by the solver.
            This solver can accept only quadratic unconstrained binary
            optimization (QUBO) problems.
        pre_solutions : np.ndarray, optional
            A 2D NumPy array of solutions, each of which defines an initial
            state of simulated annealing. This sampler performs simulated
            annealing `num_reads` times for each solution.
        **params : dict, optional
            Parameters for solving `problem`. The information of
            the parameters can be obtained by `parameters` method.

        Returns
        -------
        SolutionSet

        """
        # generate the qubo coefficients dict
        prob_params = {
            p.name: params.get(p.name, p.default)
            for p in problem.parameters
        }
        Q, const = qubo.as_coefficients_dict(problem, **prob_params)
        # set parameters to be passed into the sampler
        params_info = self.parameters()
        params_ = {
            k: v
            for k, v in params.items()
            if k in params_info and k not in {'beta_init', 'beta_fin'}
        }
        params_.update(
            {
                k: d['default']
                for k, d in params_info.items()
                if k not in params
            }
        )
        if (params.get('beta_init') is not None) \
           and (params.get('beta_fin') is not None):
            params_['beta_range'] = (params['beta_init'], params['beta_fin'])
        else:
            params_['beta_range'] = None
        if pre_solutions is not None:
            initial_states = np.repeat(pre_solutions,
                                       repeats=params_['num_reads'],
                                       axis=0).astype(np.int8)
            variable_labels = list(map(lambda v: v.name, problem.variables))
            params_.update(
                initial_states=(initial_states, variable_labels),
                num_reads=len(initial_states)
            )
            initial_states_copy = initial_states.copy()
        # sample qubo
        start_time = time.perf_counter_ns()
        samples = self._sampler.sample_qubo(Q, **params_)
        end_time = time.perf_counter_ns()
        runtime = end_time - start_time  # in nanoseconds
        # sort columns of samples in the same order as that of `problem.vars`
        var_index = {
            var_name: i
            for i, var_name in enumerate(samples.variables)
        }
        sorted_index = [var_index[v.name] for v in problem.vars]
        solutions = samples.record.sample[:, sorted_index]
        # set objectives (energy)
        objectives = (samples.record.energy.reshape(len(solutions), 1)
                      + np.full((len(solutions), 1), const))
        # set additional data 'num_occurrences'
        appendix = {'num_occurrences': samples.record.num_occurrences}
        # set information dict
        params_['beta_range'] = samples.info['beta_range']
        params_['beta_schedule_type'] = samples.info['beta_schedule_type']
        if pre_solutions is not None:
            params_['initial_states'] = initial_states_copy
            # because functions of the `neal` module or related modules
            # modify the original `initial_states` array passed
        info = {
            'solver': self.properties,
            'parameters': {**prob_params, **params_},
            'sampler_timing': {'runtime': runtime, 'unit': 'nanoseconds'}
        }
        # construct and return the solution set
        return SolutionSet(problem=problem,
                           solutions=solutions,
                           parameters=np.array(list(prob_params.values())),
                           objectives=objectives,
                           info=info,
                           **appendix)
