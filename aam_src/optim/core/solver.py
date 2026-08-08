"""A module for (abstract) optimization problem solvers.

Abstract base classes for optimization problem
solvers: `Solver` and `CompositeSolver`. The `Solver` abstract base
class defines the common interfaces of optimization problem solvers.
The `CompositeSolver` abstract base class is a subclass of `Solver`,
which is a superclass of all composite solvers. Here 'composite solver'
means a solver that solves optimization problems with using its child
solver functions. For example, a composition of an optimization problem
solver and some pre/post-processing functions is a composite solver
whose child solver is the constituent solver. Since `CompositeSolver` is
a subclass of `Solver`, clients can treat "plain" solvers and composite
solvers uniformly. See also
`composite pattern <https://en.wikipedia.org/wiki/Composite_pattern>`_.

"""

from abc import ABCMeta, abstractmethod


class Solver(metaclass=ABCMeta):
    """Abstract base class for optimization problem solvers.

    """

    @property
    @abstractmethod
    def properties(self):
        """A dictionary containing the solver's information.

        """
        pass

    @abstractmethod
    def parameters(self, problem=None):
        """Return information of parameters that can be specified.

        Parameters
        ----------
        problem : optimization.Problem, optional
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
            respectively. If the lower bound (the upper bound) does not exist
            or the type of the parameter is neither `float` nor `int`,
            `lb` (`ub`) should be `None`.

        """
        pass

    @abstractmethod
    def solve(self, problem, **params):
        """Solve an optimization problem.

        Parameters
        ----------
        problem : optimization.Problem
            An optimization problem one wants to solve by the solver.
        **params : dict, optional
            Parameters for solving `problem`. The information of
            the parameters can be obtained by `parameters` method.

        Returns
        -------
        optimization.SolutionSet

        """
        pass


class CompositeSolver(Solver):
    """Abstract base class for composite solvers.

    """

    @property
    @abstractmethod
    def child(self):
        """The child `Solver` object.

        """
        pass
