"""A module of QUBO-related functions and classes.

QUBO (Quadratic Unconstrained Binary Optimization) is a class of
combinatorial optimization. QUBO is mathematically equivalent to
the energy minimization of Ising models, which can be solved with
Ising machines.

This module includes a function for generating the coefficients
dictionary of a give QUBO problem (`as_coefficients_dict()`).
The coefficient dictionary of QUBO problems is often used as
an input format to Ising machines.

This module also includes QUBO-related error classes
(`NotQUBOError`, `NotPolynomialError`, and `NotQuadraticError`).

"""

from collections import Counter
from sympy import Add
from ..core.problem import Problem, Variable


def define_as_coefficients_dict_function_with_cache():
    """Return `as_coefficients_dict()` with caching feature."""

    Q_cache = {}

    def as_coefficients_dict(problem, **params):
        """Return the coefficients dict of a QUBO problem.

        Parameters
        ----------
        problem : Problem
            A QUBO problem.
        **params : dict, optional
            Values of parameters of `problem`. For parameters that have
            their default values and whose values are not specified
            in `**params`, their values will be considered to be
            their default values.

        Returns
        -------
        Q : dict [tuple, float]
            The coefficients dictionary of the objective function of
            `problem` in the format of ``{(u,v): coeff}`` where `u`
            and `v` are variable names and `coeff` is their associated
            coefficient.
        const : float
            The constant term of the objective function of `problem`.

        Raises
        ------
        NotQUBOError
            If `problem` is not QUBO.
        NotPolynomialError
            If the objective function of `problem` is not polynomial.
        NotQuadraticError
            If the objective function of `problem` is not quadratic.
        TypeError
            If there exists a parameter whose value is not specified.

        Notes
        -----
        This method takes into account the idempotent law: ``x*x == x``
        for any binary variables `x`. For example, the cubic expression
        ``x*(y**2)`` is treated by this method as the quadratic expression
        ``x*y`` if `x` and `y` are binary variables.

        """
        if not isinstance(problem, Problem):
            raise NotQUBOError("`problem` is not `Problem`.")
        if not all(v.is_binary for v in problem.variables):
            raise NotQUBOError("`problem` contains non-binary variables.")
        if problem.is_constrained:
            raise NotQUBOError("`problem` is constrained.")
        if problem.is_multi_objective:
            raise NotQUBOError("`problem` is multi objective.")
        if not problem.is_closed_form:
            raise NotQUBOError(
                "`problem` is not modeled by closed form expressions."
            )

        if problem in Q_cache:
            Q, const = Q_cache[problem]

        elif problem.is_single_objective:
            expr = problem.objectives[0].expr.expand()
            gens = set(problem.variables)
            Q = Counter()
            const = 0.0
            if isinstance(expr, Add):
                terms = expr.args
            else:
                terms = (expr, )
            for t in terms:
                vars_ = []
                coeff = 1
                for base, exp in t.as_powers_dict().items():
                    if base in gens and exp.is_integer and exp.is_positive:
                        vars_.append(base)
                    elif len(base.atoms(Variable)) == 0:
                        coeff *= base**exp
                    else:
                        raise NotPolynomialError(t)
                if len(vars_) == 0:
                    const += coeff
                elif len(vars_) == 1:
                    v,  = vars_
                    Q.update({(v.name, v.name): coeff})
                elif len(vars_) == 2:
                    u, v = sorted(vars_, key=lambda v: v.name)
                    Q.update({(u.name, v.name): coeff})
                else:
                    raise NotQuadraticError(t)

            if not problem.has_parameters:
                Q = {k: float(c) for k, c in Q.items()}
                const = float(const)

            Q_cache[problem] = (Q, const)

        else:
            Q = {}
            const = 0.0
            Q_cache[problem] = (Q, const)

        if problem.has_parameters:
            prob_params = {p: params.get(p.name, p.default)
                           for p in problem.parameters}
            if all(isinstance(v, (int, float)) for v in prob_params.values()):
                Q = {k: (float(c) if isinstance(c, (int, float)) else
                         float(c.xreplace(prob_params)))
                     for k, c in Q.items()}
                const = (float(const) if isinstance(const, (int, float)) else
                         float(const.xreplace(prob_params)))
            else:
                raise TypeError(
                    "Unspecified parameters: " + ', '.join(
                        p.name for p in problem.parameters
                        if not isinstance(prob_params[p], (int, float))
                    ) + "."
                )

        return Q, const

    return as_coefficients_dict


as_coefficients_dict = define_as_coefficients_dict_function_with_cache()


class NotQUBOError(Exception):
    """Problem is not QUBO."""
    pass


class NotPolynomialError(NotQUBOError):
    """Objective function is not a polynomial."""
    pass


class NotQuadraticError(NotQUBOError):
    """Objective function is not quadratic."""
    pass
