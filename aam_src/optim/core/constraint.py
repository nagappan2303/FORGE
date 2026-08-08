"""The module of the `Constraint` class.

"""

from itertools import chain

from sympy import Symbol, Add, sympify
from sympy.core.relational import Relational, Equality, GreaterThan, LessThan

from .variable import Variable
from .parameter import Parameter
from .utilities import lambdify, create_args_filtering_wrapper
from . import functions


class Constraint():
    """Constraints in optimization problems.

    Constraints can be modeled by a Python function which indicates
    whether the constraint is satisfied by a given set of values of
    variables and parameters, or by a closed form expression of
    `Variable` and `Parameter` objects such as ``Equality(x+2*y, z)``
    and ``a*x + b*y <= 0``.

    Parameters
    ----------
    name : str
        The name of the constraints.
    vars : list [Variable], optional
        The list of the variables in the constraint. If `expr` is specified,
        `vars` attribute will be automatically computed from `expr`.
        Otherwise, `vars` should be explicitly specified.
    params : list [Parameters], optional
        The list of the parameters in the constraint. If `expr` is specified,
        `params` attribute will be automatically computed from `expr`.
        Otherwise, `params` should be explicitly specified.
    func : function, optional
        A function which returns a Boolean value indicating
        whether the constraint is satisfied by a given set of values of
        variables and parameters. The function should accept values of
        variables and parameters as its arguments, e.g.,
        ``func(x=1, y=2, a=1, b=1)``.
    expr : sympy.core.relational.Relational, optional
        The mathematical expression (equality or inequality)
        representing the constraint.
    tol : float, default=1e-10
        A tolerance parameter used for checking if the constraint
        is satisfied in the following way ::

            A == B ? -> abs(A-B) <= tol ?
            A >= B ? -> A-B >= -tol ?
            A <= B ? -> A-B <= tol ?

        This parameter is valid only if `expr` is specified.

    Other Parameters
    ----------------
    lazy : bool, default=True
        If true, the creation of internal functions for the `is_satisfied()`
        and `expression_value()` methods based on a given SymPy expression
        will be delayed until these methods are called. Because the functions
        creation takes a bit long time, this lazy evaluation option saves
        computation time when the `is_satisfied()` and `expression_value()`
        methods are never called.

    Attributes
    ----------
    name : str
        The name of the constraint.
    vars : list [Variable]
        The list of the variables in the constraint.
    params : list [Parameter]
        The list of the parameters in the constraint.
    expr : sympy.core.relational.Relational or None
        The mathematical expression (equality or inequality)
        representing the constraint. If the constraint is not modeled
        by a mathematical expression, `expr` is `None`.
    tol : float or None
        A tolerance parameter used for checking if the constraint
        is satisfied by a given set of values of variables and parameters.
        If `expr` is not specified, `tol` is also not specified, i.e., `None`.

    """

    def __init__(self, name, vars=None, params=None,
                 func=None, expr=None, tol=1e-10, lazy=True):

        self._name = name

        # If the constraint is defined by a mathematical expression:
        if expr is not None:
            if not isinstance(expr, (Equality, GreaterThan, LessThan)):
                raise TypeError("The `expr` argument must be `Equality`, "
                                "`GreaterThan`, or `LessThan`.")
            self._expr = expr
            self._tol = tol
            self._vars = list(expr.atoms(Variable))
            self._params = list(expr.atoms(Parameter))
            if lazy:
                self._eval = None
                self._func = None
            else:
                self._set_functions_from_expr()

        # If the constraint is defined by a black-box function:
        elif func is not None:
            if not callable(func):
                raise TypeError("The `func` argument must be a `function`.")
            self._expr = None
            self._tol = None
            if (isinstance(vars, list)
               and all(isinstance(v, Variable) for v in vars)):
                self._vars = vars
            else:
                raise TypeError("The `vars` should be a list of `Variable`s.")
            if (isinstance(params, list)
               and all(isinstance(p, Parameter) for p in params)):
                self._params = params
            else:
                raise TypeError(
                    "The `params` should be a list of `Parameter`s."
                )
            self._func = create_args_filtering_wrapper(
                func=func,
                default_values={p.name: p.default for p in self._params
                                if p.default is not None}
            )

        # If the constraint is not defined:
        else:
            def tautology(**v): return True
            self._func = tautology
            self._expr = None
            self._tol = None
            self._vars = []
            self._params = []

    def _set_functions_from_expr(self) -> None:
        """Set `self._eval` and `self._func` based on `self.expr`.

        This method uses `optim.core.utilities.lambdify()` for
        creating the functions; the function creation takes a bit
        long time.

        """
        _eval = create_args_filtering_wrapper(
            func=lambdify(expr=(self._expr.lhs-self._expr.rhs),
                          vars=self._vars,
                          params=self._params),
            default_values={p.name: p.default for p in self._params
                            if p.default is not None}
        )

        self._eval = _eval

        tol = self._tol

        if self.is_equality:
            def _func(**v): return abs(_eval(**v)) <= tol
        elif self.is_greater_than:
            def _func(**v): return _eval(**v) >= -tol
        else:
            def _func(**v): return _eval(**v) <= tol
        self._func = _func

    @property
    def name(self):
        return self._name

    @property
    def vars(self):
        return self._vars

    @property
    def params(self):
        return self._params

    @property
    def expr(self):
        return self._expr

    @property
    def tol(self):
        return self._tol

    @property
    def is_closed_form(self):
        """Whether the constraint is expressed by a closed form expression.

        """
        return isinstance(self.expr, Relational)

    @property
    def is_equality(self):
        """Whether the constraint is an equality constraint.

        """
        return isinstance(self.expr, Equality)

    @property
    def is_inequality(self):
        """Whether the constraint is an inequality constraint.

        """
        return (self.is_greater_than or self.is_less_than)

    @property
    def is_greater_than(self):
        """Whether the constraint is an inequality constraint ``lhs >= rhs``.

        """
        return isinstance(self.expr, GreaterThan)

    @property
    def is_less_than(self):
        """Whether the constraint is an inequality constraint ``lhs <= rhs``.

        """
        return isinstance(self.expr, LessThan)

    def is_satisfied(self, **values):
        """Whether the constraint is satisfied.

        Parameters
        ----------
        **values :
            Keyword arguments specifying values of variables and parameters.
            For all variables and all parameters that do not have their
            default values, their values should be specified.
            For parameters that have their default values and whose values
            are not specified in `**values`, their values will be considered
            to be their default values.

        Returns
        -------
        bool

        """
        if self._func is None:
            self._set_functions_from_expr()
        return bool(self._func(**values))

    def expression_value(self, **values):
        """Return the difference between the left- and right-hand sides.

        Parameters
        ----------
        **values :
            Keyword arguments specifying values of variables and parameters.
            For all variables and all parameters that do not have their
            default values, their values should be specified.
            For parameters that have their default values and whose values
            are not specified in `**values`, their values will be considered
            to be their default values.

        Returns
        -------
        float
            The difference between the left and right hand sides of
            the mathematical expression of the constraint, ``lhs - rhs``.

        Raises
        ------
        Exception
            If the constraint is not expressed by a closed form expression.

        """
        if self.is_closed_form:
            if self._eval is None:
                self._set_functions_from_expr()
            return float(self._eval(**values))
        else:
            raise Exception("The constraint is not expressed "
                            "by a closed form expression.")

    def to_dict(self, export_vars=True, export_params=True):
        """Export the constraint object into a serializable dictionary.

        This serializing method is available only if the constraints is
        expressed by a closed form expression.

        Parameters
        ----------
        export_vars : bool, default=True
            If `False`, the information of variables is not exported.
        export_params : bool, default=True
            If `False`, the information of parameters is not exported.

        Returns
        -------
        dict
            A serializable dictionary of the constraint object.

        Raises
        ------
        Exception
            If the constraint is not expressed by a closed form expression.

        See Also
        --------
        Constraint.from_dict

        """
        if self.is_closed_form:
            lhs = self.expr.lhs.expand()
            if isinstance(lhs, Add):
                lhs_repr = 'Add({})'.format(
                    ', '.join(map(repr, lhs.args))
                )
            else:
                lhs_repr = repr(lhs)
            rhs = self.expr.rhs.expand()
            if isinstance(rhs, Add):
                rhs_repr = 'Add({})'.format(
                    ', '.join(map(repr, rhs.args))
                )
            else:
                rhs_repr = repr(rhs)
            if isinstance(self.expr, Equality):
                expr_repr = f'Eq({lhs_repr}, {rhs_repr})'
            elif isinstance(self.expr, GreaterThan):
                expr_repr = f'{lhs_repr} >= {rhs_repr}'
            elif isinstance(self.expr, LessThan):
                expr_repr = f'{lhs_repr} <= {rhs_repr}'
            else:
                raise ValueError(
                    f'Unsupported relational expression: {self.expr}'
                )
            d = {
                'name': self.name,
                'expr': expr_repr,
                'tol': self.tol
            }
            if export_vars:
                d.update(
                    {'vars': [v.to_dict() for v in self.vars]}
                )
            if export_params:
                d.update(
                    {'params': [p.to_dict() for p in self.params]}
                )
            return d
        else:
            raise Exception("The constraint is not expressed "
                            "by a closed form expression.")

    @classmethod
    def from_dict(cls, d, vars=None, params=None):
        """Create a constraint object from a serializable dictionary.

        Parameters
        ----------
        d : dict
            A serializable dictionary in the format of ::

                {
                    'name': name,
                    'vars': vars,
                    'params': params,
                    'expr': expr,
                    'tol': tol
                }

            where `name` is a `str`, `vars` is a `list` of serializable
            dictionaries of `Variable` objects, `params` is a list of
            serializable dictionaries of `Parameter` objects, `expr` is
            a string representation of the mathematical expression of
            the constraint, and `tol` is a tolerance parameter used
            for checking if the constraint is satisfied by a given set of
            values of variables and parameters. The item ``d['vars']``
            is optional, but either ``d['vars']`` or `vars` parameter
            must be specified. Likewise, the item ``d['params']`` is
            optional, but either ``d['params']`` or `params` parameter
            must be specified.
        vars : list [Variable]
            A list of variables. If ``d['vars']`` is not specified,
            this parameter must be specified.
        params : list [Parameter]
            A list of parameters. If ``d['params']`` is not specified,
            this parameter must be specified.

        Returns
        -------
        Constraint

        See Also
        --------
        Constraint.to_dict

        """
        if vars is None:
            vars = [Variable.from_dict(d_v) for d_v in d['vars']]
        if params is None:
            params = [Parameter.from_dict(d_p) for d_p in d['params']]
        symbol_namespace = {s.name: s for s in chain(vars, params)}
        expr = sympify(
            d['expr'],
            locals={**symbol_namespace, **functions.sympy_func}
        )
        unspecified_symbols = expr.atoms(Symbol)-{*vars, *params}
        if len(unspecified_symbols) > 0:
            raise ValueError(
                "The expression contains symbols which are "
                "neither `Variable` nor `Parameter`: " +
                str(unspecified_symbols)[1:-1] + ". "
                "Make sure that all variables and parameters "
                "are properly specified."
            )
        return cls(name=d['name'], expr=expr, tol=d['tol'])
