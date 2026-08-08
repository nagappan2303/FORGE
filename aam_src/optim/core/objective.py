"""The module of the `Objective` class.

"""

from itertools import chain

from sympy import Symbol, Expr, Add, sympify

from .variable import Variable
from .parameter import Parameter
from .utilities import lambdify, create_args_filtering_wrapper
from . import functions


class Objective():
    """Objective functions in optimization problems.

    Parameters
    ----------
    name : str
        The name of the objective.
    vars : list [Variable], optional
        The list of the variables of the objective function. If `expr` is
        specified, `vars` attribute will be automatically computed from
        `expr`. Otherwise, `vars` should be explicitly specified.
    params : list [Parameter], optional
        The list of the parameters of the objective function. If `expr` is
        specified, `params` attribute will be automatically computed from
        `expr`. Otherwise, `params` should be explicitly specified.
    func : function, optional
        The objective function which accepts values of variables and
        parameters as its arguments, e.g., ``func(x=0, y=1, a=2, b=-1)``.
    expr : sympy.Expr, optional
        The mathematical expression of the objective function.
    type : {'min', 'max'}, default='min'
        The type of the objective representing the direction of
        the optimization, ``'min'`` or ``'max'``. The default is ``'min'``.

    Other Parameters
    ----------------
    lazy : bool, default=True
        If true, the creation of an internal function for the `evaluate()`
        method based on a given SymPy expression will be delayed until
        the method is called. Because the function creation takes a bit
        long time, this lazy evaluation option saves computation time
        when the `evaluate()` method is never called.

    Attributes
    ----------
    name : str
        The name of the objective.
    vars : list [Variable]
        The list of the variables of the objective function.
    params : list [Parameter]
        The list of the parameters of the objective function.
    expr : sympy.Expr or None
        The mathematical expression of the objective function.
        If the objective function is not modeled by a mathematical
        expression, `expr` is `None`.
    type : {'min', 'max'}
        The type of the objective representing the direction of
        the optimization, ``'min'`` or ``'max'``.

    """

    def __init__(self, name, vars=None, params=None, func=None,
                 expr=None, type='min', lazy=True):

        self._name = name
        self._type = type

        # If the objective is defined by a mathematical expression:
        if expr is not None:
            if not isinstance(expr, Expr):
                raise TypeError("The `expr` argument must be a `Expr`.")
            self._expr = expr
            self._vars = list(expr.atoms(Variable))
            self._params = list(expr.atoms(Parameter))
            if lazy:
                self._func = None
            else:
                self._set_function_from_expr()

        # If the objective is defined by a black-box function:
        elif func is not None:
            if not callable(func):
                raise TypeError("The `func` argument must be a `function`.")
            self._expr = None
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

        # If the objective is not defined:
        else:
            def constant(**v): return 0.0
            self._func = constant
            self._expr = None
            self._vars = []
            self._params = []

    def _set_function_from_expr(self):
        """Set `self._func` based on `self.expr`.

        This method uses `optim.core.utilities.lambdify()` for
        creating the function; the function creation takes a bit
        long time.

        """
        self._func = create_args_filtering_wrapper(
            func=lambdify(expr=self._expr,
                          vars=self._vars,
                          params=self._params),
            default_values={p.name: p.default for p in self._params
                            if p.default is not None}
        )

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
    def type(self):
        return self._type

    @property
    def is_closed_form(self):
        """Whether the objective is modeled by a closed form expression.

        """
        return isinstance(self.expr, Expr)

    def evaluate(self, **values):
        """Evaluate the objective function.

        Parameters
        ---------
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

        """
        if self._func is None:
            self._set_function_from_expr()
        return float(self._func(**values))

    def to_dict(self, export_vars=True, export_params=True):
        """Export the objective object into a serializable dictionary.

        This serializing method is available only if the objective function
        is expressed by a closed form expression.

        Parameters
        ----------
        export_vars : bool, default=True
            If `False`, the information of variables is not exported.
        export_params : bool, default=True
            If `False`, the information of parameters is not exported.

        Returns
        -------
        dict
            A serializable dictionary of the objective object.

        Raises
        ------
        Exception
            If the objective is not expressed by a closed form expression.

        See Also
        --------
        Objective.from_dict

        """
        if self.is_closed_form:
            expr = self.expr.expand()
            if isinstance(expr, Add):
                expr_repr = 'Add({})'.format(
                    ', '.join(map(repr, expr.args))
                )
            else:
                expr_repr = repr(expr)
            d = {
                'name': self.name,
                'expr': expr_repr,
                'type': self.type
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
            raise Exception("The objective is not expressed "
                            "by a closed form expression.")

    @classmethod
    def from_dict(cls, d, vars=None, params=None):
        """Create an objective object from a serializable dictionary.

        Parameters
        ----------
        d : dict
            A serializable dictionary in the format of ::

                {
                    'name': name,
                    'vars': vars,
                    'params': params,
                    'expr': expr,
                    'type': type
                }

            where `name` is a `str`, `vars` is a `list` of serializable
            dictionaries of `Variable` objects, `params` is a list of
            serializable dictionaries of `Parameter` objects, `expr` is
            a string representation of the mathematical expression of
            an objective function, and `type` is ``'min'`` or ``'max'``.
            The item ``d['vars']`` is optional, but either ``d['vars']``
            or `vars` must be specified. Likewise, the item ``d['params']``
            is optional, but either ``d['params']`` or `params` must be
            specified.
        vars : list [Variable], optional
            A list of variables. If ``d['vars']`` is not specified,
            this parameter must be specified.
        params : list [Parameter], optional
            A list of parameters. If ``d['params']`` is not specified,
            this parameter must be specified.

        Returns
        -------
        Objective

        See Also
        --------
        Objective.to_dict

        """
        if vars is None:
            vars = [Variable.from_dict(v_d) for v_d in d['vars']]
        if params is None:
            params = [Parameter.from_dict(p_d) for p_d in d['params']]
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
        return cls(name=d['name'], expr=expr, type=d['type'])
