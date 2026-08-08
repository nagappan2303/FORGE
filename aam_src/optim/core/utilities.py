from collections.abc import Callable
from functools import lru_cache
import inspect
from itertools import chain
from operator import itemgetter
from typing import List, Dict, Any, Optional

import sympy

from .variable import Variable
from .parameter import Parameter
from . import functions


def create_args_filtering_wrapper(
        func: Callable,
        default_values: Optional[Dict[str, Any]] = None
) -> Callable:
    """Return a wrapper function that calls `func` with filtered arguments.

    Parameters
    ----------
    func
        The Python function to be wrapped. Functions that have
        variable-length arguments and variable-length keyword
        arguments are not supported.
    default_values
        A dictionary of arguments with their default values.
        The default values defined in the original function `func`
        will be ignored and overridden by those specified in
        `default_values`.

    Returns
    -------
    wrapper_func
        A Python function that (1) accepts arbitrary keyword arguments,
        (2) filters the arguments based on the original function's
        arguments, (3) sets default values for arguments whose values
        are not provided, and (4) calls the original function with
        the filtered arguments.

    """
    if default_values is None:
        default_values = {}

    sig = inspect.signature(func)
    args_names = [
        k for k, v in sig.parameters.items()
        if v.kind in (v.POSITIONAL_ONLY, v.POSITIONAL_OR_KEYWORD)]
    kwargs_names = [
        k for k, v in sig.parameters.items()
        if v.kind == v.KEYWORD_ONLY
    ]

    if len(args_names) == 0:
        def args_filter(d): return []
    elif len(args_names) == 1:
        arg_name = args_names[0]
        def args_filter(d): return [d[arg_name]]
    else:
        args_filter = itemgetter(*args_names)

    if len(kwargs_names) == 0:

        def wrapper_func(**kwargs):
            for item in default_values.items():
                kwargs.setdefault(*item)
            filtered_args = args_filter(kwargs)
            return func(*filtered_args)

    else:

        def wrapper_func(**kwargs):
            for item in default_values.items():
                kwargs.setdefault(*item)
            filtered_args = args_filter(kwargs)
            filtered_kwargs = {k: kwargs[k] for k in kwargs_names}
            return func(*filtered_args, **filtered_kwargs)

    return wrapper_func


def lambdify(
        expr: sympy.Expr,
        vars: Optional[List[Variable]] = None,
        params: Optional[List[Parameter]] = None,
        func_name: Optional[str] = None
) -> Callable:
    """Convert a SymPy expression of variables and parameters into a function.

    Parameters
    ----------
    expr
        A SymPy expression of variables and parameters.
    vars
        A list of variables. If not specified, `vars` will be the list of
        all `Variable` objects in `expr`.
    params
        A list of parameters. If not specified, `params` will be the list of
        all `Parameter` objects in `expr`.
    func_name
        The name of the function to be generated.

    Returns
    -------
    func
        A Python function that evaluates the value of the SymPy expression
        `expr` with given values of variables and parameters.

    """
    if vars is None:
        vars = list(expr.atoms(Variable))
    if params is None:
        params = list(expr.atoms(Parameter))
    if func_name is None:
        func_name = "_lambdify_generated_function"

    func_str = \
        "def {func_name}({args}): return {expr}".format(
            func_name=func_name,
            args=", ".join(str(x) for x in chain(vars, params)),
            expr=_expr_to_str(expr)
        )

    func_locals = {}
    exec(func_str, _namespace(), func_locals)

    return func_locals[func_name]


def _expr_to_str(expr: sympy.Expr) -> str:
    """Convert a Sympy expression into a Python code string.

    Notes
    -----
    - Evaluating a code string of the summation of many terms with
      the + operator may cause `RecursionError` due to exceeding
      the recursion depth limit. Thus, this function writes
      the summation using the built-in `sum()` function.
    - SymPy's `PythonCodePrinter` which prints code strings of SymPy
      expressions is slow. In mathematical optimization, polynomial
      expressions are often used. Because writing code strings of
      polynomial expressions is easy to implement, this function
      implements it in a recursive manner.
    - For writing code strings of non-polynomial expressions,
      this function employs SymPy's `NumPyPrinter`.

    """
    if isinstance(expr, (sympy.Symbol, sympy.Number)):
        return str(expr)
    if isinstance(expr, sympy.Add):
        return "sum([{}])".format(
            ", ".join(map(_expr_to_str, expr.args))
        )
    if isinstance(expr, sympy.Mul):
        return "*".join(map(_expr_to_str, expr.args))
    if isinstance(expr, sympy.Pow):
        return "**".join(map(_expr_to_str, expr.args))
    return _numpy_printer.doprint(expr)


_numpy_printer = sympy.printing.numpy.NumPyPrinter(
        {"fully_qualified_modules": False,
         "inline": True,
         "allow_unknown_functions": True,
         "user_functions": {k: k for k in functions.math_func}}
    )


@lru_cache
def _namespace() -> Dict[str, Any]:
    """Return the namespace dictionary the `exec` command in `lambdify` uses.

    """
    namespace = {}
    for command in ["import numpy", "from numpy import *",
                    "from numpy.linalg import *"]:
        exec(command, {}, namespace)
    namespace["Heaviside"] = "heaviside"
    namespace["I"] = 1j
    namespace["Abs"] = abs
    namespace.update(functions.math_func)
    return namespace
