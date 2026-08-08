from functools import partial

import sympy

from .variable import Variable

# For the details of how to implement a subclass of `sympy.Function`,
# see https://docs.sympy.org/latest/guides/custom-functions.html


class L0(sympy.Function):
    """The 0-"norm" of a vector.

    The 0-"norm" is the number of non-zero components of a vector.

    Parameters
    ----------
    x : Number or sympy.Expr or Iterable or sympy.MatrixBase
        A vector. If `x` is a number or `sympy.Expr`, `x` is regarded
        as a vector with one component. If `x` is iterable, `x` will be
        automatically converted to a `sympy.Matrix` object.

    Attributes
    ----------
    arg : sympy.Expr or sympy.MatrixBase
        The argument of the 0-"norm" function. If the argument is
        a vector with one component, return it as a scalar expression.

    """

    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, x):
        if isinstance(x, sympy.MatrixBase):
            if x.shape[0] > 1 and x.shape[1] > 1:
                raise ValueError(f"{x} is not a vector.")
            if all(isinstance(x_i, sympy.Number) for x_i in x):
                return sum(
                    [sympy.S.Zero if x_i == 0 else sympy.S.One
                     for x_i in x]
                )
        elif hasattr(x, "__iter__"):
            return L0(sympy.Matrix(x))
        else:
            return L0(sympy.Matrix([x]))

    def doit(self, **hints):
        x = self.args[0]
        if hints.get('deep', False):
            x = sympy.Matrix([x_i.doit(**hints) for x_i in x])
        if all(isinstance(x_i, sympy.Number) for x_i in x):
            return sum(
                [sympy.S.Zero if x_i == 0 else sympy.S.One
                 for x_i in x]
            )
        if x.shape == (1, 1):  # x is a vector with one component
            x_ = x[0]
            if isinstance(x_, Variable):
                if x_.lb is not None and x_.lb > 0:
                    return sympy.S.One
                elif x_.ub is not None and x_.ub < 0:
                    return sympy.S.One
            else:
                if x_.is_zero:
                    return sympy.S.Zero
                elif x_.is_positive or x_.is_negative:
                    return sympy.S.One
        return L0(x)

    def _eval_rewrite_as_sign(self, x, **hints):
        return sum([(sympy.sign(x_i))**2 for x_i in x])

    def _eval_rewrite_as_Heaviside(self, x, **hints):
        theta = partial(sympy.Heaviside, H0=0)
        expr = 0
        for x_i in x:
            if x_i.is_nonnegative:
                expr += theta(x_i)
            elif x_i.is_nonpositive:
                expr += theta(-x_i)
            else:
                expr = theta(x_i) + theta(-x_i)
        return expr

    def _eval_expand_func(self, **hints):
        x = self.args[0]
        if hints.get('deep', False):
            return sum([L0(x_i.expand(**hints)) for x_i in x])
        else:
            return sum([L0(x_i) for x_i in x])

    @classmethod
    def math_func(cls, x):
        if hasattr(x, "__iter__"):
            return sum([cls.math_func(x_i) for x_i in x])
        else:
            if x == 0:
                return 0
            else:
                return 1

    @property
    def arg(self):
        x = self.args[0]
        if x.shape == (1, 1):
            return x[0]
        else:
            return x

    def _latex(self, printer):
        x = self.args[0]
        _x = printer._print(x)
        return r'\left \| %s \right \|_0' % (_x)


l0 = L0


class L1(sympy.Function):
    """The 1-norm of a vector.

    The 1-norm is the summation of the absolute values of the components
    of a vector.

    Parameters
    ----------
    x : Number or sympy.Expr or Iterable or sympy.MatrixBase
        A vector. If `x` is a number or `sympy.Expr`, `x` is regarded
        as a vector with one component. If `x` is iterable, `x` will be
        automatically converted to a `sympy.Matrix` object.

    Attributes
    ----------
    arg : sympy.Expr or sympy.MatrixBase
        The argument of the 1-norm function. If the argument is
        a vector with one component, return it as a scalar expression.

    """

    is_nonnegative = True

    @classmethod
    def eval(cls, x):
        if isinstance(x, sympy.MatrixBase):
            if x.shape[0] > 1 and x.shape[1] > 1:
                raise ValueError(f"{x} is not a vector.")
            if all(isinstance(x_i, sympy.Number) for x_i in x):
                return sum([sympy.Abs(x_i) for x_i in x])
        elif hasattr(x, "__iter__"):
            return L1(sympy.Matrix(x))
        else:
            return L1(sympy.Matrix([x]))

    def doit(self, **hints):
        x = self.args[0]
        if hints.get('deep', False):
            x = sympy.Matrix([x_i.doit(**hints) for x_i in x])
        if all(isinstance(x_i, sympy.Number) for x_i in x):
            return sum([sympy.Abs(x_i) for x_i in x])
        if x.shape == (1, 1):  # x is a vector with one component
            x_ = x[0]
            if isinstance(x_, Variable):
                if x_.lb is not None and x_.lb >= 0:
                    return x_
                elif x_.ub is not None and x_.ub <= 0:
                    return -x_
            else:
                if x_.is_zero:
                    return sympy.S.Zero
                elif x_.is_nonnegative:
                    return x_
                elif x_.is_nonpositive:
                    return -x_
        return L1(x)

    def _eval_rewrite_as_Abs(self, x, **hints):
        return sum([sympy.Abs(x_i) for x_i in x])

    def _eval_expand_func(self, **hints):
        x = self.args[0]
        if hints.get('deep', False):
            return sum([L1(x_i.expand(**hints)) for x_i in x])
        else:
            return sum([L1(x_i) for x_i in x])

    @classmethod
    def math_func(cls, x):
        if hasattr(x, "__iter__"):
            return sum([cls.math_func(x_i) for x_i in x])
        else:
            return abs(x)

    @property
    def arg(self):
        x = self.args[0]
        if x.shape == (1, 1):
            return x[0]
        else:
            return x

    def _latex(self, printer):
        x = self.args[0]
        _x = printer._print(x)
        return r'\left \| %s \right \|_1' % (_x)


l1 = L1


# Dictionary of the sympy functions defined in this module.
# This is used for translating strings into expressions by
# `sympy.sympify` in the `Constraint.from_dict()` and
# `Objective.from_dict()`.
sympy_func = {'L0': L0, 'l0': l0, 'L1': L1, 'l1': l1}

# Dictionary of the functions, defined using only Python built-in
# functions and the `math` module functions, that correspond to
# the sympy functions defined in this module. These functions are
# used in `Constraint._set_function()` and `Objective._set_function()`.
math_func = {name: f.math_func
             for name, f in sympy_func.items()}
