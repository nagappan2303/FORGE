"""The module of the `Variable` class.

"""

from math import floor, ceil

from sympy import Symbol, S, cacheit


class Variable(Symbol):
    """Variables in optimization problems.

    `Variable` is a subclass of `sympy.Symbol`, thus you can model
    constraints and objective functions by mathematical expressions
    of `Variable` objects, such as ``var1**2 + 3*var2``. Note that
    `Variable` objects are each unique, even if they have the same name.

    Parameters
    ----------
    name : str
        The name of the variable. The valid characters for `name` are
        the same as those of identifiers in Python: the uppercase and
        lowercase letters `'A'` through `'Z'`, the underscore `'_'` and,
        except for the first character, the digits `'0'` through `'9'`.
        When modeling an optimization problem, different variables of
        the same problem need to have different names.
    type : type
        The type of the variable, `float` or `int`.
    lb : float or int, optional
        The lower bound of the variable range.
        The default is `None` meaning negative infinity.
    ub : float or int, optional
        The upper bound of the variable range.
        The default is `None` meaning positive infinity.

    Attributes
    ----------
    name : str
        The name of the variable.
    type : type
        The type of the variable.
    lb : float, int, or None
        The lower bound of the variable range.
        If `lb` is `None`, the lower bound is negative infinity.
    ub : float, int, or None
        The upper bound of the variable range.
        If `ub` is `None`, the upper bound is positive infinity.

    """
    _count = 0

    def __new__(cls, name, type, lb=None, ub=None):
        if type is float:
            self = super().__xnew__(cls, name, real=True)
            self._type = type
        elif type is int:
            self = super().__xnew__(cls, name, integer=True)
            self._type = type
        else:
            raise ValueError("The variable type must be"
                             "`float` or `int`.")
        self._index = cls._count
        cls._count += 1
        self.lb = lb
        self.ub = ub
        self.unfix_value()
        return self

    @property
    def index(self):
        return self._index

    @property
    def type(self):
        return self._type

    @property
    def lb(self):
        return self._lb

    @lb.setter
    def lb(self, lb):
        if isinstance(lb, (float, int)):
            # check if `lb` <= `ub`
            ub = getattr(self, '_ub', None)
            if (ub is not None) and (lb > ub):
                raise ValueError("The lower bound must be less than "
                                 f"or equal to the upper bound ({ub}).")
            if self.type is int:
                self._lb = ceil(lb)  # e.g. lb = -4.19 -> self.lb = -4 (int)
            elif self.type is float:
                self._lb = float(lb)  # e.g. lb = -2021 -> self.lb = -2021.0
            else:
                self._lb = None
        elif lb is None:
            self._lb = None
        else:
            raise TypeError("The lower bound must be "
                            "`float`, `int`, or `None`.")

    @property
    def ub(self):
        return self._ub

    @ub.setter
    def ub(self, ub):
        if isinstance(ub, (float, int)):
            # check if `lb` <= `ub`
            lb = getattr(self, '_lb', None)
            if (lb is not None) and (lb > ub):
                raise ValueError("The upper bound must be greater than "
                                 f"or equal to the lower bound ({lb}).")
            if self.type is int:
                self._ub = floor(ub)  # e.g. ub = 8.23 -> self.ub = 8 (int)
            elif self.type is float:
                self._ub = float(ub)  # e.g. ub = 1990 -> self.ub = 1990.0
            else:
                self._ub = None
        elif ub is None:
            self._ub = None
        else:
            raise TypeError("The upper bound must be"
                            "`float`, `int`, or `None`.")

    def fix_value(self, value):
        """Fix the value of the variable.

        Parameters
        ----------
        value : float or int
            The variable value will be fixed to `value`.

        Raises
        ------
        TypeError
            If the type of `value` is different from the variable type.
        ValueError
            If the value of `value` is not in the variable range.

        """
        if not isinstance(value, self.type):
            raise TypeError(f"The argument type ({type(value)}) "
                            f"is not the variable type ({self.type}).")
        if (self.lb is not None) and (value < self.lb):
            raise ValueError(f"The argument value ({value}) "
                             f"is less than the lower bound ({self.lb}).")
        if (self.ub is not None) and (value > self.ub):
            raise ValueError(f"The argument value ({value}) "
                             f"is greater than the upper bound ({self.ub}).")
        self._is_fixed = True
        self._fixed_value = value

    def unfix_value(self):
        """Unfix the value of the variable.

        """
        self._is_fixed = False
        self._fixed_value = None

    @property
    def is_fixed(self):
        """Whether the variable's value is fixed.

        """
        return self._is_fixed

    @property
    def fixed_value(self):
        """The fixed value of the variable.

        """
        return self._fixed_value

    @property
    def is_integer(self):
        """Whether the variable is an integer variable.

        If the variable is an integer variable, that is, the variable
        can have only integer values, `is_integer` is `True`.
        Otherwise, `is_integer` is `None`, which means that the value of
        the variable is not restricted to integers.

        Notes
        -----
        This property equals to `sympy.Symbol.is_integer` property.

        See also
        https://docs.sympy.org/latest/modules/core.html#module-sympy.core.assumptions

        """
        return super().is_integer

    @property
    def is_binary(self):
        """Whether the variable is a binary variable.

        If the variable is a binary variable, that is, the variable
        can have only binary values (0 or 1), `is_binary` is `True`.
        Otherwise, `is_binary` is `None`, which means that the value of
        the variable is not restricted to {0, 1}.

        """
        if (self.type is int) and (self.lb == 0) and (self.ub == 1):
            return True

    def to_dict(self):
        """Export the variable object into a serializable dictionary.

        Returns
        -------
        dict
            A serializable dictionary of the variable object.

        See Also
        --------
        Variable.from_dict

        """
        return {
            'name': self.name,
            'type': self.type.__name__,  # type object -> type name
            'lb': self.lb,
            'ub': self.ub,
            'is_fixed': self.is_fixed,
            'fixed_value': self.fixed_value
        }

    @classmethod
    def from_dict(cls, d):
        """Create a variable object from a serializable dictionary.

        Parameters
        ----------
        d : dict
            A serializable dictionary in the format of ::

                {
                    'name': name,
                    'type': type_name
                    'lb': lb,
                    'ub': ub,
                    'is_fixed': is_fixed,
                    'fixed_value': fixed_value,
                }

            where `name` and `type_name` are `str`,
            `lb`, `ub`, and `fixed_value` are `float`, `int`, or `None`,
            and `is_fixed` is `bool`. The items 'lb', 'ub', 'is_fixed',
            and 'fixed_value' are optional.

        Returns
        -------
        Variable

        See Also
        --------
        Variable.to_dict

        """
        name = d['name']
        type = eval(d['type'])  # type name -> type object
        lb = d.get('lb', None)
        ub = d.get('ub', None)
        var = cls(name, type, lb, ub)
        if d.get('is_fixed', False) and ('fixed_value' in d):
            var.fix_value(d['fixed_value'])
        return var

    @cacheit
    def sort_key(self, order=None):
        """Return a sort key.

        Notes
        -----
        The code of this method was written referring to
        that of `sympy.Dummy`.

        """
        return (self.class_key(), (2, (self.name, self.index)),
                S.One.sort_key(), S.One)

    def _hashable_content(self):
        """Return a tuple of information about self for computing the hash.

        Notes
        -----
        The code of this method was written referring to
        that of `sympy.Dummy`.

        """
        return super()._hashable_content() + (self.index,)
