"""The module of the `Parameter` class.

"""

from math import floor, ceil

from sympy import Symbol, S, cacheit


class Parameter(Symbol):
    """Parameters in optimization problems.

    `Parameter` is a subclass of `sympy.Symbol`, thus you can model
    constraints and objective functions by mathematical expressions
    including `Parameter` objects, such as ``param1*var1**2 + param2*var2``.
    `Parameter` objects are each unique, even if they have
    the same name.

    Parameters
    ----------
    name : str
        The name of the parameter. The valid characters for `name` are
        the same as those of identifiers in Python: the uppercase and
        lowercase letters `'A'` through `'Z'`, the underscore `'_'` and,
        except for the first character, the digits `'0'` through `'9'`.
        When modeling an optimization problem, different parameters of
        the same problem need to have different names.
    type : type
        The type of the parameter, `float` or `int`.
    lb : float or int, optional
        The lower bound of the parameter range.
        The default is `None` meaning negative infinity.
    ub : float or int, optional
        The upper bound of the parameter range.
        The default is `None` meaning positive infinity.
    default : float or int, optional
        The default value of the parameter.

    Attributes
    ----------
    name : str
        The name of the parameter.
    type : type
        The type of the parameter.
    lb : float, int, or None
        The lower bound of the parameter range.
        If `lb` is `None`, the lower bound is negative infinity.
    ub : float, int, or None
        The upper bound of the parameter range.
        If `ub` is `None`, the upper bound is positive infinity.
    default : float, int, or None
        The default value of the parameter.

    """
    _count = 0

    def __new__(cls, name, type, lb=None, ub=None, default=None):
        if type is float:
            self = super().__xnew__(cls, name, real=True)
            self._type = type
        elif type is int:
            self = super().__xnew__(cls, name, integer=True)
            self._type = type
        else:
            raise ValueError("The parameter type must be"
                             "`float` or `int`.")
        self._index = cls._count
        cls._count += 1
        self.lb = lb
        self.ub = ub
        self.default = default
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
                self._lb = ceil(lb)  # e.g. lb = -2.1 -> self.lb = -2 (int)
            elif self.type is float:
                self._lb = float(lb)  # e.g. lb = -2019 -> self.lb = -2019.0
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
                self._ub = floor(ub)  # e.g. ub = 2.2 -> self.ub = 2 (int)
            elif self.type is float:
                self._ub = float(ub)  # e.g. ub = 1989 -> self.ub = 1989.0
            else:
                self._ub = None
        elif ub is None:
            self._ub = None
        else:
            raise TypeError("The upper bound must be "
                            "`float`, `int`, or `None`.")

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, value):
        if isinstance(value, self.type):
            if (self.lb is not None) and (value < self.lb):
                raise ValueError(
                    f"The default value of '{self.name}' should be "
                    f"greater than the lower bound ({self.lb})."
                )
            if (self.ub is not None) and (value > self.ub):
                raise ValueError(
                    f"The default value of '{self.name}' should be "
                    f"less than the upper bound ({self.ub})."
                )
            self._default = value
        elif value is None:
            self._default = None
        else:
            raise TypeError(f"The default value of '{self.name}' "
                            f"should be {self.type} or `None`.")

    def fix_value(self, value):
        """Fix the value of the parameter.

        Parameters
        ----------
        value : float or int
            The parameter value will be fixed to `value`.

        Raises
        ------
        TypeError
            If the type of `value` is different from the parameter type.
        ValueError
            If the value of `value` is not in the parameter range.

        """
        if not isinstance(value, self.type):
            raise TypeError(f"The argument type ({type(value)}) "
                            f"is not the parameter type ({self.type}).")
        if (self.lb is not None) and (value < self.lb):
            raise ValueError(f"The argument value ({value}) "
                             f"is less than the lower bound ({self.lb}).")
        if (self.ub is not None) and (value > self.ub):
            raise ValueError(f"The argument value ({value}) "
                             f"is greater than the upper bound ({self.ub}).")
        self._is_fixed = True
        self._fixed_value = value

    def unfix_value(self):
        """Unfix the value of the parameter.

        """
        self._is_fixed = False
        self._fixed_value = None

    @property
    def is_fixed(self):
        """Whether the parameter's value is fixed.

        """
        return self._is_fixed

    @property
    def fixed_value(self):
        """The fixed value of the parameter.

        """
        return self._fixed_value

    @property
    def is_integer(self):
        """Whether the parameter is an integer parameter.

        If the parameter is an integer parameter, that is, the parameter
        can have only integer values, `is_integer` is `True`.
        Otherwise, `is_integer` is `None`, which means that the value of
        the parameter is not restricted to integers.

        Notes
        -----
        This property equals to `sympy.Symbol.is_integer` property.

        See also
        https://docs.sympy.org/latest/modules/core.html#module-sympy.core.assumptions

        """
        return super().is_integer

    def to_dict(self):
        """Export the parameter object into a serializable dictionary.

        Returns
        -------
        dict
            A serializable dictionary of the parameter object.

        See Also
        --------
        Parameter.from_dict

        """
        return {
            'name': self.name,
            'type': self.type.__name__,  # type object -> type name
            'lb': self.lb,
            'ub': self.ub,
            'default': self.default,
            'is_fixed': self.is_fixed,
            'fixed_value': self.fixed_value
        }

    @classmethod
    def from_dict(cls, d):
        """Create a parameter object from a serializable dictionary.

        Parameters
        ----------
        d : dict
            A serializable dictionary in the format of ::

                {
                    'name': name,
                    'type': type_name
                    'lb': lb,
                    'ub': ub,
                    'default': default,
                    'is_fixed': is_fixed,
                    'fixed_value': fixed_value,
                }

            where `name` and `type_name` are `str`,
            `lb`, `ub`, `default`, and `fixed_value` are
            `float`, `int`, or `None`, and `is_fixed` is `bool`.
            The items 'lb', 'ub', 'default', 'is_fixed',
            and 'fixed_value' are optional.

        Returns
        -------
        Parameter

        See Also
        --------
        Parameter.to_dict

        """
        name = d['name']
        type = eval(d['type'])  # type name -> type object
        lb = d.get('lb', None)
        ub = d.get('ub', None)
        default = d.get('default', None)
        param = cls(name=name, type=type,
                    lb=lb, ub=ub, default=default)
        if d.get('is_fixed', False) and ('fixed_value' in d):
            param.fix_value(d['fixed_value'])
        return param

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
