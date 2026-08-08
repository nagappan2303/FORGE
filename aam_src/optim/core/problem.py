"""The module of the `Problem` class.

"""

from .variable import Variable
from .parameter import Parameter
from .constraint import Constraint
from .objective import Objective


class Problem():
    """Optimization problems.

    Parameters
    ----------
    name : str, optional
        The name of the optimization problem.
    variables : list [Variable], default=[]
        The list of the variables in the optimization problem.
    parameters : list [Parameter], default=[]
        The list of the parameters in the optimization problem.
    constraints : list [Constraint], default=[]
        The list of the constraints in the optimization problem.
    objectives : list [Objective], default=[]
        The list of the objectives in the optimization problem.

    Attributes
    ----------
    name : str
        The name of the optimization problem.
    variables : list [Variable]
        The list of the variables in the optimization problem.
    parameters : list [Parameter]
        The list of the parameters in the optimization problem.
    constraints : list [Constraint]
        The list of the constraints in the optimization problem.
    objectives : list [Objective]
        The list of the objectives in the optimization problem.

    """

    def __init__(self, name=None, variables=None, parameters=None,
                 constraints=None, objectives=None):
        self.name = name
        self.variables = variables
        self.parameters = parameters
        self.constraints = constraints
        self.objectives = objectives

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def variables(self):
        return self._vars

    @variables.setter
    def variables(self, vars):
        if vars is None:
            vars = []
        self._vars = vars

    @property
    def parameters(self):
        return self._params

    @parameters.setter
    def parameters(self, params):
        if params is None:
            params = []
        self._params = params

    @property
    def constraints(self):
        return self._constrs

    @constraints.setter
    def constraints(self, constrs):
        if constrs is None:
            constrs = []
        self._constrs = constrs

    @property
    def objectives(self):
        return self._objs

    @objectives.setter
    def objectives(self, objs):
        if objs is None:
            objs = []
        self._objs = objs

    # aliases
    vars = variables
    params = parameters
    constrs = constraints
    objs = objectives

    @property
    def has_parameters(self):
        """Whether the problem has parameters.

        """
        return (len(self.params) >= 1)

    @property
    def is_constrained(self):
        """Whether the problem is constrained.

        """
        return (len(self.constrs) >= 1)

    @property
    def is_constraint_satisfaction_problem(self):
        """Whether the problem is a constraint satisfaction problem.

        """
        return (len(self.objs) == 0)

    @property
    def is_single_objective(self):
        """Whether the problem is single-objective.

        """
        return (len(self.objs) == 1)

    @property
    def is_multi_objective(self):
        """Whether the problem is multi-objective.

        """
        return (len(self.objs) >= 2)

    @property
    def is_closed_form(self):
        """Whether the problem is modeled by closed form expressions.

        """
        return (all([c.is_closed_form for c in self.constrs])
                and all([o.is_closed_form for o in self.objs]))

    def evaluate(self, **values):
        """Evaluate a given solution under a given parameters values setting.

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
        dict
           A dictionary in the format of ::

               {
                   'constraints': {constraint_name: is_satisfied},
                   'objectives': {objective_name: objective_value}
               }

            where `constraint_name` and `is_satisfied` are the name of
            a constraint and the boolean value indicating whether the
            constraint is satisfied by the given set of values of
            variables and parameters, respectively, and `objective_name`
            and `objective_value` are the name of an objective and
            the float value of the objective function for the given set
            of values of variables and parameters, respectively.

        """
        return {
            'constraints': {
                c.name: c.is_satisfied(**values)
                for c in self.constrs
            },
            'objectives': {
                o.name: o.evaluate(**values)
                for o in self.objs
            }
        }

    def to_dict(self):
        """Export the problem into a serializable dictionary.

        This serializing method is available only if the problem is
        modeled by closed form expressions.

        Returns
        -------
        dict
            A serializable dictionary of the problem object.

        Raises
        ------
        Exception
            If the problem is not modeled by closed form expressions.

        Notes
        -----
        The serializable dictionaries of constraints and objectives are
        exported by `Constraint.to_dict()` and `Objective.to_dict()`
        with ``export_vars=False`` and ``export_params=False``
        for reducing the size of exported dictionaries.

        See Also
        --------
        Problem.from_dict

        """
        if self.is_closed_form:
            return {
                'name': self.name,
                'variables': [
                    v.to_dict()
                    for v in self.vars
                ],
                'parameters': [
                    p.to_dict()
                    for p in self.params
                ],
                'constraints': [
                    c.to_dict(export_vars=False, export_params=False)
                    for c in self.constrs
                ],
                'objectives': [
                    o.to_dict(export_vars=False, export_params=False)
                    for o in self.objs
                ]
            }
        else:
            raise Exception("The problem is not modeled "
                            "by closed form expressions.")

    @classmethod
    def from_dict(cls, d):
        """Create a problem object from a serializable dictionary.

        Parameters
        ----------
        d : dict
            A serializable dictionary of the format of ::

                {
                    'name': name,
                    'variables': variables,
                    'parameters': parameters
                    'constraints': constraints,
                    'objectives': objectives
                }

            where `name` is a `str`, `variables` is a `list` of
            serializable dictionaries of `Variable` objects,
            `parameters` is a `list` of serializable dictionaries
            of `Parameter` objects, `constraints` is a `list` of
            serializable dictionaries of `Constraint` objects,
            and `objectives` is a `list` of serializable dictionaries
            of `Objective` objects. For the format of serializable
            dictionaries of `Variable`, `Parameter`, `Constraint`,
            and `Objective` objects, see `Variable.from_dict`,
            `Parameter.from_dict`, `Constraint.from_dict`,
            and `Objective.from_dict`, respectively.

        Returns
        -------
        Problem

        See Also
        --------
        Problem.to_dict
        Variable.from_dict
        Parameter.from_dict
        Constraint.from_dict
        Objective.from_dict

        """
        vars = [
            Variable.from_dict(v_d)
            for v_d in d['variables']
        ]
        params = [
            Parameter.from_dict(p_d)
            for p_d in d['parameters']
        ]
        constrs = [
            Constraint.from_dict(c_d, vars=vars, params=params)
            for c_d in d['constraints']
        ]
        objs = [
            Objective.from_dict(o_d, vars=vars, params=params)
            for o_d in d['objectives']
        ]
        return cls(name=d['name'], variables=vars, parameters=params,
                   constraints=constrs, objectives=objs)
