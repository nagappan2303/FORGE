"""A module for optimization result reporting.

"""

import numpy as np
from .problem import Problem


class SolutionSet():
    """Solution sets of optimization problems.

    Parameters
    ----------
    problem : optimization.Problem
        An optimization problem.
    solutions : numpy.ndarray, optional
        2D NumPy array where ``solutions[i, j]`` is the value of
        ``problem.variables[j]`` for the solution with index `i`.
    parameters : numpy.ndarray, optional
        2D NumPy array where ``parameter[i, j]`` is the value of
        ``problem.parameters[j]`` used for computing the solution
        with index `i`. If all solutions have the same set of values
        of parameters, you can specify the common set of values of
        parameters by 1D NumPy array where `j`-th item is the value
        of ``problem.parameters[j]``. For a problem which has parameters,
        `parameters` should be specified. If you specify `parameters`,
        you must also specify `solutions`.
    constraints : numpy.ndarray, optional
        2D NumPy array where ``constraints[i, j]`` is the boolean
        value indicating whether the solution with index `i` satisfies
        ``problem.constraints[j]``.
        If you specify `constraints`, you must also specify `solutions`.
        If you specify `solutions` but do not specify `constraints`,
        the `constraints` array will be automatically calculated
        from `solutions` and `problem` when `constraints` property
        is called (lazy evaluation).
    objectives : numpy.ndarray, optional
        2D NumPy array where ``objectives[i, j]`` is the float value
        of ``problem.objectives[j]`` for the solution with index `i`.
        If you specify `objectives`, you must also specify `solutions`.
        If you specify `solutions` but do not specify `objectives`,
        the `objectives` array will be automatically calculated
        from `solutions` and `problem` when `objectives` property
        is called (lazy evaluation).
    info : dict, optional
        Information of the solution set as a whole.
    children : list [SolutionSet], optional
        The list of child `SolutionSet` objects, which may be solution sets
        of subproblems, solution sets of mapped problems, etc.
    **appendix : dict [str, numpy.ndarray], optional
        Additional data for each solution. The keys are the data labels
        and the values are 1D NumPy arrays of the corresponding data.
        If you specify `appendix`, you must also specify `solutions`.

    Raises
    ------
    TypeError
        If an argument's type is wrong.
    ValueError
        If the shape of `solutions`, `parameters`, `constraints`,
        `objectives`, or `appendix` is inconsistent.

    Attributes
    ----------
    problem : optimization.Problem
        The optimization problem.
        The `problem` attribute can not be changed.
    solutions : numpy.ndarray
        2D NumPy array where ``solutions[i, j]`` is the value of
        ``problem.variables[j]`` for the solution with index `i`.
    parameters : numpy.ndarray
        2D NumPy array where ``parameters[i, j]`` is the value of
        ``problem.parameters[j]`` used for computing the solution
        with index `i`.
    constraints : numpy.ndarray
        2D NumPy array where ``constraints[i, j]`` is the boolean
        value indicating whether the solution with index `i` satisfies
        ``problem.constraints[j]``.
    objectives : numpy.ndarray
        2D NumPy array where ``objectives[i, j]`` is the float value
        of ``problem.objectives[j]`` for the solution with index `i`.
    info : dict
        Information of the solution set as a whole.
    children : list [SolutionSet]
        The list of child `SolutionSet` objects, which may be solution sets
        of subproblems, solution sets of mapped problems, etc.
    appendix : dict [str, numpy.ndarray]
        Additional data for each solution. The keys are the data labels
        and the values are 1D NumPy arrays of the corresponding data.

    """

    def __init__(self, problem, solutions=None, parameters=None,
                 constraints=None, objectives=None, info=None,
                 children=None, **appendix):
        self.problem = problem
        self.solutions = solutions
        self.parameters = parameters
        self.constraints = constraints
        self.objectives = objectives
        self.info = info
        self.children = children
        self.appendix = appendix

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, problem):
        if hasattr(self, '_problem'):
            raise Exception("`problem` can not be changed.")
        elif isinstance(problem, Problem):
            self._problem = problem
        else:
            raise TypeError("`problem` must be a `optimization.Problem`.")

    @property
    def solutions(self):
        return self._solutions

    @solutions.setter
    def solutions(self, solutions):
        if solutions is None:
            self._solutions =\
                np.empty([0, len(self.problem.variables)])
        elif isinstance(solutions, np.ndarray) and (solutions.ndim == 2):
            if solutions.shape[1] == len(self.problem.variables):
                self._solutions = solutions
            else:
                raise ValueError("The number of the columns of `solutions` "
                                 "must be the same as the number of "
                                 "variables of `problem`.")
        else:
            raise TypeError("`solutions` must be a 2D NumPy array.")

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if self.problem.has_parameters:
            if isinstance(parameters, np.ndarray):
                if parameters.ndim == 1:
                    if len(parameters) == len(self.problem.parameters):
                        self._parameters = \
                            np.tile(parameters, (len(self.solutions), 1))
                    else:
                        raise ValueError(
                            "The number of the items of `parameters` "
                            "must be the same as the number of "
                            "parameters of `problem`."
                        )
                elif parameters.ndim == 2:
                    if parameters.shape[1] == len(self.problem.parameters):
                        if parameters.shape[0] == self.solutions.shape[0]:
                            self._parameters = parameters
                        else:
                            raise ValueError(
                                "The number of the rows of `parameters` "
                                "must be the same as that of `solutions`."
                            )
                    else:
                        raise ValueError(
                            "The number of the columns of `parameters` "
                            "must be the same as the number of "
                            "parameters of `problem`."
                        )
                else:
                    raise TypeError(
                        "`parameters` should be a 1D or 2D NumPy array."
                    )
            elif parameters is None:
                if len(self.solutions) == 0:
                    self._parameters = \
                        np.empty((0, len(self.problem.parameters)))
                else:
                    raise TypeError(
                        "For a problem which has parameters, "
                        "`parameters` should be specified."
                    )
            else:
                raise TypeError(
                    "`parameters` should be a 1D or 2D NumPy array."
                )
        else:
            self._parameters = np.empty((len(self.solutions), 0))

    @property
    def constraints(self):
        if len(self._constraints) != len(self.solutions):
            # lazy evaluation of constraints satisfaction
            self._constraints = self._evaluate_constraints_satisfaction()
        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        if constraints is None:
            self._constraints =\
                np.empty([0, len(self.problem.constraints)], dtype='bool')
        elif isinstance(constraints, np.ndarray) and (constraints.ndim == 2):
            if constraints.shape[1] != len(self.problem.constraints):
                raise ValueError("The number of the columns of `constraints` "
                                 "must be the same as the number of "
                                 "constraints of `problem`.")
            elif constraints.shape[0] != self.solutions.shape[0]:
                raise ValueError("The number of the rows of `constraints` "
                                 "must be the same as that of `solutions`.")
            else:
                self._constraints = constraints
        else:
            raise TypeError("`constraints` must be a 2D NumPy array.")

    def _evaluate_constraints_satisfaction(self):
        """Evaluate constraints satisfaction.

        Returns
        -------
        sat : numpy.ndarray
            2D NumPy array where ``sat[i, j]`` is the Boolean value
            indicating whether the solution ``self.solutions[i]``
            satisfies ``self.problem.constraints[j]``.

        """
        variables = self.problem.variables
        parameters = self.problem.parameters
        constraints = self.problem.constraints
        sat = np.empty([len(self.solutions), len(constraints)], dtype='bool')
        if self.problem.has_parameters:
            for i, (s, pv) in enumerate(zip(self.solutions, self.parameters)):
                d = {v.name: s[k] for k, v in enumerate(variables)}
                d.update({p.name: pv[k] for k, p in enumerate(parameters)})
                sat[i] = [c.is_satisfied(**d) for c in constraints]
        else:
            for i, s in enumerate(self.solutions):
                d = {v.name: s[k] for k, v in enumerate(variables)}
                sat[i] = [c.is_satisfied(**d) for c in constraints]
        return sat

    @property
    def objectives(self):
        if len(self._objectives) != len(self.solutions):
            # lazy evaluation of objectives
            self._objectives = self._evaluate_objectives()
        return self._objectives

    @objectives.setter
    def objectives(self, objectives):
        if objectives is None:
            self._objectives =\
                np.empty([0, len(self.problem.objectives)])
        elif isinstance(objectives, np.ndarray) and (objectives.ndim == 2):
            if objectives.shape[1] != len(self.problem.objectives):
                raise ValueError("The number of the columns of `objectives` "
                                 "must be the same as the number of "
                                 "objectives of `problem`.")
            elif objectives.shape[0] != self.solutions.shape[0]:
                raise ValueError("The number of the rows of `objectives` "
                                 "must be the same as that of `solutions`.")
            else:
                self._objectives = objectives
        else:
            raise TypeError("`objectives` must be a 2D NumPy array.")

    def _evaluate_objectives(self):
        """Evaluate objective functions values.

        Returns
        -------
        obj_values : numpy.ndarray
            2D NumPy array where ``obj_values[i, j]`` is the float value
            of ``self.problem.objectives[j]`` for ``self.solutions[i]``.

        """
        variables = self.problem.variables
        parameters = self.problem.parameters
        objectives = self.problem.objectives
        obj_values = np.empty([len(self.solutions), len(objectives)])
        if self.problem.has_parameters:
            for i, (s, pv) in enumerate(zip(self.solutions, self.parameters)):
                d = {v.name: s[k] for k, v in enumerate(variables)}
                d.update({p.name: pv[k] for k, p in enumerate(parameters)})
                obj_values[i] = [o.evaluate(**d) for o in objectives]
        else:
            for i, s in enumerate(self.solutions):
                d = {v.name: s[k] for k, v in enumerate(variables)}
                obj_values[i] = [o.evaluate(**d) for o in objectives]
        return obj_values

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, info):
        if isinstance(info, dict):
            self._info = info
        elif info is None:
            self._info = dict()
        else:
            raise TypeError("`info` must be a `dict`.")

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children):
        if isinstance(children, list):
            self._children = children
        elif children is None:
            self._children = []
        else:
            raise TypeError("`children` must be a `list`.")

    @property
    def appendix(self):
        return self._appendix

    @appendix.setter
    def appendix(self, d):
        self._appendix = {}
        if isinstance(d, dict):
            for key, val in d.items():
                if isinstance(key, str):
                    if isinstance(val, np.ndarray) and val.ndim == 1:
                        if len(val) == len(self.solutions):
                            self._appendix[key] = val
                        else:
                            raise ValueError(
                                "The size of each appendix data array must be "
                                "the same as the number of rows of "
                                "`solutions`."
                            )
                    else:
                        raise TypeError(
                            "Each appendix data must be specified as "
                            f"a 1D NumPy array, but the appendix data '{key}' "
                            "is not a 1D NumPy array."
                        )
                else:
                    raise TypeError(
                        "The keywords of `appendix` must be strings."
                    )
        elif d is not None:
            raise TypeError("`appendix` must be a `dict`.")

    def feasible_solutions(self, returns=None):
        """Return feasible solutions in the solution set.

        Parameters
        ----------
        returns : set [str], optional
            A set of labels of data to be returned in addition to a feasible
            solutions array. For example, if ``returns={'objectives'}``,
            the array of the objective functions' values of the feasible
            solutions also will be returned. For appendix data, specify their
            labels, not `'appendix'`. As default, no additional data will be
            returned.

        Returns
        -------
        feasible_solutions : numpy.ndarray
            2D NumPy array of the feasible solutions in the solution set.
        feasible_solutions_data : dict [str, numpy.ndarray]
            The dictionary of data associated with the feasible solutions.
            The keys are data labels specified by `returns` parameter, and
            the values are 1D or 2D NumPy arrays of the data. As default,
            this is an empty dict.

        """
        index = np.all(self.constraints, axis=1)
        feasible_solutions = self.solutions[index]
        feasible_solutions_data = {}
        if returns is not None:
            if 'parameters' in returns:
                feasible_solutions_data.update(
                    parameters=self.parameters[index]
                )
                returns.discard('parameters')
            if 'constraints' in returns:
                feasible_solutions_data.update(
                    constraints=self.constraints[index]
                )
                returns.discard('constraints')
            if 'objectives' in returns:
                feasible_solutions_data.update(
                    objectives=self.objectives[index]
                )
                returns.discard('objectives')
            if len(returns) > 0:
                appendix = self.appendix
                feasible_solutions_data.update(
                    {label: appendix[label][index] for label in returns}
                )
        return feasible_solutions, feasible_solutions_data

    def best_solutions(self, returns=None):
        """Return best feasible solutions in the solution set.

        This method is not available if `self.problem` is multi-objective.

        Parameters
        ----------
        returns : set [str], optional
            A set of labels of data to be returned in addition to a best
            solutions array. For example, if ``returns={'objectives'}``,
            the array of the objective functions' values of the best
            solutions also will be returned. For appendix data, specify
            their labels, not `'appendix'`. As default, no additional data
            will be returned.

        Returns
        -------
        best_solutions : numpy.ndarray
            2D NumPy array of the best feasible solutions
            in the solution set.
        best_solutions_data : dict [str, numpy.ndarray]
            The dictionary of data associated with the best solutions.
            The keys are data labels specified by `returns` parameter, and
            the values are 1D or 2D NumPy arrays of the data. As default,
            this is an empty dict.

        Raises
        ------
        Exception
            If `self.problem` is multi-objective.

        """
        if self.problem.is_constraint_satisfaction_problem:
            return self.feasible_solutions(returns=returns)
        elif self.problem.is_single_objective:
            if returns is None:
                returns = set()
            f_solutions, f_data = \
                self.feasible_solutions(returns=(returns | {'objectives'}))
            if len(f_solutions) == 0:  # all solutions are infeasible
                return f_solutions, {label: f_data[label] for label in returns}
            else:  # there are feasible solutions
                if self.problem.objectives[0].type == 'min':
                    best_value = np.min(f_data['objectives'], axis=0)[0]
                elif self.problem.objectives[0].type == 'max':
                    best_value = np.max(f_data['objectives'], axis=0)[0]
                arg_best = np.where(f_data['objectives'] == best_value)[0]
                best_solutions = f_solutions[arg_best]
                best_solutions_data = {
                    label: f_data[label][arg_best] for label in returns
                }
                return best_solutions, best_solutions_data
        else:
            raise Exception("The `problem` is multi-objective.")

    def to_dict(self, exclude=None):
        """Export the solution set object into a serializable dictionary.

        Parameters
        ----------
        exclude : set [str], optional
            A set of labels of items you want to exclude from a dictionary
            to be returned. For example, if ``exclude={'problem'}``, the
            dictionary to be returned will not include the information of
            `problem`. As default, all information will be included in the
            dictionary to be returned. This argument will be passed also
            to the children's `to_dict` methods.

        Returns
        -------
        dict [str, any]
            A serializable dictionary of the solution set object.
            For the details of the format, see also the docstring of
            `SolutionSet.from_dict()` method.

        See Also
        --------
        SolutionSet.from_dict

        """
        if exclude is None:
            exclude = {}
        d = dict()
        if 'problem' not in exclude:
            d['problem'] = self.problem.to_dict()
        if 'solutions' not in exclude:
            d['solutions'] = self.solutions.tolist()
        if 'parameters' not in exclude:
            d['parameters'] = self.parameters.tolist()
        if 'constraints' not in exclude:
            d['constraints'] = self.constraints.tolist()
        if 'objectives' not in exclude:
            d['objectives'] = self.objectives.tolist()
        if 'info' not in exclude:
            d['info'] = self.info
        if 'children' not in exclude:
            d['children'] = [
                child.to_dict(exclude=exclude)
                for child in self.children
            ]
        if 'appendix' not in exclude:
            d['appendix'] = {
                k: v.tolist() for k, v in self.appendix.items()
            }
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a solution set object from a serializable dictionary.

        Parameters
        ----------
        d : dict
            A serializable dictionary in the format of ::

                {
                    'problem': problem,
                    'solutions': solutions,
                    'parameters': parameters,
                    'constraints': constraints,
                    'objectives': objectives,
                    'info': info,
                    'children': children,
                    'appendix': appendix
                }

            where `problem` is a serializable dictionary of a `Problem`
            object, `solutions`, `parameters`, `constraints`, and
            `objectives` are ``list [list]`` corresponding to 2D
            NumPy arrays of input parameters for the construction of
            a `SolutionSet` object, `info` is a `dict` of information
            of the solution set as a whole, `children` is a `list` of
            serializable dictionaries of children, `appendix` is
            a ``dict [str, list]`` whose keys are data labels and
            values are lists corresponding to 1D NumPy arrays of data.
            Each item, except for 'problem', can be omitted as appropriate.

        Returns
        -------
        SolutionSet

        See Also
        --------
        SolutionSet.to_dict
        Problem.from_dict

        """
        kwargs = {}
        kwargs['problem'] = Problem.from_dict(d['problem'])
        if 'solutions' in d:
            kwargs['solutions'] = np.array(d['solutions'])
        if 'parameters' in d:
            kwargs['parameters'] = np.array(d['parameters'])
        if 'constraints' in d:
            kwargs['constraints'] = np.array(d['constraints'])
        if 'objectives' in d:
            kwargs['objectives'] = np.array(d['objectives'])
        if 'info' in d:
            kwargs['info'] = d['info']
        if 'children' in d:
            kwargs['children'] = [
                cls.from_dict(child_dict) for child_dict in d['children']
            ]
        if 'appendix' in d:
            kwargs.update(
                {k: np.array(v) for k, v in d['appendix'].items()}
            )
        return cls(**kwargs)
