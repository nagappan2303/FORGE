from optim.core.problem import *
from optim.ising.samplers import NealSampler

from functools import partial
from math import ceil, log, exp, sqrt
from statistics import mean, stdev
import time
from scipy.special import zeta
import networkx as nx
import pandas as pd
from sympy import Add

class IndependentSetSampler():

    def __init__(self, sampler, G):
        self.sampler = sampler
        self.G = G
        self.qubo = self.to_qubo(G)
        self._num_success = {}

    @staticmethod
    def to_qubo(G):
        x = []
        v_to_i = {}
        for i, v in enumerate(G.nodes):
            x.append(Variable(name=f'x_{v}', type=int, lb=0, ub=1))
            v_to_i[v] = i
        obj = Objective(name='objective', vars=x,
                        expr=1*(-Add(*x)+2*Add(*(x[v_to_i[u]]*x[v_to_i[v]]
                                              for (u, v) in G.edges))),
                        type='min')
        Q = Problem(name='sample', variables=x, objectives=[obj])
        return Q

    def sample(self, threshold=0, **kwargs):

        num_success, num_total, p_success = \
            self._num_success.get(threshold, (0, 0, 1.0))

        solve = self.sampler.solve
        Q = self.qubo

        node_to_varindex = {i: vi for vi, i in enumerate(self.G.nodes)}

        def is_target(s):
            penalty = sum(s[node_to_varindex[i]]*s[node_to_varindex[j]]
                          for (i, j) in self.G.edges)
            if (penalty == 0) and (sum(s) >= threshold):
                return True
            else:
                return False

        while True:
            num_reads = ceil(1/p_success)
            samples = solve(Q, num_reads=num_reads, beta_schedule_type='linear', num_sweeps=1000)
            target_solutions = \
                samples.solutions[list(map(is_target, samples.solutions))]

            if len(target_solutions) > 0:
                target_sample = {v.name: target_solutions[0][j]
                                 for j, v in enumerate(Q.variables)}
                independent_set = frozenset(
                    node for node in self.G.nodes
                    if target_sample[f'x_{node}'] == 1
                )
                size = len(independent_set)
                num_total += num_reads
                num_success += len(target_solutions)
                p_success = num_success/num_total
                break

            num_total += num_reads
            p_success = max(num_success, 1)/num_total

        self._num_success[threshold] = (num_success, num_total, p_success)

        return independent_set, size
    
class MaxCliques:
    def __init__(self, G):
        """
        Initialize the class with the graph G and prepare other needed variables.
        
        Parameters
        ----------
        G : networkx.Graph
            Input graph to find the maximum cliques.
        """
        self.G = G

    def find_maximum_cliques_cp(self, time_budget_s=None):
        """
        Find all maximum cliques by integrating Carraghan's pruning into `networkx.find_cliques()`.

        Parameters
        ----------
        time_budget_s : float or None
            Optional soft wall-clock budget. Exact enumeration is exponential
            in the worst case (dense modular products with tens of thousands
            of edges run for hours); when the budget is exceeded the search
            aborts and returns (None, elapsed) so the caller can fall back to
            the stochastic SA enumerator.

        Returns
        -------
        max_cliques : list or None
            All maximum cliques, or None when the time budget was exceeded.
        execution_time : int
            Time taken to execute the clique-finding algorithm (in seconds).
        """
        start = time.perf_counter_ns()
        deadline = None if time_budget_s is None else start + int(time_budget_s * 1e9)
        _tick = 0
        cbc_size = 0
        cbc_list = []

        if len(self.G) == 0:
            return set(), 0

        adj = {u: {v for v in self.G[u] if v != u} for u in self.G}

        Q = []
        cand = set(self.G)
        subg = cand.copy()
        stack = []
        Q.append(None)

        u = max(subg, key=lambda u: len(cand & adj[u]))
        ext_u = cand - adj[u]

        try:
            while True:
                if deadline is not None:
                    _tick += 1
                    if (_tick & 0x3FF) == 0 and time.perf_counter_ns() > deadline:
                        return None, (time.perf_counter_ns() - start) * 1e-9
                if ext_u:
                    q = ext_u.pop()
                    cand.remove(q)
                    Q[-1] = q
                    if len(Q) + len(cand) >= cbc_size:
                        adj_q = adj[q]
                        subg_q = subg & adj_q
                        if not subg_q:
                            if len(Q) == cbc_size:
                                cbc_list.append(Q[:])
                            elif len(Q) > cbc_size:
                                cbc_size = len(Q)
                                cbc_list = [Q[:]]
                        else:
                            cand_q = cand & adj_q
                            if cand_q:
                                stack.append((subg, cand, ext_u))
                                Q.append(None)
                                subg = subg_q
                                cand = cand_q
                                u = max(subg, key=lambda u: len(cand & adj[u]))
                                ext_u = cand - adj[u]
                    else:
                        Q.pop()
                        subg, cand, ext_u = stack.pop()
                else:
                    Q.pop()
                    subg, cand, ext_u = stack.pop()
        except IndexError:
            pass

        end = time.perf_counter_ns()
        execution_time = (end - start)* 1e-9
        
        #return set(frozenset(clique) for clique in cbc_list), execution_time
        return list( list(clique) for clique in cbc_list), execution_time
    
    def find_maximum_cliques_sa(self):
        """
        Find all maximum cliques by simulated by integrating our enumeration approach`.
        
        Returns
        -------
        max_cliques : set
            A set of frozensets where each frozenset represents a maximum clique.
        execution_time : int
            Time taken to execute the clique-finding algorithm (in seconds).
        """
        def enumerate_by_sampling(sampling_method, epsilon, kappa):

            def enumerate_with_threshold(S, t, m, threshold):
                while True:
                    while t < ceil(m*log(m*kappa/epsilon)):
                        sample, score = sampling_method(threshold=threshold)
                        if score == threshold:
                            S.add(sample)
                            t += 1
                        else:
                            S = set()
                            S.add(sample)
                            t = 1
                            m = 2
                            return S, t, m, score
                    if len(S) < m:
                        return S, t, m, threshold
                    m += 1

            S = set()
            t = 0
            m = 1
            threshold = 0
            while True:
                S, t, m, score = enumerate_with_threshold(S, t, m, threshold)
                if score == threshold:
                    break
                else:
                    threshold = score
            return S

        sa_sampler = NealSampler()
        # Confidence of the max-clique enumeration: probability of missing a
        # maximum clique is <= epsilon per call. Tightened 1e-2 -> 1e-4
        # (Jul 2026): at 1e-2 roughly 1% of borderline reactions received
        # unstable verdicts across runs (measured ~1.5% on 200 controls).
        # Cost is ~2x more anneal confirmations per threshold.
        epsilon = 1e-4
        alpha = log(1 / epsilon) - 1
        beta = ((exp(-1) + (1 / 3) * log(1 / 3)) / (exp(-1) - 1 / 3)) * alpha
        inf_sum = zeta(2 * alpha) - sum([k ** (-2 * alpha) for k in range(1, 6)])
        r = exp(-alpha / (exp(1) - 1))
        kappa = (4 ** alpha) / (1 - exp(-beta)) * inf_sum + (2 - r) / ((1 - r) ** 2)
        sampler = IndependentSetSampler(sa_sampler, nx.complement(self.G))
        start = time.perf_counter_ns()
        cliques = enumerate_by_sampling(sampler.sample, epsilon, kappa)
        # Start time
        end = time.perf_counter_ns()
        execution_time = (end - start) * 1e-9  # Convert ns to seconds
        #return cliques, execution_time
        return [ list( c ) for c in cliques ], execution_time