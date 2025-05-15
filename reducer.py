###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2025: Amazon Web Services, Inc. - Contributions from JPMC
###############################################################################
"""reducer.py: Reducer for unweighted MIS instances."""

from enum import Enum
from itertools import combinations
import networkx as nx
from sortedcontainers import SortedList
import copy


class State(Enum):
    UNKNOWN_STATE = 0
    REMOVED = 1
    SELECTED = 2


class Candidate(object):
    """Readable and comparable pair of {degree, node_id}.

    These are sorted ascendingly by degree (first) and then node_id (second).
    """

    def __init__(self, degree, node):
        self.degree = degree
        self.node = node

    def __lt__(self, other):
        if self.degree == other.degree:
            return self.node < other.node
        return self.degree < other.degree

    def __eq__(self, other):
        return self.node == other.node

    def __repr__(self):
        return f"{self.node}({self.degree})"


class Reducer(object):
    """Class implementation to hold the graph and metadata while reducing."""

    def __init__(self, verbose=False, max_clique_size=None):
        """Just store settings internally."""
        self.verbose = verbose
        self.max_clique_size = max_clique_size

    def has_unresolved(self):
        """Are there clique candidates left to check (should we continue)?"""
        return len(self.unresolved) > 0

    def is_below_max_clique_size(self):
        """Of the candidates left, is the smallest below max_clique_size?"""
        if self.max_clique_size is None:
            return True
        return self.unresolved[0].degree < self.max_clique_size

    def reduce(self, graph: nx.Graph):
        """Actual implementation of the reduction logic (entry point).

        This *will* modify graph and return its reduced version.
        """

        # Initial setup: store the graph and create sorted lists for caching
        # what still needs to be checked and what can be removed.
        self.graph = graph
        self.unresolved = SortedList()
        self.states = {node: State.UNKNOWN_STATE for node in graph.nodes}

        # Remove nodes with a self-loop
        for node in list(graph.nodes):
            if node in graph.adj[node]:
                if self.verbose:
                    print(f"Removing {node} (self-loop)")
                graph.remove_node(node)
                self.states[node] = State.REMOVED

        # Initialize remaining nodes as unresolved in the cache
        for node in graph.nodes:
            self.unresolved.add(Candidate(len(graph.adj[node]), node))

        # Inner loop that identifies and removes exposed cliques
        while self.has_unresolved() and self.is_below_max_clique_size():
            candidate = self.unresolved.pop(0).node
            if self.is_exposed_clique(candidate):
                if self.verbose:
                    print(f"Removing {self.clique_str(candidate)}")
                self.remove_exposed_clique(candidate)
            elif self.verbose:
                print(f"Not a clique {self.clique_str(candidate)}")

        # Reduction has completed (either because we removed all reducible
        # parts or because the max_clique_size has been reached).
        removed = [n for n, s in self.states.items()
                   if s == State.REMOVED or s == State.SELECTED]
        selected = [n for n, s in self.states.items()
                    if s == State.SELECTED]
        return graph, removed, selected

    def remove_node(self, node):
        """Remove the designated node from the graph."""
        # Update and invalidate cache entries for all its neighbors
        for i in self.graph.adj[node]:
            k = len(self.graph.adj[i])
            self.unresolved.discard(Candidate(k, i))
            self.unresolved.add(Candidate(k - 1, i))

        # Mark the node itself as removed and erase it from the graph.
        k = len(self.graph.adj[node])
        self.unresolved.discard(Candidate(k, node))
        self.states[node] = State.REMOVED
        self.graph.remove_node(node)

    def remove_exposed_clique(self, corner):
        """Remove the exposed clique identified by this corner."""
        for i in list(self.graph.adj[corner]):
            self.remove_node(i)
        self.unresolved.discard(Candidate(0, corner))
        self.states[corner] = State.SELECTED  # mark as in the set
        self.graph.remove_node(corner)  # remove from graph

    def is_exposed_clique(self, corner) -> bool:
        """Check whether nodes form a clique."""
        # Pre-check for sufficient degrees -- O(N)
        neighbors = self.graph.adj[corner]
        for i in neighbors:
            if len(self.graph.adj[i]) < len(neighbors):
                return False
        # Check for pairwise connections -- O(N^2)
        for i, j in combinations(neighbors, 2):
            if j not in self.graph.adj[i]:
                return False
        return True

    def clique_str(self, node):
        """Show a clique candidate as a curly-braced list."""
        clique = [node] + list(self.graph.adj[node])
        clique = ",".join(str(x) for x in clique)
        return "{" + clique + "}"


def get_reduced_graph(graph, verbose=False):
    """Expose a compatible interface to utils_reduction.py."""
    graph_ = copy.deepcopy(graph)
    reducer = Reducer(verbose=verbose)
    graph_, removed, selected = reducer.reduce(graph_)
    return graph_, [], [[x] for x in selected]


def run_recursive_simplification(
        graph,
        mis_candidates_total=[],
        isolated_nodes_total=[],
        verbose=False):
    graph_ = copy.deepcopy(graph)
    reducer = Reducer(verbose=verbose)
    graph_, removed, selected = reducer.reduce(graph_)
    return graph_, [], [[x] for x in selected]


def run_simple_reduction(
        graph,
        isolated_total=[],
        dangling_total=[],
        verbose=False):
    graph_ = copy.deepcopy(graph)
    reducer = Reducer(verbose=verbose, max_clique_size=3)
    graph_, removed, selected = reducer.reduce(graph_)
    return graph_, [], [[x] for x in selected]


def run_reduction_cutoff(
        graph,
        cutoff,
        mis_candidates_total=[],
        isolated_nodes_total=[],
        seed=0,
        verbose=False):
    graph_ = copy.deepcopy(graph)
    reducer = Reducer(verbose=verbose, max_clique_size=cutoff)
    graph_, removed, selected = reducer.reduce(graph_)
    return graph_, [], [[x] for x in selected]


if __name__ == "__main__":
    graph = nx.Graph()
    nodes = list(range(1, 14))
    graph.add_nodes_from(nodes)
    edges = [(2, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 5), (4, 6), (4, 7), (5, 6), (5, 7),
             (6, 7), (6, 8), (7, 8), (7, 10), (9, 10), (10, 11), (10, 12), (11, 12), (12, 13)]
    graph.add_edges_from(edges)
    print(Reducer(verbose=True).reduce(graph))
