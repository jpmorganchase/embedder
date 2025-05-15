###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2025: Amazon Web Services, Inc. - Contributions from JPMC
###############################################################################
"""utils_checker.py: Checker logic for Rydberg hardware compatibility."""

import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix

def vertices_to_graph(vertices, radius=np.sqrt(2)):
    """
    Helper function to convert positions of vertices in 2D into a UD graph for given radius.
    Input:
        vertices: Positions of nodes (atoms) in 2D square lattice, given as list of tuples [(0,0), (0,1), ...].
        radius: Rydberg radius (defaults to \sqrt{2})
    Output:
        nx_graph: NetworkX OrderedGraph with UD connectivity
    """
    dmat = distance_matrix(vertices, vertices)
    adj = (dmat <= radius).astype(int) - np.eye(len(vertices))
    return nx.from_numpy_array(adj)


# helper function for finding max degree nodes 
def get_max_degree_nodes(graph): 
    """
    Helper function to find max degree nodes 
    Input:
        graph: networkx graph 
    Output:
        nodes_maxdegree: list of nodes with maximum degree
        max_degrees: list of maximum degrees
    """
    # get degree centrality
    centrality = nx.degree_centrality(graph)
    # get max degree
    max_centrality = max(centrality.values())
    # find nodes with max degree
    nodes_maxdegree = [node for node, value in centrality.items() if value == max_centrality]
    # get max degrees 
    max_degrees = [graph.degree[node] for node in nodes_maxdegree]

    return nodes_maxdegree, max_degrees


# dictionary with possible number of triangles for given max degree as key
dic_triangle_check = {'dmax_8': {12}, 
                      'dmax_7': {8, 10}, 
                      'dmax_6': {4, 5, 6, 7, 8}, 
                      'dmax_5': {2, 3, 4, 5, 6},
                      'dmax_4': {0, 1, 2, 3, 4}
                      }

def check_compatibility(graph):
    """
    Helper function to perfrom simple compatibility check with 
    Input:
        graph: networkx graph 
    Output:
        check: Boolean variable indicating whether pass was checked
    """
    # set check variable
    check = True
    # find max degree nodes
    nodes_maxdegree, max_degrees = get_max_degree_nodes(graph)
    print('Maximum degrees found:', max_degrees)
    # degree larger than 8 not compatible with UJ QPU
    if max(max_degrees) > 8:
        check = False 
    # perfrom triangle check  
    elif max(max_degrees) in {4, 5, 6, 7, 8}: 
        # look up valid set of triangles
        valid_set = dic_triangle_check['dmax_'+str(max(max_degrees))]
        # loop over all max degree nodes
        for node in nodes_maxdegree: 
            # get (local) number of triangles 
            number_triangles = nx.triangles(graph, node)
            print('Number of triangles found:', number_triangles)
            if number_triangles not in valid_set:
                check = False 

    return check 